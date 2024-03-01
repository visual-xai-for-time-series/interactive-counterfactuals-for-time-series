import os

import numpy as np

import torch
import torch.nn as nn

from captum.attr import GradientShap, IntegratedGradients, ShapleyValueSampling, Saliency, DeepLift

from sklearn.neighbors import NearestNeighbors

from helper import *
from data import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


directory_to_search = '/data/'
substrings_to_find = ['resnet', 'cnn']
endings_to_find = ['pt', 'pt']

files = find_files(directory_to_search, substrings_to_find, endings_to_find)

models_in_memory = {}
def load_models():
    print(files)
    for file in files:
        print(file)
        if files[file] is not None and os.path.exists(f'{files[file][:-3]}-extracted'):
            models_in_memory[file] = torch.load(files[file])

load_models()

def predict_and_get_activations_from_model(data, model_name='resnet'):
    model = models_in_memory[model_name]
    model.eval()

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if name not in activations:
                activations[name] = []
            output_transformed = output
            if isinstance(model, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                output_transformed = torch.amax(output, dim=1)
            data = output_transformed.detach().cpu().numpy().tolist()
            activations[name].extend(data)
        return hook

    def get_last_layer(model):
        possible_layers = []
        for layer in model.children():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                possible_layers.append(layer)
            elif isinstance(layer, nn.Sequential):
                possible_layers.extend(get_last_layer(layer))
            elif isinstance(layer, nn.Module):
                possible_layers.extend(get_last_layer(layer))
        return possible_layers

    possible_layers_to_look_at = get_last_layer(model)
    layer_to_look_at = possible_layers_to_look_at[-2]
    fc1_handle = layer_to_look_at.register_forward_hook(get_activation('new'))

    data = torch.from_numpy(data).reshape(1, 1, -1).float().to(device)
    predictions = torch.argmax(model(data).detach().cpu(), axis=1).numpy().tolist()

    fc1_handle.remove()

    return activations['new'], predictions


def predict_and_get_attributions_from_model(data, model_name='resnet'):
    model = models_in_memory[model_name]
    model.eval()

    data = torch.from_numpy(data).reshape(1, 1, -1).float().to(device)
    data.requires_grad_(True)

    predictions = torch.argmax(model(data), axis=1)

    explainer = DeepLift(model)
    attribution = explainer.attribute(data, target=predictions)
    attribution = attribution.detach().cpu().reshape(1, -1).numpy()

    predictions = predictions.detach().cpu().numpy().tolist()

    return attribution, predictions


def predict_using_model(data, model_name='resnet'):
    model = models_in_memory[model_name]
    model.eval()

    data = torch.from_numpy(data).reshape(1, 1, -1).float().to(device)
    model_predictions = model(data).detach().cpu()
    predictions = torch.argmax(model_predictions, axis=1).numpy().tolist()

    return predictions


def generate_time_series_from_activation(data, model_name='resnet', steps=10):
    model = models_in_memory[model_name]
    model.eval()

    cur_activations = {}
    def get_activation(dict_to_save, name_to_save):
        def hook(model, input, output):
            output_transformed = output
            if isinstance(model, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                output_transformed = torch.amax(output, dim=1)
            dict_to_save[name_to_save] = output_transformed
        return hook

    def get_last_layer(model):
        possible_layers = []
        for layer in model.children():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                possible_layers.append(layer)
            elif isinstance(layer, nn.Sequential):
                possible_layers.extend(get_last_layer(layer))
            elif isinstance(layer, nn.Module):
                possible_layers.extend(get_last_layer(layer))
        return possible_layers

    possible_layers_to_look_at = get_last_layer(model)
    layer_to_look_at = possible_layers_to_look_at[-2]
    fc1_handle = layer_to_look_at.register_forward_hook(get_activation(cur_activations, 'layer_to_look_at'))

    ts_base = get_time_series_data()['data']
    act_base = np.array(get_activations_data()['data'])

    ts_values = ts_base[0]

    print(ts_values.shape)

    border_max = np.max(ts_values, axis=0)
    border_min = np.min(ts_values, axis=0)

    k_candidates = 1
    nbrs = NearestNeighbors(n_neighbors=k_candidates, algorithm='auto').fit(act_base)
    _, indices = nbrs.kneighbors(data)

    ts_candidates = np.array(ts_values)[indices[0]]
    ts_candidate = np.mean(ts_candidates, axis=0)
    # ts_candidate = np.random.default_rng().uniform(border_min, border_max, border_max.shape)

    border_max = torch.from_numpy(border_max).float().to(device)
    border_min = torch.from_numpy(border_min).float().to(device)

    alpha = torch.tensor(0.5)
    if border_max is not None and border_min is not None:
        alpha = (border_max - border_min).reshape(-1) / steps

    criterion = nn.MSELoss()
    data = torch.from_numpy(data).float().to(device)

    ts_candidate = torch.from_numpy(ts_candidate).float().to(device)

    best_solution = [1000000, None]

    for x in range(steps):
        ts_candidate.requires_grad_(True)
        ts_candidate.retain_grad()

        predictions = model(ts_candidate.reshape(1, 1, -1))
        cur_activation = cur_activations['layer_to_look_at'].reshape(data.shape)

        loss = criterion(cur_activation, data)
        print(f'loss: {loss}')
        if best_solution[0] > loss:
            best_solution = [loss, ts_candidate]
        loss = loss * -1 # fix for gradient ascent
        loss.backward(retain_graph=True)

        ts_candidate_grad = ts_candidate.grad.reshape(-1)
        ts_candidate = torch.add(
            ts_candidate.reshape(-1),
            torch.mul(
                ts_candidate_grad,
                alpha),
            )

        # Regularization:
        with torch.no_grad():
            # Clamp to borders
            ts_candidate = torch.clamp(ts_candidate, border_min, border_max)

            # Random addition
            # ts_candidate = ts_candidate.reshape(-1) * (0.2 * torch.rand(ts_candidate.reshape(-1).shape[0]) + 0.80)

            # Smooth time series
            # ts_candidate = moving_average_torch(ts_candidate.reshape(-1), 3)

    ts_candidate = best_solution[1]
    predictions = torch.argmax(model(ts_candidate.reshape(1, 1, -1)).detach().cpu(), axis=1).numpy().tolist()
    ts_candidate = ts_candidate.detach().cpu().reshape(1, -1).numpy().tolist()
    fc1_handle.remove()

    return ts_candidate, predictions


def generate_time_series_from_attribution(data, model_name='resnet'):
    model = models_in_memory[model_name]


def generate_activation_maximization_paper(class_label, model_name='resnet'):
    # https://www.biorxiv.org/content/biorxiv/early/2021/10/12/2021.10.10.463830.full.pdf

    # Steps:
    # 1. Find class prototype by using FFT on each sample of a class, take median, and convert it back
    # 2. Get activation for sample
    # 3. FFT values pertrubed by 0.001, converted back, and activations are extracted to calculcate difference
    # 4. Change and original perturbation size are used as gradient
    # 5. Modify FFT values by gradient and predefined step size
    # 6. Repeat 2 to 6 for a number and take largest activation sample
    # 7. To this for varying step size values
    # In general, clamp by FFT borders

    model = models_in_memory[model_name]


def generate_activation_maximization_own(class_label, model_name='resnet'):
    model = models_in_memory[model_name]
