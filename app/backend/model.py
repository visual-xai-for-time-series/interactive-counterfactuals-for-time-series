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


class ModelsInMemory:
    selected_model = None

    models_in_memory = {}

    def __init__(self, selected_model='resnet-ecg5000'):
        files = find_files(directory_to_search, substrings_to_find, endings_to_find)
        for base_name, file_path in files.items():
            if isinstance(file_path, list):
                for file_name in file_path:
                    if file_name is not None:
                        file_base_name = file_name[:-3]
                        if os.path.exists(f'{file_base_name}-extracted'):
                            self.models_in_memory[file_base_name] = torch.load(file_name)

        print(f'Found models: {list(self.models_in_memory.keys())}')

        self.change_model(selected_model)


    def __call__(self, selected_model=None):
        if selected_model in self.models_in_memory:
            self.selected_model = selected_model
        return self.models_in_memory[self.selected_model]


    def change_model(self, selected_model='resnet-ecg5000'):
        if selected_model is None:
            self.selected_model = list(self.models_in_memory.keys())[0]
        else:
            self.selected_model = [x for x in self.models_in_memory.keys() if selected_model in x][0]
        print(f'Selected model: {self.selected_model}')


    def get_possible_models(self):
        return list(self.models_in_memory.keys())


models_in_memory = ModelsInMemory()


def predict_and_get_activations_from_model(data):
    model = models_in_memory()
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
    activation_handle = layer_to_look_at.register_forward_hook(get_activation('new'))

    data = torch.from_numpy(data).reshape(1, 1, -1).float().to(device)
    predictions = torch.argmax(model(data).detach().cpu(), axis=1).numpy().tolist()

    activation_handle.remove()

    return activations['new'], predictions


def predict_and_get_attributions_from_model(data, attribution=None, baseline_samples=10):
    model = models_in_memory()
    model.eval()

    data = torch.from_numpy(data).reshape(1, 1, -1).float().to(device)
    shape = list(data.shape)
    data.requires_grad_(True)

    predictions = torch.argmax(model(data), axis=1)

    attribution_data = get_attributions_data()['data']
    if attribution is None:
        attribution = list(attribution_data.keys())[0]
    attribution_technqiue = name_to_class(attribution)

    dataset = get_time_series_data()['data'][0]
    random_samples = np.random.randint(0, shape[0], size=baseline_samples)
    baselines = torch.from_numpy(dataset[random_samples]).reshape(-1, *shape[1:]).float().to(device)

    attribution_technqiue_called = attribution_technqiue(model)
    attributions = attribution_technqiue_called.attribute(data, target=predictions, baselines=baselines)
    attributions = attributions.detach().cpu().reshape(1, -1).numpy()

    predictions = predictions.detach().cpu().numpy().tolist()

    return attributions, predictions


def predict_using_model(data):
    model = models_in_memory()
    model.eval()

    data = torch.from_numpy(data).reshape(1, 1, -1).float().to(device)
    model_predictions = model(data).detach().cpu()
    predictions = torch.argmax(model_predictions, axis=1).numpy().tolist()

    return predictions


def generate_time_series_from_activation(data, steps=10):
    model = models_in_memory()
    model.eval()

    cur_activations = {}
    def get_activation(dict_to_save, name_to_save):
        def hook(model, input, output):
            output_transformed = output
            if isinstance(model, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                output_transformed = torch.amax(output, dim=1)
            dict_to_save[name_to_save] = output_transformed
        return hook

    possible_layers_to_look_at = get_last_layer(model)
    layer_to_look_at = possible_layers_to_look_at[-2]
    hook_callback = get_activation(cur_activations, 'layer_to_look_at')
    activation_handle = layer_to_look_at.register_forward_hook(hook_callback)

    ts_base = get_time_series_data()['data']
    act_base = np.array(get_activations_data()['data'])

    ts_values = ts_base[0]

    border_max = np.max(ts_values, axis=0)
    border_min = np.min(ts_values, axis=0)

    k_candidates = 1
    nbrs = NearestNeighbors(n_neighbors=k_candidates, algorithm='auto').fit(act_base)
    _, indices = nbrs.kneighbors(data)

    ts_candidates = np.array(ts_values)[indices[0]]
    ts_candidate = np.mean(ts_candidates, axis=0)

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
        # print(f'loss: {loss}')
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
    activation_handle.remove()

    return ts_candidate, predictions


def generate_time_series_from_attribution(data, attribution=None, baseline_samples=10, steps=10, k_candidates=10):
    model = models_in_memory()
    model.eval()

    attribution_data = get_attributions_data()['data']
    if attribution is None:
        attribution = list(attribution_data.keys())[0]
    attribution_technqiue = name_to_class(attribution)

    ts_base = get_time_series_data()['data']
    att_base = np.array(attribution_data[attribution])

    ts_values = ts_base[0]
    shape = list(ts_values.shape)

    dataset = ts_values
    random_samples = np.random.randint(0, shape[0], size=baseline_samples)
    baselines = torch.from_numpy(dataset[random_samples]).reshape(-1, 1, shape[-1]).float().to(device)

    border_max = np.max(ts_values, axis=0)
    border_min = np.min(ts_values, axis=0)

    nbrs = NearestNeighbors(n_neighbors=k_candidates, algorithm='auto').fit(att_base)
    _, indices = nbrs.kneighbors(data)

    ts_candidates = np.array(ts_values)[indices[0]]
    ts_candidate = np.mean(ts_candidates, axis=0)

    border_max = torch.from_numpy(border_max).float().to(device)
    border_min = torch.from_numpy(border_min).float().to(device)

    alpha = torch.tensor(0.5)
    if border_max is not None and border_min is not None:
        alpha = (border_max - border_min).reshape(-1) / steps

    data = torch.from_numpy(data).float().to(device)
    ts_candidate = torch.from_numpy(ts_candidate).float().to(device)

    best_solution = [1000000, None]

    attribution_tech = attribution_technqiue(model)

    criterion = nn.MSELoss()
    for x in range(steps):
        ts_candidate.requires_grad_(True)
        ts_candidate.retain_grad()

        predictions = model(ts_candidate.reshape(1, 1, -1))
        predictions = torch.argmax(predictions, axis=1)

        cur_attribution = attribution_tech.attribute(ts_candidate.reshape(1, 1, -1), target=predictions, baselines=baselines)

        loss = criterion(cur_attribution, data)
        # print(f'loss: {loss}')
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

    return ts_candidate, predictions
