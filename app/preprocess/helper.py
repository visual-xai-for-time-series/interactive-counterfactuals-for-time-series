import torch
import torch.nn as nn

from captum.attr import GradientShap, IntegratedGradients, ShapleyValueSampling, Saliency, DeepLift

from sklearn.neighbors import NearestNeighbors

import numpy as np


def get_device_from_model(model):
    return next(model.parameters()).device


def get_possible_layer(model):
    possible_layers = []
    for layer in model.children():
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
            possible_layers.append(layer)
        elif isinstance(layer, nn.Sequential):
            possible_layers.extend(get_possible_layer(layer))
        elif isinstance(layer, nn.Module):
            possible_layers.extend(get_possible_layer(layer))
    return possible_layers


def get_activation(dict_to_save, name_to_save):
    def hook(model, input, output):
        output_transformed = output
        if isinstance(model, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
            output_transformed = torch.amax(output, dim=1)
        dict_to_save[name_to_save] = output_transformed
    return hook


def get_activations(dict_to_save, name_to_save):
    def hook(model, input, output):
        output_transformed = output
        if isinstance(model, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
            output_transformed = torch.amax(output, dim=1)
        if name_to_save not in dict_to_save:
            dict_to_save[name_to_save] = []
        dict_to_save[name_to_save].extend(output_transformed.detach().cpu().numpy().tolist())
    return hook


def get_time_series_from_activations(model, activation, time_series_shape, layer=-1, activations=None, time_series=None, k_candidates=10, steps=10):
    device = get_device_from_model(model)

    model.eval()

    cur_activations = {}
    possible_layers_to_look_at = get_possible_layer(model)
    layer_to_look_at = possible_layers_to_look_at[layer]
    activation_hook = get_activation(cur_activations, 'layer_to_look_at')
    activation_handle = layer_to_look_at.register_forward_hook(activation_hook)

    border_max = None
    border_min = None

    ts_candidate = torch.zeros(time_series_shape).float().to(device)
    if activations is not None and time_series is not None:
        activations_shape = activations.shape
        activations = activations.reshape(activations_shape[0], -1)
        nbrs = NearestNeighbors(n_neighbors=k_candidates, algorithm='auto').fit(activations)
        _, indices = nbrs.kneighbors(activation.reshape(1, -1))

        ts_candidates = torch.from_numpy(time_series[indices[0]]).float().to(device)
        ts_candidate = torch.mean(ts_candidates, dim=0)

        border_max = np.max(time_series, axis=0)
        border_min = np.min(time_series, axis=0)

        border_max = torch.from_numpy(border_max).float().to(device)
        border_min = torch.from_numpy(border_min).float().to(device)

    best_solution = [1000000, None]

    alpha = torch.tensor(0.5)
    if border_max is not None and border_min is not None:
        alpha = (border_max - border_min).reshape(-1) / steps

    criterion = nn.MSELoss()
    for _ in range(steps):
        ts_candidate.requires_grad_(True)
        ts_candidate.retain_grad()

        predictions = model(ts_candidate.reshape(time_series_shape))
        cur_activation = cur_activations['layer_to_look_at'].flatten()

        loss = criterion(cur_activation, activation)
        if best_solution[0] > loss:
            best_solution = [loss, ts_candidate]
        # print(f'loss: {loss}')
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
            if border_max is not None and border_min is not None:
                # Clamp to borders
                ts_candidate = torch.clamp(ts_candidate, border_min, border_max)

    activation_handle.remove()

    ts_candidate = best_solution[1]

    predictions = model(ts_candidate.reshape(time_series_shape)).detach().cpu()
    predictions = predictions.reshape(-1).numpy().tolist()

    ts_candidate = ts_candidate.detach().cpu().reshape(-1).numpy().tolist()

    return ts_candidate, predictions


def get_time_series_from_attributions(model, attribution, time_series_shape, attribution_technqiue=DeepLift, attributions=None, time_series=None, k_candidates=10, steps=10):
    device = get_device_from_model(model)

    model.eval()

    ts_candidate = torch.zeros(time_series_shape).float().to(device)
    if attributions is not None and time_series is not None:
        attributions_shape = attributions.shape
        attributions = attributions.reshape(attributions_shape[0], -1)
        nbrs = NearestNeighbors(n_neighbors=k_candidates, algorithm='auto').fit(attributions)
        _, indices = nbrs.kneighbors(attribution.reshape(1, -1))

        ts_candidates = torch.from_numpy(time_series[indices[0]]).float().to(device)
        ts_candidate = torch.mean(ts_candidates, dim=0)

        border_max = np.max(time_series, axis=0)
        border_min = np.min(time_series, axis=0)

        border_max = torch.from_numpy(border_max).float().to(device)
        border_min = torch.from_numpy(border_min).float().to(device)

    best_solution = [1000000, None]

    alpha = torch.tensor(0.5)
    if border_max is not None and border_min is not None:
        alpha = (border_max - border_min).reshape(-1) / steps

    attribution_technqiue, attribution_technqiue_kwargs = attribution_technqiue
    attribution_tech = attribution_technqiue(model)

    criterion = nn.MSELoss()
    for _ in range(steps):
        ts_candidate.requires_grad_(True)
        ts_candidate.retain_grad()

        predictions = model(ts_candidate.reshape(time_series_shape))
        predictions = torch.argmax(predictions, axis=1)

        cur_attribution = attribution_tech.attribute(ts_candidate.reshape(time_series_shape), target=predictions, **attribution_technqiue_kwargs)

        loss = criterion(cur_attribution.flatten(), attribution.flatten())
        if best_solution[0] > loss:
            best_solution = [loss, ts_candidate]
        # print(f'loss: {loss}')
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
            if border_max is not None and border_min is not None:
                # Clamp to borders
                ts_candidate = torch.clamp(ts_candidate, border_min, border_max)

    ts_candidate = best_solution[1]

    predictions = model(ts_candidate.reshape(time_series_shape)).detach().cpu()
    predictions = predictions.reshape(-1).numpy().tolist()

    ts_candidate = ts_candidate.detach().cpu().reshape(-1).numpy().tolist()

    return ts_candidate, predictions

