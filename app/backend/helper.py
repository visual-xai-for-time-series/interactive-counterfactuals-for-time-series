import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from captum.attr import Saliency, InputXGradient, IntegratedGradients, GuidedBackprop
from captum.attr import DeepLift, LRP
from captum.attr import ShapleyValueSampling, GradientShap, DeepLiftShap
from captum.attr import Occlusion, FeaturePermutation, Lime, KernelShap

import numpy as np

color_scales = {
    'Dark2': np.array([[0.10588235294117647, 0.6196078431372549, 0.4666666666666667], [0.8509803921568627, 0.37254901960784315, 0.00784313725490196], [0.4588235294117647, 0.4392156862745098, 0.7019607843137254], [0.9058823529411765, 0.1607843137254902, 0.5411764705882353], [0.4, 0.6509803921568628, 0.11764705882352941], [0.9019607843137255, 0.6705882352941176, 0.00784313725490196], [0.6509803921568628, 0.4627450980392157, 0.11372549019607843], [0.4, 0.4, 0.4]])
}


def find_files(directory, substrings, endings):
    # Dictionary to store matching file paths
    matching_files = {substring: [] for substring in substrings}

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        # Check each file in the current directory
        for filename in files:
            # Check if the file name matches any of the conditions
            for substring, ending in zip(substrings, endings):
                if filename.endswith(ending) and substring in filename:
                    # If the conditions are met, add the file path to the dictionary
                    # if matching_files[substring] is None:
                    matching_files[substring].append(os.path.join(root, filename))
                    break  # Break the inner loop to prevent duplicates

    return matching_files


def moving_average_torch(data, window_size):
    """
    Apply moving average smoothing to a time series data using PyTorch.

    Parameters:
    - data: 1D torch tensor, the input time series data.
    - window_size: int, size of the moving window.

    Returns:
    - smoothed_data: 1D torch tensor, the smoothed time series data.
    """
    # Define the kernel for the moving average
    kernel = torch.ones(window_size) / window_size

    # Pad the data at the beginning and end to handle edge cases
    pad_width = window_size // 2

    # Perform 1D convolution with the moving average kernel
    smoothed_data = F.conv1d(data.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding='same')

    return smoothed_data.flatten()


def color_mixer(data, color_map='Dark2'):
    coordinate_data, prediction_data = data

    # Check if at least one value exceeds 0.8 along the last axis
    max_value = np.max(prediction_data, axis=-1)
    indices_to_set = np.where(max_value > 0.75)

    # Set values to 1.0 where the condition is met, otherwise keep them unchanged
    for i in indices_to_set[0]:
        idx = np.argmax(prediction_data[i])
        prediction_data[i] = 0.0
        prediction_data[i, idx] = 1.0

    cmap = color_scales[color_map]

    shape = list(prediction_data.shape)
    color_data = np.full((shape[0], 3), 0)
    for i in range(shape[0]):
        col = prediction_data[i, :]
        color = np.sum([np.array(cmap[i]) * c for i, c in enumerate(col)], axis=0)
        color_data[i, :] = (color * 255).astype(np.uint8)

    return coordinate_data, color_data


def argmax(data, axis=1):
    return np.argmax(data, axis=axis)


def name_to_class(name='DeepLift'):
    if name == 'Saliency':
        return Saliency
    elif name == 'InputXGradient':
        return InputXGradient
    elif name == 'IntegratedGradients':
        return IntegratedGradients
    elif name == 'GuidedBackprop':
        return GuidedBackprop
    elif name == 'DeepLift':
        return DeepLift
    elif name == 'LRP':
        return LRP
    elif name == 'ShapleyValueSampling':
        return ShapleyValueSampling
    elif name == 'GradientShap':
        return GradientShap
    elif name == 'DeepLiftShap':
        return DeepLiftShap
    elif name == 'Occlusion':
        return Occlusion
    elif name == 'FeaturePermutation':
        return FeaturePermutation
    elif name == 'Lime':
        return Lime
    elif name == 'KernelShap':
        return KernelShap
    return None


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
