#!/usr/bin/env python

import os
import sys
import dill
import time
import shutil
import random
import argparse

import numpy as np
import pandas as pd

import umap

import json
from json import JSONEncoder

from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from captum.attr import Saliency, InputXGradient, IntegratedGradients, GradientShap
from captum.attr import ShapleyValueSampling, DeepLift, DeepLiftShap

from sklearn.preprocessing import OneHotEncoder

from sktime.datasets import load_UCR_UEA_dataset

from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor


random_seed = 13

torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models import *
from helper import *


class TimeSeriesDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        inputs = self.X[idx]
        label = self.y[idx]

        return inputs, label


class NumpyArrayEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return JSONEncoder.default(self, o)


def inverse_transform_chunk(model, low_dim_data):
    return model.inverse_transform(low_dim_data)


def parallel_inverse_transform(model, reduced_data, num_processes=4):
    # Calculate the size of each chunk of data
    chunk_size = len(reduced_data) // num_processes
    data_chunks = [reduced_data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
    
    # Using ProcessPoolExecutor to parallelize tasks
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(inverse_transform_chunk, model, chunk) for chunk in data_chunks]
        results = [future.result() for future in futures]
    return np.vstack(results)


def main():
    overall_time = time.process_time()

    parser = argparse.ArgumentParser(description='Calculate orderings for a model for a selected dataset.')

    # Add arguments
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        default='FordA', help='Specify the dataset (e.g., FordA)')
    parser.add_argument('--model', '-m', type=str, required=True, 
                        choices=['cnn', 'resnet'], 
                        help='Specify the model type (choose from: cnn, resnet)')
    parser.add_argument('--base_path', '-p', type=str, default='/data/', 
                        help='Path to save/load the model (default: data/)')
    parser.add_argument('---create-new', '-cn', type=str, default='0', 
                        help='Create a new file even if an old exists in /new/ (default: 0)')

    print('Setting the stage')

    # Parse the arguments
    args = parser.parse_args()

    dataset = args.dataset
    model_type = args.model
    base_data_path = args.base_path
    create_new = args.create_new

    print(f'for {dataset}')

    ######## Set directories

    model_base_name = f'{model_type.lower()}-{dataset.lower()}'
    model_file = f'{model_base_name}.pt'

    model_path = os.path.join(base_data_path, model_file)

    extracted_data_path = os.path.join(base_data_path, f'{model_base_name}-extracted')
    if bool(create_new):
        print('Create new')
        extracted_data_path = os.path.join(extracted_data_path, 'new')
        if os.path.exists(extracted_data_path) and os.path.isdir(extracted_data_path):
            shutil.rmtree(extracted_data_path)

    os.makedirs(base_data_path, exist_ok=True)
    os.makedirs(extracted_data_path, exist_ok=True)

    ######## Get the data

    print('Getting the data')

    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='train', return_type='numpyflat')
    X_test, y_test = load_UCR_UEA_dataset(name=dataset, split='test', return_type='numpyflat')

    print(f'Length training data: {len(X_train)} labels: {len(y_train)} test data: {len(X_test)} labels: {len(y_test)}')

    encoder = OneHotEncoder(categories='auto', sparse_output=False)

    y_train_ohe = encoder.fit_transform(np.expand_dims(y_train, axis=-1))
    y_test_ohe = encoder.transform(np.expand_dims(y_test, axis=-1))

    y_train_norm = y_train_ohe.argmax(axis=-1)
    y_test_norm = y_test_ohe.argmax(axis=-1)

    labels_nr = len(encoder.categories_[0])

    dataset_train = TimeSeriesDataset(X_train, y_train_ohe)
    dataset_test = TimeSeriesDataset(X_test, y_test_ohe)

    dataloader_train = DataLoader(dataset_train, batch_size=120, shuffle=False)
    dataloader_train_not_shuffled = DataLoader(dataset_train, batch_size=120, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=120, shuffle=False)

    ######## Load model

    print('Loading the model')

    model = torch.load(model_path, map_location=device)
    model.eval()

    ######## Get model accuracy on data

    preds = []
    labels = []
    for x in dataloader_train_not_shuffled:
        input_, label_ = x
        input_ = input_.reshape(input_.shape[0], 1, -1)
        input_ = input_.float().to(device)
        label_ = label_.float().to(device)

        pred_ = model(input_)
        preds.extend(pred_)
        labels.extend(label_)

    preds = torch.stack(preds)
    labels = torch.stack(labels)
    print('Prediction Accuracy Train', np.round((preds.argmax(dim=-1) == labels.argmax(dim=-1)).int().sum().float().item() / len(preds), 4))

    y_train_pred = preds.cpu().detach().numpy().round(3)
    
    model.eval()

    preds = []
    labels = []
    for x in dataloader_test:
        input_, label_ = x
        input_ = input_.reshape(input_.shape[0], 1, -1)
        input_ = input_.float().to(device)
        label_ = label_.float().to(device)

        pred_ = model(input_)
        preds.extend(pred_)
        labels.extend(label_)

    preds = torch.stack(preds)
    labels = torch.stack(labels)
    print('Prediction Accuracy Test', np.round((preds.argmax(dim=-1) == labels.argmax(dim=-1)).int().sum().float().item() / len(preds), 4))

    y_test_pred = preds.cpu().detach().numpy().round(3)

    data_path = os.path.join(extracted_data_path, f'{model_base_name.lower()}-data.pkl')
    if not os.path.exists(data_path):

        data_to_save = {
            'train': (X_train, y_train_norm.astype(int), y_train_pred),
            'test': (X_test, y_test_norm.astype(int), y_test_pred),
        }

        with open(data_path, 'wb') as file:
            dill.dump(data_to_save, file)

    ######## Get activations

    print('Getting the activation')

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
    possible_layers_to_look_at = get_possible_layer(model)
    layer_to_look_at = possible_layers_to_look_at[-2]

    activations_path = os.path.join(extracted_data_path, f'{model_base_name.lower()}-activations.pkl')
    if os.path.exists(activations_path):
        with open(activations_path, 'rb') as file:
            activations = dill.load(file)
    else:
        activations = {}
        model.eval()

        print(f'Layer to look at the activations: {layer_to_look_at}')
        activation_handle = layer_to_look_at.register_forward_hook(get_activations(activations, 'train'))

        for idx, (inputs, labels) in enumerate(dataloader_train_not_shuffled):
            inputs = inputs.reshape(inputs.shape[0], 1, -1)
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            preds = model(inputs)

        activation_handle.remove()

        activation_handle = layer_to_look_at.register_forward_hook(get_activations(activations, 'test'))

        for idx, (inputs, labels) in enumerate(dataloader_test):
            inputs = inputs.reshape(inputs.shape[0], 1, -1)
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            preds = model(inputs)

        activation_handle.remove()

        with open(activations_path, 'wb') as file:
            dill.dump(activations, file)

    activations_train = np.array(activations['train'])
    activations_test = np.array(activations['test'])
    print(f'Length of train activations: {activations_train.shape}')
    print(f'Length of test activations: {activations_test.shape}')

    ######## Get attributions

    print('Getting the attributions')

    sample, label = dataset_train[0]
    shape = sample.reshape(1, -1).shape

    torch.manual_seed(random_seed)
    baselines = torch.from_numpy(np.array([dataset_train[torch.randint(len(dataset_train), (1,))][0] for _ in range(10)])).reshape(-1, *shape).float().to(device)

    attribution_techniques = [
        # ['LRP', LRP],
        # ['Saliency', Saliency, {}],
        # ['InputXGradient', InputXGradient, {}],
        ['DeepLift', DeepLift, {}],
        ['DeepLiftShap', DeepLiftShap, {'baselines': baselines}],
        # ['IntegratedGradients', IntegratedGradients, {}],
        # ['GradientShap', GradientShap, {'baselines': baselines}],
        # ['ShapleyValueSampling', ShapleyValueSampling, {}],
        # ['Occlusion', Occlusion, {'sliding_window_shapes': (1, 5)}],
    ]
    attribution_techniques_dict = {k: [v, vv] for k, v, vv in attribution_techniques}

    attributions_path = os.path.join(extracted_data_path, f'{model_base_name.lower()}-attributions.pkl')
    if os.path.exists(attributions_path):
        with open(attributions_path, 'rb') as file:
            attributions = dill.load(file)

        attribution_techniques = [[x, None] for x in attributions['train'].keys()]
    else:
        model.eval()

        attributions = {'train': {}, 'test': {}}

        for at in attribution_techniques:
        
            at_name, at_function, at_kwargs = at
            attribute_tec = at_function(model)

            print(f'Calculcate: {at_name}')
            print(f'Start with train')

            attributions_tmp = []
            for x in dataloader_train_not_shuffled:
                input_, label_ = x
                input_ = input_.reshape(input_.shape[0], 1, -1)
                input_ = input_.float().to(device)
                label_ = label_.float().to(device)

                attribution = attribute_tec.attribute(input_.reshape(-1, *shape), target=torch.argmax(label_, axis=1), **at_kwargs)
                attributions_tmp.extend(attribution)

            attributions_tmp = torch.stack(attributions_tmp)
            attributions['train'][at_name] = attributions_tmp.detach().cpu().reshape(-1, shape[-1]).numpy()
            del attributions_tmp

            print(f'Start with test')

            attributions_tmp = []
            for x in dataloader_test:
                input_, label_ = x
                input_ = input_.reshape(input_.shape[0], 1, -1)
                input_ = input_.float().to(device)
                label_ = label_.float().to(device)

                attribution = attribute_tec.attribute(input_.reshape(-1, *shape), target=torch.argmax(label_, axis=1), **at_kwargs)
                attributions_tmp.extend(attribution)

            attributions_tmp = torch.stack(attributions_tmp)
            attributions['test'][at_name] = attributions_tmp.detach().cpu().reshape(-1, shape[-1]).numpy()
            del attributions_tmp

        with open(attributions_path, 'wb') as file:
            dill.dump(attributions, file)

    attributions_train = attributions['train']
    attributions_test = attributions['test']

    print(f'Length of train attributions: {len(attributions_train)} - {attributions_train.keys()}')
    print(f'Length of test attributions: {len(attributions_test)} - {attributions_test.keys()}')

    ######## Base Data

    base_data_for_maps = {
        'data': {
            'train': X_train,
            'test': X_test
        },
        'activations': {
            'train': activations_train,
            'test': activations_test
        },
        'attributions': {
            'train': attributions_train,
            'test': attributions_test
        },
    }

    ######## Projections

    mappers_path = os.path.join(extracted_data_path, f'{model_base_name.lower()}-mappers.pkl')
    if os.path.exists(mappers_path):
        with open(mappers_path, 'rb') as file:
            mappings_to_create = dill.load(file)
    else:

        print('Start generating projections')

        mappings_to_create = {
            'data': {
                'train': None,
                'test': None
            },
            'activations': {
                'train': None,
                'test': None
            },
            'attributions': {
                'train': None,
                'test': None
            },
        }


        def project(data):
            mapper = umap.UMAP(n_components=2, random_state=random_seed)
            projected_data = mapper.fit_transform(data)
            return mapper, projected_data


        for map_to_create in mappings_to_create:
            print(map_to_create)
            for stage in mappings_to_create[map_to_create]:
                data = base_data_for_maps[map_to_create][stage]
                if isinstance(data, dict):
                    tmp = {}
                    for d in data:
                        tmp[d] = project(data[d])
                else:
                    tmp = project(data)

                mappings_to_create[map_to_create][stage] = tmp


        with open(mappers_path, 'wb') as file:
            dill.dump(mappings_to_create, file)

    print('Done with generating projections')

    ######## Density Map

    density_path = os.path.join(extracted_data_path, f'{model_base_name.lower()}-density.pkl')
    if os.path.exists(density_path):
        with open(density_path, 'rb') as file:
            density_map = dill.load(file)
    else:

        print('Start with generating density maps')

        density_map = {
            'data': {
                'train': None,
                'test': None
            },
            'activations': {
                'train': None,
                'test': None
            },
            'attributions': {
                'train': {},
                'test': {}
            },
        }


        def create_decision_border_dense_map_for_data(data, mapper, projected_data=None, sample_nr=250):
            # Generate augmented data using mapper
            if projected_data is None:
                projected_data = mapper.transform(data)
            projected_data_max = np.max(projected_data, axis=0)
            projected_data_min = np.min(projected_data, axis=0)
            projected_data_pad = (projected_data_max - projected_data_min) * 0.05
            projected_data_max += projected_data_pad
            projected_data_min -= projected_data_pad
            dim_x = np.linspace(projected_data_min[0], projected_data_max[0], sample_nr)
            dim_y = np.linspace(projected_data_min[1], projected_data_max[1], sample_nr)
            grid_search = np.array(np.meshgrid(dim_x, dim_y)).T.reshape(-1, 2)

            inverse_augmented_data = parallel_inverse_transform(mapper, grid_search, num_processes=10)

            return grid_search, inverse_augmented_data


        def create_predictions(data):
            shape = data.shape
            data = torch.from_numpy(data).reshape(shape[0], 1, shape[1]).float().to(device)
            predictions = model(data)
            predictions = predictions.detach().cpu().numpy()
            return predictions

        # Generate grid data and prediction density for data
        print('Start with density time series map')
        augmented_time_series = {}
        for stage in base_data_for_maps['data']:
            print('\t', f'Start with {stage}')
            mapper, projected_data = mappings_to_create['data'][stage]
            data = base_data_for_maps['data'][stage]

            # Create further time series data
            mapped_data, augmented_stage_time_series = create_decision_border_dense_map_for_data(data, mapper, projected_data)
            density_predictions = create_predictions(augmented_stage_time_series)
            augmented_time_series[stage] = [augmented_stage_time_series, density_predictions]
            density_map['data'][stage] = [mapped_data, density_predictions]

        # Generate activation prediction density
        print('Start with density activations map')
        for stage in base_data_for_maps['activations']:
            print('\t', f'Start with {stage}')
            mapper, projected_data = mappings_to_create['activations'][stage]

            # Create further activation data
            activation_data = base_data_for_maps['activations'][stage]
            mapped_data, augmented_stage_activations = create_decision_border_dense_map_for_data(activation_data, mapper, projected_data)

            density_predictions = []
            ts_shape = list(base_data_for_maps['data'][stage].shape)
            ts_shape = [1, 1, ts_shape[-1]]
            ts_data = np.array(base_data_for_maps['data'][stage])
            for activation in tqdm(augmented_stage_activations):
                activation = torch.from_numpy(activation).float().to(device)
                _, prediction = get_time_series_from_activations(model, activation, ts_shape, -2, activation_data, ts_data)
                density_predictions.append(prediction)
            density_predictions = np.array(density_predictions)

            density_map['activations'][stage] = [mapped_data, density_predictions]
        print('Done with activations')

        # Generate attribution prediction density
        print('Start with density attributions map')
        for stage in base_data_for_maps['attributions']:
            print('\t', f'Start with {stage}')
            density_map['attributions'][stage] = {}

            for attribution in base_data_for_maps['attributions'][stage]:
                print('\t\t', f'Start with {attribution}')
                mapper, projected_data = mappings_to_create['attributions'][stage][attribution]

                # Create further attribution data
                attribution_data = base_data_for_maps['attributions'][stage][attribution]
                mapped_data, augmented_stage_attribution = create_decision_border_dense_map_for_data(attribution_data, mapper, projected_data)

                density_predictions = []
                ts_shape = list(base_data_for_maps['data'][stage].shape)
                ts_shape = [1, 1, ts_shape[-1]]
                ts_data = np.array(base_data_for_maps['data'][stage])
                for attr in tqdm(augmented_stage_attribution):
                    attr = torch.from_numpy(attr).float().to(device)
                    _, prediction = get_time_series_from_attributions(model, attr, ts_shape, attribution_techniques_dict[attribution], attribution_data, ts_data)
                    density_predictions.append(prediction)
                density_predictions = np.array(density_predictions)

                density_map['attributions'][stage][attribution] = [mapped_data, density_predictions]
        print('Done with attributions')

        with open(density_path, 'wb') as file:
            dill.dump(density_map, file)

    print('Done with generating density maps')
    print(f'Extracting data done for {model_base_name}.')


if __name__ == '__main__':
    main()
