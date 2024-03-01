#!/usr/bin/env python

import os
import sys
import dill
import time
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

from captum.attr import GradientShap, IntegratedGradients, ShapleyValueSampling, Saliency, DeepLift

from sklearn.preprocessing import OneHotEncoder

from sktime.datasets import load_UCR_UEA_dataset

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

    print('Setting the stage')

    # Parse the arguments
    args = parser.parse_args()

    dataset = args.dataset
    model_type = args.model
    base_data_path = args.base_path

    ######## Set directories

    model_base_name = f'{model_type.lower()}-{dataset.lower()}'
    model_file = f'{model_base_name}.pt'

    model_path = os.path.join(base_data_path, model_file)

    extracted_data_path = os.path.join(base_data_path, f'{model_base_name}-extracted')

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
        activation_handle = layer_to_look_at.register_forward_hook(get_activation(activations, 'train'))

        for idx, (inputs, labels) in enumerate(dataloader_train_not_shuffled):
            inputs = inputs.reshape(inputs.shape[0], 1, -1)
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            preds = model(inputs)

        activation_handle.remove()

        activation_handle = layer_to_look_at.register_forward_hook(get_activation(activations, 'test'))

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

    attributions_path = os.path.join(extracted_data_path, f'{model_base_name.lower()}-attributions.pkl')
    if os.path.exists(attributions_path):
        with open(attributions_path, 'rb') as file:
            attributions = dill.load(file)

        attribution_techniques = [[x, None] for x in attributions['train'].keys()]
    else:
        model.eval()
        
        attribution_techniques = [
            # ['Saliency', Saliency],
            ['DeepLift', DeepLift],
            # ['IntegratedGradients', IntegratedGradients],
            # ['ShapleyValueSampling', ShapleyValueSampling],
        ]

        attributions = {'train': {}, 'test': {}}

        for at in attribution_techniques:
        
            at_name, at_function = at
            attribute_tec = at_function(model)

            print(f'Calculcate: {at_name}')
            print(f'Start with train')

            attributions_tmp = []
            for x in dataloader_train_not_shuffled:
                input_, label_ = x
                input_ = input_.reshape(input_.shape[0], 1, -1)
                input_ = input_.float().to(device)
                label_ = label_.float().to(device)

                attribution = attribute_tec.attribute(input_.reshape(-1, *shape).float().to(device), target=torch.argmax(label_, axis=1))
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

                attribution = attribute_tec.attribute(input_.reshape(-1, *shape).float().to(device), target=torch.argmax(label_, axis=1))
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


        def create_decision_border_dense_map_for_data(data, mapper):
            # Generate augmented data using mapper
            projected_data = mapper.transform(data)
            projected_data_max = np.max(projected_data, axis=0)
            projected_data_min = np.min(projected_data, axis=0)
            # sample_nr = 250
            sample_nr = 100
            dim_x = np.linspace(projected_data_min[0], projected_data_max[0], sample_nr)
            dim_y = np.linspace(projected_data_min[1], projected_data_max[1], sample_nr)
            grid_search = np.array(np.meshgrid(dim_x, dim_y)).T.reshape(-1, 2)
            inverse_augmented_data = mapper.inverse_transform(grid_search)
            return grid_search, inverse_augmented_data


        def create_predictions(data):
            shape = data.shape
            data = torch.from_numpy(data).reshape(shape[0], 1, shape[1]).float().to(device)
            predictions = model(data)
            predictions = predictions.detach().cpu().numpy()
            return predictions


        density_ts_path = os.path.join(extracted_data_path, f'{model_base_name.lower()}-ts-density.pkl')
        if os.path.exists(density_ts_path):
            with open(density_ts_path, 'rb') as file:
                density_ts_map = dill.load(file)
                [[mapped_data, density_predictions], augmented_time_series] = density_ts_map
        else:

            # Generate grid data and prediction density for data
            print('Start with density time series map')
            augmented_time_series = {}
            for stage in base_data_for_maps['data']:
                print('\t', f'Start with {stage}')
                mapper = mappings_to_create['data'][stage][0]
                data = base_data_for_maps['data'][stage]

                mapped_data, augmented_stage_time_series = create_decision_border_dense_map_for_data(data, mapper)
                density_predictions = create_predictions(augmented_stage_time_series)
                augmented_time_series[stage] = [augmented_stage_time_series, density_predictions]
                density_map['data'][stage] = [mapped_data, density_predictions]

            density_ts_map = [[mapped_data, density_predictions], augmented_time_series]

            with open(density_ts_path, 'wb') as file:
                dill.dump(density_ts_map, file)


        # Generate activation prediction density
        print('Start with density activations map')
        for stage in base_data_for_maps['activations']:
            print('\t', f'Start with {stage}')
            mapper = mappings_to_create['activations'][stage][0]

            # Take time series to get more activations
            augmented_ts_data, predictions = augmented_time_series[stage]

            activations = {}
            hook = get_activation(activations, 'layer_to_look_at')
            activation_handle = layer_to_look_at.register_forward_hook(hook)
            create_predictions(augmented_ts_data)
            augmented_activations = activations['layer_to_look_at'].detach().numpy()
            activation_handle.remove()

            print(f'activations: {augmented_activations}')

            mapped_augmented_data = mapper.transform(augmented_activations)

            print(f'activations: {mapped_augmented_data}')

            # Create further activation data
            activation_data = base_data_for_maps['activations'][stage]
            complete_activations = np.concatenate((activation_data, augmented_activations), axis=0)
            mapped_data, augmented_stage_activations = create_decision_border_dense_map_for_data(complete_activations, mapper)

            print(len(augmented_stage_activations))

            density_predictions = []
            ts_shape = list(base_data_for_maps['data'][stage].shape)
            ts_shape = [1, 1, ts_shape[-1]]
            ts_data = np.array(base_data_for_maps['data'][stage])
            for activation in augmented_stage_activations:
                activation = torch.from_numpy(activation).float().to(device)
                _, prediction = get_time_series_from_activations(model, activation, ts_shape, -2, activation_data, ts_data)
                print(prediction)
                density_predictions.append(prediction)
            density_predictions = np.array(density_predictions)

            density_map['activations'][stage] = [mapped_data, density_predictions]


        # Generate attribution prediction density
        print('Start with density attributions map')
        for stage in base_data_for_maps['attributions']:
            print('\t', f'Start with {stage}')

            for attribution in base_data_for_maps['attributions'][stage]:
                mapper = mappings_to_create['attributions'][stage][attribution][0]
                data, predictions = augmented_time_series[stage]

                create_predictions(data)


                density_map['attributions'][stage] = [mapped_data, predictions]


        with open(density_path, 'wb') as file:
            dill.dump(density_map, file)

    print('Done with generating density maps')
    print(f'Extracting data done for {model_base_name}.')


if __name__ == '__main__':
    main()
