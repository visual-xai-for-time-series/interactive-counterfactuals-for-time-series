import os

import dill
import random

import umap

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from helper import *
from model import *
from data import *


random_seed = 13

torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def project_time_series_data(stage, data):
    path = files['mappers']
    mapper_data = get_data_from_path(path)
    mapper = mapper_data['data'][stage][0]
    predictions = predict_using_model(data)
    proj_data = mapper.transform(data)
    return {'data': proj_data, 'prediction': predictions}


def project_activations_data(stage, data):
    path = files['mappers']
    mapper_data = get_data_from_path(path)
    mapper = mapper_data['activations'][stage][0]
    data, predictions = predict_and_get_activations_from_model(data)
    proj_data = mapper.transform(data)
    return {'data': proj_data, 'prediction': predictions}


def project_attributions_data(stage, data, attribution=None):
    path = files['mappers']
    mapper_data = get_data_from_path(path)
    if attribution is None:
        attribution = list(mapper_data['attributions'][stage].keys())[0]
    mapper = mapper_data['attributions'][stage][attribution][0]
    data, predictions = predict_and_get_attributions_from_model(data)
    proj_data = mapper.transform(data)
    return {'data': proj_data, 'prediction': predictions}


def inverse_project_time_series_data(stage, data):
    path = files['mappers']
    mapper_data = get_data_from_path(path)
    mapper = mapper_data['data'][stage][0]
    inv_data = mapper.inverse_transform(data)
    predictions = predict_using_model(inv_data)
    return {'data': inv_data, 'prediction': predictions}


def inverse_project_activations_data(stage, data):
    path = files['mappers']
    mapper_data = get_data_from_path(path)
    mapper = mapper_data['activations'][stage][0]
    inv_data = mapper.inverse_transform(data)

    data, predictions = generate_time_series_from_activation(inv_data)

    return {'data': data, 'prediction': predictions}


def inverse_project_attributions_data(stage, data, attribution=None):
    path = files['mappers']
    mapper_data = get_data_from_path(path)
    if attribution is None:
        attribution = list(mapper_data['attributions'][stage].keys())[0]
    mapper = mapper_data['attributions'][stage][attribution][0]
    inv_data = mapper.inverse_transform(data)
    return {'data': inv_data}
