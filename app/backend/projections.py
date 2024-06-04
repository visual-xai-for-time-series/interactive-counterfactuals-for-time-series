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


def project_time_series_data(data):
    mapper_data = data_in_memory('mappers')
    stage = data_in_memory.get_stage()
    mapper, _ = mapper_data['data'][stage]
    predictions = predict_using_model(data)
    proj_data = mapper.transform(data)
    return {'data': proj_data, 'prediction': predictions}


def project_activations_data(data):
    mapper_data = data_in_memory('mappers')
    stage = data_in_memory.get_stage()
    mapper, _ = mapper_data['activations'][stage]
    data, predictions = predict_and_get_activations_from_model(data)
    proj_data = mapper.transform(data)
    return {'data': proj_data, 'prediction': predictions}


def project_attributions_data(data, attribution=None):
    mapper_data = data_in_memory('mappers')
    stage = data_in_memory.get_stage()
    mapped_attributions = mapper_data['attributions'][stage]
    if attribution is None:
        attribution = list(mapped_attributions.keys())[0]
    mapper, _ = mapped_attributions[attribution]
    data, predictions = predict_and_get_attributions_from_model(data)
    proj_data = mapper.transform(data)
    return {'data': proj_data, 'prediction': predictions}


def inverse_project_time_series_data(data):
    mapper_data = data_in_memory('mappers')
    stage = data_in_memory.get_stage()
    mapper, _ = mapper_data['data'][stage]
    inv_data = mapper.inverse_transform(data)
    predictions = predict_using_model(inv_data)
    return {'data': inv_data, 'prediction': predictions}


def inverse_project_activations_data(data):
    mapper_data = data_in_memory('mappers')
    stage = data_in_memory.get_stage()
    mapper, _ = mapper_data['activations'][stage]
    inv_data = mapper.inverse_transform(data)
    data, predictions = generate_time_series_from_activation(inv_data)
    return {'data': data, 'prediction': predictions}


def inverse_project_attributions_data(data, attribution=None):
    mapper_data = data_in_memory('mappers')
    stage = data_in_memory.get_stage()
    mapped_attributions = mapper_data['attributions'][stage]
    if attribution is None:
        attribution = list(mapped_attributions.keys())[0]
    mapper, _ = mapped_attributions[attribution]
    inv_data = mapper.inverse_transform(data)
    data, predictions = generate_time_series_from_attribution(inv_data)
    return {'data': data, 'prediction': predictions}
