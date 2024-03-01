import dill

from helper import *


directory_to_search = '/data/resnet-ecg5000-extracted/'
substrings_to_find = ['data', 'activations', 'attributions', 'mappers', 'density']
endings_to_find = ['pkl', 'pkl', 'pkl', 'pkl', 'pkl']

files = find_files(directory_to_search, substrings_to_find, endings_to_find)


in_memory = {}
def get_data_from_path(path, load_in_memory=True):
    if path in in_memory:
        return in_memory[path]
    with open(path, 'rb') as f:
        data = dill.load(f)
    if load_in_memory:
        in_memory[path] = data
    return data


def get_time_series_data(stage='train'):
    path = files['data']
    data = get_data_from_path(path)
    tmp_data = [
        data[stage][0],
        data[stage][1] - 1,
        argmax(data[stage][2]),
        data[stage][2],
    ]
    return {'data': tmp_data}


def get_activations_data(stage='train'):
    path = files['activations']
    data = get_data_from_path(path)
    return {'data': data[stage]}


def get_attributions_data(stage='train'):
    path = files['attributions']
    data = get_data_from_path(path)
    return {'data': data[stage]}


def get_projected_time_series_data(stage='train'):
    path = files['mappers']
    data = get_data_from_path(path)
    return {'data': data['data'][stage][1]}


def get_projected_activations_data(stage='train'):
    path = files['mappers']
    data = get_data_from_path(path)
    return {'data': data['activations'][stage][1]}


def get_projected_attributions_data(stage='train', attribution=None):
    path = files['mappers']
    data = get_data_from_path(path)
    if attribution is None:
        attribution = list(data['attributions'][stage].keys())[0]
    return {'data': data['attributions'][stage][attribution][1]}


def get_projected_time_series_density_data(stage='train'):
    path = files['density']
    data = get_data_from_path(path)
    data = color_mixer(data['data'][stage])
    return {'data': data}


def get_projected_activations_density_data(stage='train'):
    path = files['density']
    data = get_data_from_path(path)
    data = color_mixer(data['activations'][stage])
    return {'data': data}


def get_projected_attributions_density_data(stage='train', attribution=None):
    path = files['density']
    data = get_data_from_path(path)
    print(data['attributions'].keys())
    return {'data': []}
    # if attribution is None:
    #     attribution = list(data['attributions'][stage].keys())[0]
    # return {'data': data['attributions'][stage][attribution]}


def preload():
    get_time_series_data()
    get_activations_data()
    get_attributions_data()

    get_projected_time_series_data()

    get_projected_time_series_density_data()


preload()
