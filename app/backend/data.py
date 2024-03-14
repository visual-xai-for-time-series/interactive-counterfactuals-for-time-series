import dill

from helper import *


substrings_to_find = ['data', 'activations', 'attributions', 'mappers', 'density']
endings_to_find = ['pkl', 'pkl', 'pkl', 'pkl', 'pkl']

base_path = '/data/'

base = 'resnet-ecg5000'
# base = 'resnet-forda'
files = None

directories_found = {}
def get_directories(path):
    global files
    directories = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    for directory_to_search in directories:
        directory_files = find_files(directory_to_search, substrings_to_find, endings_to_find)
        directories_found[directory_to_search] = directory_files

    print(f'Found:\n{directories_found}')
    for k in directories_found:
        if base in k:
            files = directories_found[k]
            break


data_in_memory = {}
def get_data_from_path(path, load_in_memory=True):
    if path in data_in_memory:
        return data_in_memory[path]
    with open(path, 'rb') as f:
        data = dill.load(f)
    if load_in_memory:
        data_in_memory[path] = data
    return data


def get_time_series_data(stage='train'):
    path = files['data']
    data = get_data_from_path(path)
    tmp_data = [
        data[stage][0],
        data[stage][1],
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


density_color_dict = {}


def get_projected_time_series_density_data(stage='train'):
    path = files['density']
    if f'{path}-data' not in density_color_dict:
        data = get_data_from_path(path)
        data = color_mixer(data['data'][stage])
        density_color_dict[f'{path}-data'] = data
    else:
        data = density_color_dict[f'{path}-data']
    return {'data': data}


def get_projected_activations_density_data(stage='train'):
    path = files['density']
    if f'{path}-activations' not in density_color_dict:
        data = get_data_from_path(path)
        data = color_mixer(data['activations'][stage])
        density_color_dict[f'{path}-activations'] = data
    else:
        data = density_color_dict[f'{path}-activations']
    return {'data': data}


def get_projected_attributions_density_data(stage='train', attribution=None):
    path = files['density']
    data = get_data_from_path(path)
    if attribution is None:
        attribution = list(data['attributions'][stage].keys())[0]
    if f'{path}-attributions-{attribution}' not in density_color_dict:
        data = color_mixer(data['attributions'][stage][attribution])
        density_color_dict[f'{path}-attributions-{attribution}'] = data
    else:
        data = density_color_dict[f'{path}-attributions-{attribution}']
    return {'data': data}


def preload():
    get_time_series_data()
    get_activations_data()
    get_attributions_data()

    get_projected_time_series_data()

    get_projected_time_series_density_data()


def download_data(url, local_filename):
    if not os.path.exists(local_filename):
        dir_name = os.path.dirname(local_filename)
        os.makedirs(dir_name, exist_ok=True)

        with urllib.request.urlopen(url) as response, open(local_filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        
        print(f'Download finished: {local_filename}')
    else:
        print(f'File already exists: {local_filename}')


def download_all_files(path):
    resnet_ecg5000 = [
        ('https://data.time-series-xai.dbvis.de/icfts/resnet-ecg5000.pt', 'resnet-ecg5000.pt'),
        ('https://data.time-series-xai.dbvis.de/icfts/resnet-ecg5000-extracted/resnet-ecg5000-activations.pkl', 'resnet-ecg5000-extracted/resnet-ecg5000-activations.pkl'),
        ('https://data.time-series-xai.dbvis.de/icfts/resnet-ecg5000-extracted/resnet-ecg5000-attributions.pkl', 'resnet-ecg5000-extracted/resnet-ecg5000-attributions.pkl'),
        ('https://data.time-series-xai.dbvis.de/icfts/resnet-ecg5000-extracted/resnet-ecg5000-data.pkl', 'resnet-ecg5000-extracted/resnet-ecg5000-data.pkl'),
        ('https://data.time-series-xai.dbvis.de/icfts/resnet-ecg5000-extracted/resnet-ecg5000-density.pkl', 'resnet-ecg5000-extracted/resnet-ecg5000-density.pkl'),
        ('https://data.time-series-xai.dbvis.de/icfts/resnet-ecg5000-extracted/resnet-ecg5000-mappers.pkl', 'resnet-ecg5000-extracted/resnet-ecg5000-mappers.pkl'),
    ]

    files = [resnet_ecg5000]
    for file in files:
        if isinstance(file, list):
            for link in file:
                download_data(link[0], os.path.join(path, link[1]))


download_all_files(base_path)

get_directories('/data/')

preload()
