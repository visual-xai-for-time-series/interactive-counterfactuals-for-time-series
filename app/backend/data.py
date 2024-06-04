import dill
import urllib

from helper import *


substrings_to_find = ['data', 'activations', 'attributions', 'mappers', 'density']
endings_to_find = ['pkl', 'pkl', 'pkl', 'pkl', 'pkl']

base_path = '/data/'


def download_data(url, local_filename):
    if not os.path.exists(local_filename):
        dir_name = os.path.dirname(local_filename)
        os.makedirs(dir_name, exist_ok=True)

        with urllib.request.urlopen(url) as response, open(local_filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        
        print(f'Download finished: {local_filename}')
    else:
        if DEBUG:
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


class DataInMemory:
    found_directories = {}
    selected_files = {}
    loaded_files = {}
    loaded_color = {}

    selected_base = None
    selected_stage = None


    # resnet-ecg5000 resnet-wafer
    def __init__(self, path, base='resnet-ecg5000', stage='train'):
        directories = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

        for directory_to_search in directories:
            directory_files = find_files(directory_to_search, substrings_to_find, endings_to_find)
            directory_files = {k: v[0] for k, v in directory_files.items() if len(v) > 0}

            self.found_directories[directory_to_search] = directory_files

        print(f'Found data: {list(self.found_directories.keys())}')

        self.change_base(base)
        self.change_stage(stage)


    def __call__(self, base, load_in_memory=True):
        path = self.selected_files[base]
        data = self.__load_data(path, load_in_memory)
        return data


    def __load_data(self, path, load_in_memory=True):
        if path in self.loaded_files:
            data = self.loaded_files[path]
        else:
            try:
                with open(path, 'rb') as f:
                    data = dill.load(f)
            except Exception as e:
                print(f'Error with {path}')
                print(str(e))

            if load_in_memory:
                self.loaded_files[path] = data
        if isinstance(data, dict) and self.selected_stage in data:
            return data[self.selected_stage]
        return data


    def change_base(self, base):
        if base is None:
            selection = [[k, v] for k, v in self.found_directories.items()][0]
            self.selected_base, self.selected_files = selection
        else:
            selection = [[k, v] for k, v in self.found_directories.items() if base in k][0]
            self.selected_base, self.selected_files = selection
        print(f'Selected data: {self.selected_base}')

    
    def change_stage(self, stage='train'):
        self.selected_stage = stage
        print(f'Selected stage: {self.selected_stage}')


    def get_stages(self):
        stages = []
        path = self.selected_files['activations']
        if path in self.loaded_files:
            data = self.loaded_files[path]
            stages = list(data.keys())
        return stages


    def get_stage(self):
        return self.selected_stage


    def get_density(self, base):
        path = self.selected_files['density']
        if f'{path}-{base}' not in self.loaded_color:
            data = self.__load_data(path)
            data = color_mixer(data[base][self.selected_stage])
            self.loaded_color[f'{path}-{base}'] = data
        else:
            data = self.loaded_color[f'{path}-{base}']
        return data


    def get_density_multiple(self, base, attribution=None):
        path = self.selected_files['density']
        data = self.__load_data(path)
        attributions = data[base][self.selected_stage]
        if attribution is None:
            attribution = list(attributions.keys())[0]
        if f'{path}-{base}-{attribution}' not in self.loaded_color:
            data = color_mixer(data[base][self.selected_stage][attribution])
            self.loaded_color[f'{path}-{base}-{attribution}'] = data
        else:
            data = self.loaded_color[f'{path}-{base}-{attribution}']
        return data


download_all_files(base_path)
data_in_memory = DataInMemory(base_path)


def get_time_series_data():
    data = data_in_memory('data')
    tmp_data = [
        data[0],
        data[1],
        argmax(data[2]),
        data[2],
    ]
    return {'data': tmp_data}


def get_activations_data():
    data = data_in_memory('activations')
    return {'data': data}


def get_attributions_data():
    data = data_in_memory('attributions')
    return {'data': data}


def get_projected_time_series_data():
    data = data_in_memory('mappers')
    stage = data_in_memory.get_stage()
    _, mapped_data = data['data'][stage]
    return {'data': mapped_data}


def get_projected_activations_data():
    data = data_in_memory('mappers')
    stage = data_in_memory.get_stage()
    _, mapped_data = data['activations'][stage]
    return {'data': mapped_data}


def get_projected_attributions_data(attribution=None):
    data = data_in_memory('mappers')
    stage = data_in_memory.get_stage()
    mapped_attributions = data['attributions'][stage]
    if attribution is None:
        attribution = list(mapped_attributions.keys())[0]
    _, mapped_data = mapped_attributions[attribution]
    return {'data': mapped_data}


def get_projected_time_series_density_data():
    data = data_in_memory.get_density('data')
    coordinate_data, color_data, prediction_data = data
    return {'data': [coordinate_data, color_data], 'predictions': prediction_data}


def get_projected_activations_density_data():
    data = data_in_memory.get_density('activations')
    coordinate_data, color_data, prediction_data = data
    return {'data': [coordinate_data, color_data], 'predictions': prediction_data}


def get_projected_attributions_density_data(attribution=None):
    data = data_in_memory.get_density_multiple('attributions', attribution)
    coordinate_data, color_data, prediction_data = data
    return {'data': [coordinate_data, color_data], 'predictions': prediction_data}


def preload():
    get_time_series_data()
    get_activations_data()
    get_attributions_data()

    get_projected_time_series_data()

    get_projected_time_series_density_data()


preload()
