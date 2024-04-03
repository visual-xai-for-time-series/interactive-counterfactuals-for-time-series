import json

import numpy as np

from fastapi import FastAPI, Body
from fastapi.responses import Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from model import *
from data import *
from projections import *


origins = [
    'http://localhost',
    'http://localhost:4200',
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()


def create_json_response(data):
    # Use custom JSON encoder to serialize NumPy array
    json_data = json.dumps(data, cls=CustomJSONEncoder)

    # Return JSON response
    return JSONResponse(content=json_data)


@app.get('/api/get_time_series/')
async def get_time_series():
    data = get_time_series_data()
    return create_json_response(data)


@app.get('/api/get_activations/')
async def get_activations():
    data = get_activations_data()
    return create_json_response(data)


@app.get('/api/get_attributions/')
async def get_attributions():
    data = get_attributions_data()
    return create_json_response(data)


@app.get('/api/get_projected_time_series/')
async def get_projected_time_series():
    data = get_projected_time_series_data()
    return create_json_response(data)


@app.get('/api/get_projected_activations/')
async def get_projected_activations():
    data = get_projected_activations_data()
    return create_json_response(data)


@app.get('/api/get_projected_attributions/')
async def get_projected_attributions():
    data = get_projected_attributions_data()
    return create_json_response(data)


@app.get('/api/get_projected_time_series_density/')
async def get_projected_time_series_density():
    data = get_projected_time_series_density_data()
    return create_json_response(data)


@app.get('/api/get_projected_activations_density/')
async def get_projected_activations_density():
    data = get_projected_activations_density_data()
    return create_json_response(data)


@app.get('/api/get_projected_attributions_density/')
async def get_projected_attributions_density():
    data = get_projected_attributions_density_data()
    return create_json_response(data)


@app.post('/api/project_time_series/')
async def project_time_series(request_body: dict = Body()):
    data = np.array(request_body['data'])
    if data.shape[-1] > 2:
        data = project_time_series_data(data)
        return create_json_response(data)
    else:
        return 'Error wrong format'


@app.post('/api/project_activations/')
async def project_activations(request_body: dict = Body()):
    data = np.array(request_body['data'])
    if data.shape[-1] > 2:
        data = project_activations_data(data)
        return create_json_response(data)
    else:
        return 'Error wrong format'


@app.post('/api/project_attributions/')
async def project_attributions(request_body: dict = Body()):
    data = np.array(request_body['data'])
    if data.shape[-1] > 2:
        data = project_attributions_data(data)
        return create_json_response(data)
    else:
        return 'Error wrong format'


@app.post('/api/inverse_project_time_series/')
async def inverse_project_time_series(request_body: dict = Body()):
    data = np.array(request_body['data'])
    if data.shape[-1] == 2:
        data = inverse_project_time_series_data(data)
        return create_json_response(data)
    else:
        return 'Error wrong format'


@app.post('/api/inverse_project_activations/')
async def inverse_project_activations(request_body: dict = Body()):
    data = np.array(request_body['data'])
    if data.shape[-1] == 2:
        data = inverse_project_activations_data(data)
        return create_json_response(data)
    else:
        return 'Error wrong format'


@app.post('/api/inverse_project_attributions/')
async def inverse_project_attributions(request_body: dict = Body()):
    data = np.array(request_body['data'])
    if data.shape[-1] == 2:
        data = inverse_project_attributions_data(data)
        return create_json_response(data)
    else:
        return 'Error wrong format'


@app.get('/api/get_stages/')
async def get_stages():
    return {'data': data_in_memory.get_stages()}


@app.get('/api/get_models/')
async def get_models():
    return {'data': models_in_memory.get_possible_models()}


@app.post('/api/set_stage/')
async def set_stage(stage: str = 'train'):
    data_in_memory.change_stage(stage)
    return 200


@app.post('/api/set_model/')
async def set_model(model: str = 'resnet-ecg5000'):
    models_in_memory.change_model(model)
    data_in_memory.change_base(model)
    return 200



@app.get('/')
async def home():
    return 'Server is running! You are not!'


if __name__ == '__main__':
    print('Start app')
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
