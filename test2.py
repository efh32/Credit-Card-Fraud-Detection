import google.datalab as datalab
import requests
import pandas as pd
import tensorflow as tf
import base64
import numpy as np
import json

from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient import discovery
credentials = ServiceAccountCredentials.from_json_keyfile_name('your_credential_name.json')

project_name = 'your_project_name'
model_name = 'your_model_name'
model_version ='v2'

#api is the address of the model, url a formatted version of api
api = 'https://ml.googleapis.com/v1/projects/{project}/models/{model}/versions/{version}:predict'
url = api.format(project=project_name,
                 model=model_name,
                 version=model_version)

model_1 = discovery.build('ml', 'v1', credentials=credentials)
name = 'projects/{}/models/{}/versions/{}'.format(project_name, model_name, model_version)


img = base64.b64encode(example_str)

body = {
  'instances': [
        {"inputs": {'b64': img}}
    ]
}

request = model_1.projects().predict(
    name = name,
    body = body
)

response = request.execute()

response
            
