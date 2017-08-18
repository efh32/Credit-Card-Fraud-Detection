import google.datalab as datalab
import requests
import pandas as pd
import tensorflow as tf
import base64
import numpy as np

model_name = 'your_model_name'
model_version = 'v1'
api = 'https://ml.googleapis.com/v1/projects/{project}/models/{model}/versions/{version}:predict'
url = api.format(project=datalab.Context.default().project_id,
                 model=model_name,
                 version=model_version)

headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer ' + datalab.Context.default().credentials.get_access_token().access_token
}
#print headers
numpy_arr = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30]

body = {
  'instances': [
    numpy_arr
  ]
}

print(body)

response = requests.post(url, json=body, headers=headers)
print(response.json())
predictions = response.json()['predictions']

predictions

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
  
model_name = 'your_model_name'
model_version = 'v2'
api = 'https://ml.googleapis.com/v1/projects/{project}/models/{model}/versions/{version}:predict'
url = api.format(project=datalab.Context.default().project_id,
                 model=model_name,
                 version=model_version)

headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer ' + datalab.Context.default().credentials.get_access_token().access_token
}
#print headers

numpy_arr = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.10,\
             0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,\
             0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30])
example = tf.train.Example(features=tf.train.Features(feature={"nn_features": _floats_feature(numpy_arr)}))
#print(example)
example_str = base64.b64encode(example.SerializeToString())
#print(example_str)

body = {
  'instances': [
    {'b64': example_str}
  ]
}

#print(body)

response = requests.post(url, json=body, headers=headers)
#print(response.json())
predictions = response.json()['predictions']

predictions


