# Credit-Card-Fraud-Detection

## Project Description <a name="descrip"/> 

This model detects fradulent credit card transactions.  The data contains a large class imbalance where normal transactions greatly outnumber fraudulent transactions.  This model can be deployed using App Engine.    

Links: 
1)https://www.kaggle.com/mlg-ulb/creditcardfraud/home

## Table of Contents

[Project Description](#descrip) 

[Background](#background)

[Requirements](#requirements)

[How to Run](#run)

[License](#license)
 
## Background <a name="background"/>

[File Information](#fileInfo)
 
[Helpful Links](#concepts)


### File Information <a name="fileInfo"/>

1. model.py - This is the model trained using the data from Kaggle.  

2. test.py - Tests the model with a test numpy array.  For the second test, the numpy array is turned into a [protocol buffer](https://developers.google.com/protocol-buffers/docs/pythontutorial).  

3. test2.py -

4. App Engine Folder - 

    1.  app.yaml - Configuration file for App Engine's settings.  
    2.  appengine_config.py - Copy third party libraries into application directory. 
    3.  main.py - Deploys the image classification model as a web application.
    4.  main_test.py - Test web application.

### Helpful Links <a name="concepts"/>

1. Further explaination of transfer learning process: https://codelabs.developers.google.com/codelabs/cpb102-txf-learning/index.html?index=..%2F..%2Findex#0

2. Additional information on app.yaml: https://cloud.google.com/appengine/docs/standard/python/config/appref

3. Additional information on appengine_config.py: https://cloud.google.com/appengine/docs/standard/python/tools/using-libraries-python-27

4. How to deploy model: https://cloud.google.com/appengine/docs/standard/python/getting-started/deploying-the-application



## Requirements <a name="requirements"/>

1. Python 3 - https://www.python.org/getit/
2. TensorFlow - https://www.tensorflow.org/install/
3. Google Cloud - https://cloud.google.com/products/
    * Cloud Datalab - https://cloud.google.com/datalab/
    * App Engine - https://cloud.google.com/appengine/ 

  
## How to Run <a name="run"/>
The following steps are performed in Google Cloud Datalab.  Refer to [this link](https://cloud.google.com/datalab/docs/quickstart) to get started with Datalab.  

1. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).  Store the csv file into a Google Cloud bucket.  Change line 9 to the link of the csv file in the Google cloud bucket.

```Python
creditcardcsv = #path to credit card fraud data
df = pd.read_csv(StringIO(creditcardcsv))
```

2. Follow the following instructions to deploy the model into a google cloud bucket.  https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models

3. Name the model and set the version number.  For example the model classifies subspecies of snake and the version of the model is 'beta1'.  

``` Python
Models().create('snake')
ModelVersions('snake').deploy('beta1', model_dir)
```

4. Run validate.py in order to test the model.  Make sure the bucket that stores the validation csv is set.  

Lines 7 and 8 contain the following:
```Python
#set the validation_data to the validation data on google cloud
validation_data = CsvDataSet('gs://my-bucket/snake_images_validation'
			, schema='image_url:STRING,label:STRING')
```

5. In the main.py file (located in the App Engine folder) make sure the project name, model name and model version match the corresponding files in train.py and validate.py.  These are found in lines 35-38.  

6. OAuth is used to authenticate the application.  Link: https://cloud.google.com/appengine/docs/standard/python/oauth/#oauth_20_and_openid_connect

7. Deploy the application using the steps from the following link: https://cloud.google.com/appengine/docs/standard/python/getting-started/deploying-the-application


## License <a name="license"/>
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

