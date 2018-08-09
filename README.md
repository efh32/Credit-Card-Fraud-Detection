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

2. test.py - The first test performs a prediction with the model using a predefined numpy array.  The second testFor the second test, the numpy array is turned into a [protocol buffer](https://developers.google.com/protocol-buffers/docs/pythontutorial).  

3. test2.py - Tests the authentication of the model.  

4. App Engine Folder - 

    1.  app.yaml - Configuration file for App Engine's settings.  
    2.  appengine_config.py - Copy third party libraries into application directory. 
    3.  main.py - Deploys the image classification model as a web application.
    4.  main_test.py - Test web application.

### Helpful Links <a name="concepts"/>

1. Additional information on app.yaml: https://cloud.google.com/appengine/docs/standard/python/config/appref

2. Additional information on appengine_config.py: https://cloud.google.com/appengine/docs/standard/python/tools/using-libraries-python-27

3. How to deploy model: https://cloud.google.com/appengine/docs/standard/python/getting-started/deploying-the-application



## Requirements <a name="requirements"/>

1. Python 3 - https://www.python.org/getit/
2. TensorFlow - https://www.tensorflow.org/install/
3. Google Cloud - https://cloud.google.com/products/
    * Cloud Datalab - https://cloud.google.com/datalab/
    * App Engine - https://cloud.google.com/appengine/ 

  
## How to Run <a name="run"/>
The following steps are performed in Google Cloud Datalab.  Refer to [this link](https://cloud.google.com/datalab/docs/quickstart) to get started with Datalab.  

1. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).  Store the csv file into a Google Cloud bucket.  Change line 9 to the link of the csv file in the Google cloud bucket.

The following are lines 9 and 10 from model.py.  Change line 9 to a string representation of the path to the csv file downloaded from Kaggle.
```Python
creditcardcsv = #path to credit card fraud data
df = pd.read_csv(StringIO(creditcardcsv))
```

2. The following is line 211 from model.py.  Change the argument found in the SavedModelBuilder function to the path of the Google Cloud bucket used to store the model.    
```Python
builder = tf.saved_model.builder.SavedModelBuilder("/path_for_local_model")
```

3. Run model.py in the Google Cloud datalab environment.  

The following instructions may provide help to deploy the model into a google cloud bucket:  https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models

4. In the main.py file (located in the App Engine folder) make sure the project name, model name and model version match the corresponding variables located in train.py and validate.py.  

Lines 44-46 in main.py contain the project name, model name and model version.  
``` Python
project_name = 'your_project_name'
model_name = 'your_model_name'
model_version ='v1'
```

5. OAuth is used to authenticate the use of the model for our App Engine main.py file.  
Follow the instructions in this link to generate OAuth credential.  Download the credential as a .json file.   
Link: https://cloud.google.com/video-intelligence/docs/common/auth

Store the credential in the same directory where the main.py is located in.  Change line 26 so that the argument in the function matches the name of the .json credential.  

The following is line 32 in main.py.
```Python
redentials = ServiceAccountCredentials.from_json_keyfile_name('your_credential_name.json')
```

6. Deploy the application in datalab using the steps from the following link: https://cloud.google.com/appengine/docs/standard/python/getting-started/deploying-the-application

The web application should display web forms that correspond to each feature in the data.


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

