# Copyright 2015 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sample application that demonstrates how to use the App Engine Blobstore API.

For more information, see README.md.
"""

# [START all]
from google.appengine.api import users
from google.appengine.ext import blobstore
from google.appengine.ext import ndb
from google.appengine.ext.webapp import blobstore_handlers
import webapp2
import wsgiref.handlers

#upload credentials
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient import discovery
credentials = ServiceAccountCredentials.from_json_keyfile_name('your_credential_name.json')


#import for turning uploaded image into json
import base64
import sys
import json




#variables to access api
project_name = 'your_project_name'
model_name = 'your_model_name'
model_version ='v1'

#set the url to the api address
api = 'https://ml.googleapis.com/v1/projects/{project}/models/{model}/versions/{version}:predict'
url = api.format(project=project_name,
                 model=model_name,
                 version=model_version)

#this is the handler that handles incoming requests
class MainFraudDataHandler(webapp2.RequestHandler):
	

        	
	formstring ="""
	<form method="post" action="/">
	      <div>Time:<input type="text" name="Time" value = {{input_Time}}></div> 
	      <div>V1:<input type="text" name="V1" value={{input_V1}} ></div> 
	      <div>V2:<input type="text" name="V2" value = {{input_V2}}></div> 
	      <div>V3:<input type="text" name="V3" value={{input_V3}} ></div> 
	      <div>V4:<input type="text" name="V4" value = {{input_V4}}></div> 
	      <div>V5:<input type="text" name="V5" value={{input_V5}} ></div> 
	      <div>V6:<input type="text" name="V6" value = {{input_V6}}></div> 
	      <div>V7:<input type="text" name="V7" value={{input_V7}} ></div> 
	      <div>V8:<input type="text" name="V8" value = {{input_V8}}></div> 
	      <div>V9:<input type="text" name="V9" value={{input_V9}} ></div> 
	      <div>V10:<input type="text" name="V10" value = {{input_V10}}></div> 
	      <div>V11:<input type="text" name="V11" value={{input_V11}} ></div>
	      <div>V12:<input type="text" name="V12" value = {{input_V12}}></div> 
	      <div>V13:<input type="text" name="V13" value={{input_V13}} ></div> 
	      <div>V14:<input type="text" name="V14" value = {{input_V14}}></div> 
	      <div>V15:<input type="text" name="V15" value={{input_V15}} ></div> 
	      <div>V16:<input type="text" name="V16" value = {{input_V16}}></div> 
	      <div>V17:<input type="text" name="V17" value={{input_V17}} ></div> 
	      <div>V18:<input type="text" name="V18" value = {{input_V18}}></div> 
	      <div>V19:<input type="text" name="V19" value={{input_V19}} ></div> 
	      <div>V20:<input type="text" name="V20" value = {{input_V20}}></div> 
	      <div>V21:<input type="text" name="V21" value={{input_V21}} ></div> 
	      <div>V22:<input type="text" name="V22" value = {{input_V22}}></div> 
	      <div>V23:<input type="text" name="V23" value={{input_V23}} ></div>
	      <div>V24:<input type="text" name="V24" value = {{input_V24}}></div> 
	      <div>V25:<input type="text" name="V25" value={{input_V25}} ></div> 
	      <div>V26:<input type="text" name="V26" value = {{input_V26}}></div> 
	      <div>V27:<input type="text" name="V27" value={{input_V27}} ></div> 
	      <div>V28:<input type="text" name="V28" value = {{input_V28}}></div> 
	      <div>Amount:<input type="text" name="Amount" value={{input_Amount}} ></div> 
	      <div><input type="submit" value="Get Fraud Result"></div>
	</form>"""
	def get(self):
		self.response.out.write('<p>Please Enter Credit Card information</p>\n')
		self.response.out.write(self.formstring)

	def post(self):
		#get string input from the forms and convert them into int
		st_input_Time = self.request.get("Time")
		input_Time = float(st_input_Time)
		
		st_input_V1 = self.request.get("V1")
		input_V1 = float(st_input_V1)
		
		st_input_V2 = self.request.get("V2")
		input_V2 = float(st_input_V2)
		
		st_input_V3 = self.request.get("V3")
		input_V3 = float(st_input_V3)

		st_input_V4 = self.request.get("V4")
		input_V4 = float(st_input_V4)

		st_input_V5 = self.request.get("V5")
		input_V5 = float(st_input_V5)

		st_input_V6 = self.request.get("V6")
		input_V6 = float(st_input_V6)

		st_input_V7 = self.request.get("V7")
		input_V7 = float(st_input_V7)

		st_input_V8 = self.request.get("V8")
		input_V8 = float(st_input_V8)

		st_input_V9 = self.request.get("V9")
		input_V9 = float(st_input_V9)

		st_input_V10 = self.request.get("V10")
		input_V10 = float(st_input_V10)

		st_input_V11 = self.request.get("V11")
		input_V11 = float(st_input_V11)

		st_input_V12 = self.request.get("V12")
		input_V12 = float(st_input_V12)

		st_input_V13 = self.request.get("V13")
		input_V13 = float(st_input_V13)

		st_input_V14 = self.request.get("V14")
		input_V14 = float(st_input_V14)

		st_input_V15 = self.request.get("V15")
		input_V15 = float(st_input_V15)

		st_input_V16 = self.request.get("V16")
		input_V16 = float(st_input_V16)

		st_input_V17 = self.request.get("V17")
		input_V17 = float(st_input_V17)

		st_input_V18 = self.request.get("V18")
		input_V18 = float(st_input_V18)

		st_input_V19 = self.request.get("V19")
		input_V19 = float(st_input_V19)

		st_input_V20 = self.request.get("V20")
		input_V20 = float(st_input_V20)

		st_input_V21 = self.request.get("V21")
		input_V21 = float(st_input_V21)

		st_input_V22 = self.request.get("V22")
		input_V22 = float(st_input_V22)

		st_input_V23 = self.request.get("V23")
		input_V23 = float(st_input_V23)

		st_input_V24 = self.request.get("V24")
		input_V24 = float(st_input_V24)

		st_input_V25 = self.request.get("V25")
		input_V25 = float(st_input_V25)

		st_input_V26 = self.request.get("V26")
		input_V26 = float(st_input_V26)

		st_input_V27 = self.request.get("V27")
		input_V27 = float(st_input_V27)

		st_input_V28 = self.request.get("V28")
		input_V28 = float(st_input_V28)

		st_input_Amount = self.request.get("Amount")
		input_Amount = float(st_input_Amount)

		#send form input data into the fraud detection model and return a prediction
		model_1 = discovery.build('ml', 'v1', credentials=credentials)
		name = 'projects/{}/models/{}/versions/{}'.format(project_name, model_name, model_version)


     		numpy_arr = [input_Time, input_V1, input_V2, input_V3, input_V4, input_V5, input_V6, input_V7, input_V8, input_V9,\
				input_V10, input_V11, input_V12, input_V13, input_V14, input_V15, input_V16, input_V17, input_V18,\
				input_V19, input_V20, input_V21, input_V22, input_V23, input_V24, input_V25, input_V26, input_V27,\
				input_V28, input_Amount]



		body = {
		'instances': [
			numpy_arr
		]}

		response = model_1.projects().predict(
			name = name,
			body = body
		).execute()

		if 'error' in response:
			raise RuntimeError(response['error'])

		self.response.out.write(response['predictions'])



app = webapp2.WSGIApplication([
    ('/.*', MainFraudDataHandler),
], debug=True)
wsgiref.handlers.CGIHandler().run(app)
# [END all]


