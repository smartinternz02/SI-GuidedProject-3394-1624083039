# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:50:35 2021
@author: moturi
"""
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import os

import requests
import json
# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "L4PxrhSMIBXiEKF--VCFCe5zydIDVNUPQG8G8oQd_gVA"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app=Flask(__name__)
with open('model.pkl','rb') as handle:
    model=pickle.load(handle)

@app.route('/')# route to display the home page
def home():
    return render_template('index.html')
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    return render_template('index1.html')
@app.route('/home',methods=['POST','GET'])
def my_home():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    #features_values=[np.array(input_features)]
    #feature_name=['cab_type','name','product_id','source','destination']
    #x=pd.DataFrame(features_values,columns=feature_name)
    #prediction=model.predict(x)
    #print("Prediction is : ",prediction)
    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"field": [['cab_type','name','product_id','source','destination']], "values": [input_features]}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/4a0e0804-7b03-4b37-9dbb-671254a1aad6/predictions?version=2021-06-28', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    predictions=response_scoring.json()
    predictions=round(predictions['predictions'][0]['values'][0][0],2)
    print(predictions)
    
    return render_template("results.html",predictions=predictions)

if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
    predict()