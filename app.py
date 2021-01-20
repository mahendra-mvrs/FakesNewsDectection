
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('multiclas.pkl','rb')) # multi classifier model

multicv = pickle.load(open('cv.pkl', 'rb')) #countvectorizer

svm_baggingclassifier = pickle.load(open('svm_bagging.pkl', 'rb')) #svm model

sgd_baggingclassifer = pickle.load(open('sgd_bagging.pkl', 'rb'))#sgd model

gpc_model = pickle.load(open('gpc.pkl', 'rb'))#gpc model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    texts = request.form.values()
    cv_prediction = multicv.transform(texts).toarray()
    
    #prediction for main model
    prediction_model = model.predict(cv_prediction)
    
    #prediction for gpc model
    prediction_gpc = gpc_model.predict(cv_prediction)
    
    #prediction for sgd bagging
    prediction_sgdbag = sgd_baggingclassifer.predict(cv_prediction)
    
    #prediction for svm bagging 
    prediction_svmbag = svm_baggingclassifier.predict(cv_prediction)
    
    return render_template('index.html',
                           prediction_text_model='{}'.format(prediction_model[0]),
                           prediction_text_gpc='{}'.format(prediction_gpc[0]),
                           prediction_text_sgdbag='{}'.format(prediction_sgdbag[0]),
                           prediction_text_svmbag='{}'.format(prediction_svmbag[0]))

if __name__ == "__main__":
    app.run(debug=True)






