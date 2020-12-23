# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:32:51 2020

@author: Nagesh

Code is taken from https://github.com/krishnaik06/Heroku-Demo/blob/master/app.py
"""

import pickle
from flask import Flask, request, render_template



app = Flask(__name__)



with open("vec_model.pkl", "rb") as f_:
    vec = pickle.load(f_)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    prediction_text = []
    try:
        text = request.form.get("text")
        prediction = model.predict(vec.transform([str(text)]))
        
        prediction_text = "SPAM"
        if int(prediction[0]) == 0:
            prediction_text = "NOT SPAM"
    except Exception  as e:
            print("error: ", e)
    return render_template('index.html', prediction_text = "The sms is {}.".format(prediction_text))

    
if __name__ == "__main__":
    app.run(debug = True)