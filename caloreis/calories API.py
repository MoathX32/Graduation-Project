import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import json


sav = joblib.load("F:\\Data Science\\Training\\GitHub\\Machine-Learning\\Regression\\fitness-calorie-burn-rate-predict-99\\calories.sav")
model = sav["model"]
scl = sav["scaler"]

app = Flask(__name__)
@app.route("/prediction", methods = ["GET" ,"POST"])

def prediction():

    json_ = request.json
    dict1 = dict(json_)

    if dict1["Gender"] == "female":
        dict1.update( Gender = 0)
    elif dict1["Gender"] =="male":
        dict1.update( Gender = 1)

    dic = dict1.values()
    inpt = list(dic)



    input_data_as_numpy_array = np.asarray(inpt)


    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    input = scl.transform(input_data_reshaped)
    prediction = (model.predict(input))


    return json.dumps({"calories":float(prediction)})

app.config["SERVER_NAME"] = "localhost:8000"
if __name__ == '__main__':
    app.run(debug=True)
