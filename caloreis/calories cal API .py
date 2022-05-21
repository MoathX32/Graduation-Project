import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import json

Gym_Activities = (pd.read_excel("E:\\desktop\\Project\\Moath\\caloreis\\Gym Activities.xlsx", index_col=None))
Gym_Activities.rename(columns = {'57 kg':'burning Level 1', '70 kg':'burning Level 2', '84 kg':'burning Level 3'}, inplace = True)
Home_Daily_Life_Activities = (pd.read_excel("E:\\desktop\\Project\\Moath\\caloreis\\Home & Daily Life Activities.xlsx", index_col=None))
Home_Daily_Life_Activities.rename(columns = {'57 kg':'burning Level 1', '70 kg':'burning Level 2', '84 kg':'burning Level 3'}, inplace = True)
Outdoor_Activities = (pd.read_excel("E:\\desktop\\Project\\Moath\\caloreis\\Outdoor Activities.xlsx", index_col=None))
Outdoor_Activities.rename(columns = {'57 kg':'burning Level 1', '70 kg':'burning Level 2', '84 kg':'burning Level 3'}, inplace = True)
Training_and_Sports_Activities = (pd.read_excel("E:\\desktop\\Project\\Moath\\caloreis\\Training and Sports Activities.xlsx", index_col=None))
Training_and_Sports_Activities.rename(columns = {'57 kg':'burning Level 1', '70 kg':'burning Level 2', '84 kg':'burning Level 3'}, inplace = True)


app = Flask(__name__)
@app.route("/calculator", methods = ["GET" ,"POST"])

def calculator():

    json_ = request.json
    dict1 = dict(json_)

    if dict1["Gender"] == "female":
        c1 = 66
        hm = 6.2 * dict1["height"]
        wm = 12.7 * dict1["weight"]
        am = 6.76 * dict1["Age"]

    elif dict1["Gender"] =="male":
        dict1.update( Gender = 1)
        c1 = 655.1
        hm = 4.35 * dict1["height"]
        wm = 4.7 * dict1["weight"]
        am = 4.7 * dict1["Age"]

    bmr_result = c1 + hm + wm - am

    if dict1["activity_level"] == 'none':
        activity = 1.2 * bmr_result
    elif dict1["activity_level"] == 'light':
        activity = 1.375 * bmr_result
    elif dict1["activity_level"] == 'moderate':
        activity = 1.55 * bmr_result
    elif dict1["activity_level"] == 'heavy':
        activity = 1.725 * bmr_result
    elif dict1["activity_level"] == 'extreme':
        activity = 1.9 * bmr_result

    if dict1["goals"] == 'lose':
        calories = activity - 500
    elif dict1["goals"] == 'maintain':
        calories = activity
    elif dict1["goals"] == 'gain 1':
        calories = activity + 500
    elif dict1["goals"] == 'gain 2':
        calories = activity + 1000

    if dict1["preferred_activity"] == 'Gym Activities':
        if dict1["weight"] <= 60 :
            out = Gym_Activities.loc[:,["Gym Activities","burning Level 1"]]

        elif dict1["weight"] in range(60,80) :
            out = Gym_Activities.loc[:,["Gym Activities","burning Level 2"]]

        elif dict1["weight"] >= 80 :
            out = Gym_Activities.loc[:,["Gym Activities","burning Level 3"]]


    elif dict1["preferred_activity"] == 'Home & Daily Life Activities':
        if dict1["weight"] <= 60 :
            out = Home_Daily_Life_Activities.loc[:,["Home & Daily Life Activities","burning Level 1"]]

        elif dict1["weight"] in range(60,80) :
            out = Home_Daily_Life_Activities.loc[:,["Home & Daily Life Activities","burning Level 2"]]

        elif dict1["weight"] >= 80 :
            out = Home_Daily_Life_Activities.loc[:,["Home & Daily Life Activities","burning Level 3"]]
            
            
            
    elif dict1["preferred_activity"] == 'Outdoor Activities':
        if dict1["weight"] <= 60 :
            out = Outdoor_Activities.loc[:,["Outdoor Activities","burning Level 1"]]

        elif dict1["weight"] in range(60,80) :
            out = Outdoor_Activities.loc[:,["Outdoor Activities","burning Level 2"]]

        elif dict1["weight"] >= 80 :
            out = Outdoor_Activities.loc[:,["Outdoor Activities","burning Level 3"]]
            
            
            
    elif dict1["preferred_activity"] == 'Training and Sports Activities':
        if dict1["weight"] <= 60 :
            out = Training_and_Sports_Activities .loc[:,["Training and Sports Activities","burning Level 1"]]

        elif dict1["weight"] in range(60,80) :
            out = Training_and_Sports_Activities .loc[:,["Training and Sports Activities","burning Level 2"]]

        elif dict1["weight"] >= 80 :
            out = Training_and_Sports_Activities .loc[:,["Training and Sports Activities","burning Level 3"]]
            
            

    return json.dumps({"your goal": dict1["goals"] ,"your daily caloric goals should be": int(calories), "Calories Burned in 30-minute activities and Suitable activities for You": (out.to_dict('records'))})

app.config["SERVER_NAME"] = "localhost:8000"
if __name__ == '__main__':
    app.run(debug=True)
