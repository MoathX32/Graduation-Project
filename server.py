import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import json
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from keras.models import load_model
from PIL import Image
import imagehash

nltk.download('stopwords')

data = joblib.load('Data.sav')
cnn_model = load_model("KerasModel.h5")
cnn_model_2 = load_model('final_CNN.h5')

app = Flask(__name__)

@app.route("/X_Ray", methods = ["GET" ,"POST"])
def prediction():
    
    img = request.files['image']
    x = Image.open(img).convert('L')
    hash0 = imagehash.average_hash(Image.open('test.jpg'))
    hash1 = imagehash.average_hash(x)
    cutoff = 28

    if hash0 - hash1 < cutoff:
        x = x.resize((224, 224))
        x = np.array(x) / 255
        x = x.reshape(-1, 224, 224, 1)

        pred1 = cnn_model.predict(x)[0].argmax()
        pred2 = cnn_model_2.predict(x)[0].argmax()
        if pred1 == pred2 :
            pred = pred1
        else :
            pred = pred2

        predd = pred.tolist()

        for i in predd:

            if max(i) > 0.4:
                prediction = pred
                re = max(i) * 100
                res = "%s%%"%(round(re,2))
            else:
                prediction = np.array([[4]])
                res = str("Possible An Other Infirmity")



    else:
        prediction = np.array([[4]])
        res = str("Possible An Invalid Image")



    labels = {'COVID19': 0, 'NORMAL': 1, 'PNEUMONIA': 2, 'TURBERCULOSIS': 3 ,'UNRECOGNIZED Image': 4}
    out = [k for k, v in labels.items() if v == prediction][0]
    return json.dumps({"The X-ray Image show that probably": str(out),"The accuracy on the predicted result is  لو حالة نسبة رقم بس": res})

@app.route("/diagnosis", methods = ["GET" ,"POST"])
def diagnosis():
    sav = data["Disease model text"]
    description = sav['description']
    precaution = sav['precaution'].fillna('Take care of yourself well')
    model1 = sav['model']
    vectorizer = sav['vectorizer']

    json_ = request.json
    di = dict(json_)
    dic = di.values()


    stem = PorterStemmer()

    def stemming(content):
        content = re.sub('[^a-zA-Z]', ' ', str(content))
        content = content.lower()
        content = content.split()
        content = [stem.stem(word) for word in content if not word in stopwords.words('english')]
        content = ' '.join(content)
        return content

    inpt = stemming(dic)
    inpt = vectorizer.transform([inpt])

    o = model1.predict(((inpt)).reshape(1, -1))[0]

    def listToString(s):
        str1 = ""
        for ele in s:
            str1 += ele
        return str1

    a = listToString(o)

    for i in description['Disease']:
        if o == i:
            b = (description.loc[description[description['Disease'] == i].index[0], 'Description'])
            b = listToString(b)

    for i in precaution['Disease']:
        if o == i:
            c = (precaution.loc[precaution[precaution['Disease'] == i].index[0], 'Precaution_1'])
            c = listToString(c)
            d = (precaution.loc[precaution[precaution['Disease'] == i].index[0], 'Precaution_2'])
            d = listToString(d)
            e = (precaution.loc[precaution[precaution['Disease'] == i].index[0], 'Precaution_3'])
            e = listToString(e)
            f = (precaution.loc[precaution[precaution['Disease'] == i].index[0], 'Precaution_4'])
            f = listToString(f)


    return json.dumps({"prediction": (a) ,"description":(b) ,"precaution":(c,d,e,f)})

@app.route("/breast", methods = ["GET" ,"POST"])
def breast():
    sav = data["breast"]
    model = sav["model"]

    json_ = request.json
    dict1 = dict(json_)
    di = dict1.values()
    inpt = list(di)
    dic = np.asarray(inpt)

    input = dic.reshape(1,-1)
    prediction = (model.predict(input))

    if (prediction[0] == 0):
        prediction = ('Benign')
    else:
        prediction = ('possible Malignant')

    return json.dumps({"prediction":str(prediction)})

@app.route("/heart", methods = ["GET" ,"POST"])
def heart():
    sav = data["heart"]
    model = sav["model"]
    scaler = sav["scaler"]
    json_ = request.json
    dict1 = dict(json_)

    di = dict1.values()
    inpt = list(di)
    dic = np.asarray(inpt)


    input_data_reshaped = dic.reshape(1,-1)
    input = scaler.transform(input_data_reshaped)
    prediction = model.predict(input)

    if (prediction[0] == 0):
        prediction = ('The Person does not have a Heart Disease')
    else:
        prediction = ('The Person possible has Heart Disease')

    return json.dumps({"prediction":str(prediction)})

@app.route("/drug", methods = ["GET" ,"POST"])
def drug():
    df = data["drug_df"]
    similarity = data["drug_similarity"]

    json_ = request.json
    di = dict(json_)
    dic = di.values()
    dik = di.keys()


    def listToString(s):
        str1 = ""
        for ele in s:
            str1 += ele
        return str1

    diks = listToString(dik)

    if diks == "Drug":
        drug = listToString(dic)
        index_replaces = df[df['drugName'] == drug].index[0]
        distances_replaces = sorted(list(enumerate(similarity[index_replaces])),reverse=True,key = lambda x: x[1])
        count = 0
        for i in distances_replaces[0:5]:

            q =(df.iloc[i[0]].drugName)
            w =(df.iloc[i[0]].Prescribed_for)
            e =(df.iloc[i[0]].User_Rating * 10)
            r =(df.iloc[i[0]].Drug_Review)
            t =(df.iloc[i[0]].Date)

            count += 1

            if count == 1:
                q1 = str(q)
                w1 = str(w)
                e1 = str(e)
                r1 = str(r)
                t1 = str(t)
            elif count == 2:
                q2 = str(q)
                w2 = str(w)
                e2 = str(e)
                r2 = str(r)
                t2 = str(t)
            elif count == 3:
                q3 = str(q)
                w3 = str(w)
                e3 = str(e)
                r3 = str(r)
                t3 = str(t)
            elif count == 4:
                q4 = str(q)
                w4 = str(w)
                e4 = str(e)
                r4 = str(r)
                t4 = str(t)
            elif count == 5:
                q5 = str(q)
                w5 = str(w)
                e5 = str(e)
                r5 = str(r)
                t5 = str(t)

    elif diks == "Disease":
        disease = listToString(dic)
        index_replaces = df[df['Prescribed_for'] == disease].index[0]
        distances_replaces = sorted(list(enumerate(similarity[index_replaces])),reverse=True,key = lambda x: x[1])
        count = 0
        for i in distances_replaces[0:5]:

            q =(df.iloc[i[0]].drugName)
            w =(df.iloc[i[0]].Prescribed_for)
            e =(df.iloc[i[0]].User_Rating * 10)
            r =(df.iloc[i[0]].Drug_Review)
            t =(df.iloc[i[0]].Date)

            count += 1

            if count == 1:
                q1 = str(q)
                w1 = str(w)
                e1 = str(e)
                r1 = str(r)
                t1 = str(t)
            elif count == 2:
                q2 = str(q)
                w2 = str(w)
                e2 = str(e)
                r2 = str(r)
                t2 = str(t)
            elif count == 3:
                q3 = str(q)
                w3 = str(w)
                e3 = str(e)
                r3 = str(r)
                t3 = str(t)
            elif count == 4:
                q4 = str(q)
                w4 = str(w)
                e4 = str(e)
                r4 = str(r)
                t4 = str(t)
            elif count == 5:
                q5 = str(q)
                w5 = str(w)
                e5 = str(e)
                r5 = str(r)
                t5 = str(t)



    return json.dumps({
        "recommendation_1": {'Drug Name':q1,'Prescribed for':w1 ,'Users Rating':e1,'Drug Review':r1,'Review Date':t1},
        "recommendation_2": {'Drug Name':q2,'Prescribed for':w2 ,'Users Rating':e2,'Drug Review':r2,'Review Date':t2},
        "recommendation_3": {'Drug Name':q3,'Prescribed for':w3 ,'Users Rating':e3,'Drug Review':r3,'Review Date':t3},
        "recommendation_4": {'Drug Name':q4,'Prescribed for':w4 ,'Users Rating':e4,'Drug Review':r4,'Review Date':t4},
        "recommendation_4": {'Drug Name':q5,'Prescribed for':w5 ,'Users Rating':e5,'Drug Review':r5,'Review Date':t5}
                        })

@app.route("/diabetes", methods = ["GET" ,"POST"])
def diabetes():
    model = data["diabetes"]


    json_ = request.json
    dict1 = dict(json_)

    di = dict1.values()
    inpt = list(di)
    dic = np.asarray(inpt)

    input = dic.reshape(1,-1)

    prediction = (model.predict(input))
    if (prediction[0] == 0):
        prediction = ('not a diabetes')
    else:
        prediction = ('possible diabetes')



    return json.dumps({"prediction":str(prediction)})

@app.route("/AIcalories", methods = ["GET" ,"POST"])
def AIcalories():
    sav = data["calories"]
    model = sav["model"]
    scl = sav["scaler"]

    json_ = request.json
    dict1 = dict(json_)
    dic = dict1.values()
    inpt = list(dic)

    """    if inpt[0] == "female":
        inpt[0].replace(0)
    elif inpt[0] =="male":
        inpt[0].replace(1)
    """
    """dic = dict1.values()"""




    input_data_as_numpy_array = np.asarray(inpt)


    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    input = scl.transform(input_data_reshaped)
    prediction = (model.predict(input))


    return json.dumps({"calories":float(prediction)})

@app.route("/calories_calculator", methods = ["GET" ,"POST"])
def calories_calculator():
    sav = data["Activities"]
    Gym_Activities = sav["Gym_Activities"]
    Home_Daily_Life_Activities = sav["Home_Daily_Life_Activities"]
    Outdoor_Activities = sav["Outdoor_Activities"]
    Training_and_Sports_Activities = sav["Training_and_Sports_Activities"]

    json_ = request.json
    dict1 = dict(json_)

    if dict1["Gender"] == "female":
        c1 = 66
        hm = 6.2 * dict1["height"]
        wm = 12.7 * dict1["weight"]
        am = 6.76 * dict1["Age"]

    elif dict1["Gender"] == "male":
        dict1.update(Gender=1)
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
        if dict1["weight"] <= 60:
            out = Gym_Activities.loc[:, ["Gym Activities", "burning Level 1"]]

        elif dict1["weight"] in range(60, 80):
            out = Gym_Activities.loc[:, ["Gym Activities", "burning Level 2"]]

        elif dict1["weight"] >= 80:
            out = Gym_Activities.loc[:, ["Gym Activities", "burning Level 3"]]


    elif dict1["preferred_activity"] == 'Home & Daily Life Activities':
        if dict1["weight"] <= 60:
            out = Home_Daily_Life_Activities.loc[:, ["Home & Daily Life Activities", "burning Level 1"]]

        elif dict1["weight"] in range(60, 80):
            out = Home_Daily_Life_Activities.loc[:, ["Home & Daily Life Activities", "burning Level 2"]]

        elif dict1["weight"] >= 80:
            out = Home_Daily_Life_Activities.loc[:, ["Home & Daily Life Activities", "burning Level 3"]]



    elif dict1["preferred_activity"] == 'Outdoor Activities':
        if dict1["weight"] <= 60:
            out = Outdoor_Activities.loc[:, ["Outdoor Activities", "burning Level 1"]]

        elif dict1["weight"] in range(60, 80):
            out = Outdoor_Activities.loc[:, ["Outdoor Activities", "burning Level 2"]]

        elif dict1["weight"] >= 80:
            out = Outdoor_Activities.loc[:, ["Outdoor Activities", "burning Level 3"]]



    elif dict1["preferred_activity"] == 'Training and Sports Activities':
        if dict1["weight"] <= 60:
            out = Training_and_Sports_Activities.loc[:, ["Training and Sports Activities", "burning Level 1"]]

        elif dict1["weight"] in range(60, 80):
            out = Training_and_Sports_Activities.loc[:, ["Training and Sports Activities", "burning Level 2"]]

        elif dict1["weight"] >= 80:
            out = Training_and_Sports_Activities.loc[:, ["Training and Sports Activities", "burning Level 3"]]

    return json.dumps({"your goal": dict1["goals"], "your daily caloric goals should be": int(calories),
                       "Calories Burned in 30-minute activities and Suitable activities for You": (
                           out.to_dict('records'))})


if __name__ == '__main__':
    app.run(host="localhost", port=8000 ,debug=True)
