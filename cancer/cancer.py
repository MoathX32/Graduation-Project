from flask import Flask, request
import json
from keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("Model.h5")
model1 = load_model("final_CNN.h5")
model2 = load_model("base_CNN.h5")


app = Flask(__name__)
@app.route("/cancer", methods = ["GET" ,"POST"])

def cancer():
    img = request.files['image']
    x  = Image.open(img).resize((96, 96))
    x = np.array(x) / 255
    x = x.reshape(-1, 96, 96, 3)


    pred = model.predict(x)[0].argmax()


    y1 = Image.open(img).resize((50, 50))
    y1 = np.array(y1) / 255
    y1 = y1.reshape( -1,50, 50, 3)


    pred1 = model1.predict(y1)[0].argmax()
    pred2 = model2.predict(y1)[0].argmax()



    from collections import Counter
    vote = Counter([pred, pred1, pred2])
    print(pred)
    print(pred1)
    print(pred2)
    print(vote[0])

    if vote[0] == 1:
        out = str("IDC")
    else :
        out = str("NO IDC")

    return json.dumps({"The Image show that probably": str(out)})

app.config["SERVER_NAME"] = "localhost:8000"
if __name__ == '__main__':
    app.run(debug=True)