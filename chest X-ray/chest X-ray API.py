from flask import Flask, request
import json
from keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("KerasModel.h5")

app = Flask(__name__)
@app.route("/prediction", methods = ["GET" ,"POST"])

def prediction():
    img = request.files['image']
    x = Image.open(img).convert('L')
    #show = x.show()
    x = x.resize((224, 224))
    x = np.array(x) / 255
    x = x.reshape(-1, 224, 224, 1)


    pred = model.predict(x)

    predd = pred.tolist()

    for i in predd :
        print(i)
        if max(i) > 0.75 :
          prediction = np.argmax(pred, axis=1)
        else :
          prediction = np.array([[4]])

    labels = {'COVID19': 0, 'NORMAL': 1, 'PNEUMONIA': 2, 'TURBERCULOSIS': 3 ,'something els': 4}
    out = [k for k, v in labels.items() if v == prediction][0]
    return json.dumps({"The X-ray Image show that probably": str(out),"The accuracy on the predicted result is": "87.5%" })

app.config["SERVER_NAME"] = "localhost:8000"
if __name__ == '__main__':
    app.run(debug=True)