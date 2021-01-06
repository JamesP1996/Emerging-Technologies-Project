import flask
from tensorflow import  keras
import numpy as np
import json
from flask import request

app = flask.Flask(__name__)

model = keras.models.load_model("./myModel.h5")


@app.route('/')
def home():
    return app.send_static_file('index.html')

    
@app.route('/predict',methods=["POST"])
def predict():
    data = {"success":False}
    print(model)
    # get the request parameters
    params = flask.request.form
    if (params == None):
        params = flask.request.form    # if parameters are found, echo the msg parameter 
    if (params != None):
        parsedValue = float(params['value'])
        this_is_an_array = np.array([parsedValue])
        x = model.predict(this_is_an_array)
        list = x.tolist()
        json_str = json.dumps(list)
        data["response"] = json_str
        data["success"] = True    # return a response in json format 
    return flask.jsonify(data)



# start the flask app, allow remote connections
app.run(host='127.168.1.1')

    