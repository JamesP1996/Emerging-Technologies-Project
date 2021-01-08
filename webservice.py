import flask
from tensorflow import  keras
import numpy as np
import json
from flask import request

# Create Flask App
app = flask.Flask(__name__)

# Import Prediction Model
model = keras.models.load_model("./myModel.h5")


# Home Route Return Index.HTML
@app.route('/')
def home():
    return app.send_static_file('index.html')

# Prediction Route
@app.route('/predict',methods=["POST"])
def predict():
    # Set the Data Success to False
    data = {"success":False}
    # Print Read in Model
    print(model)
    # get the request parameters
    params = flask.request.form
    if (params == None):
        params = flask.request.form    # Get the Params from request form
    if (params != None):
        # Parse the "Value" Param as a Float
        parsedValue = float(params['value'])
        # Use the parsed value and make it a Numpy Array
        this_is_an_array = np.array([parsedValue])
        # Make a prediction on the value array
        x = model.predict(this_is_an_array)
        # Send it to a List
        list = x.tolist()
        # Dump the List to a String
        json_str = json.dumps(list)
        # Set the Response from the server to the Json STR
        data["response"] = json_str
        # Successful is now true
        data["success"] = True    # return a response in json format 
        # Return the Jsonified Data
    return flask.jsonify(data)



# start the flask app, allow remote connections
app.run(host='0.0.0.0')

    