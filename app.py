from flask import Flask, request, jsonify
import traceback
import pandas as pd
from pathlib import Path
import numpy as np

from models import predictIntentClassifier, trainIntentClassifier

# API definition
app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    try:
        json_ = request.json
        # get api parameters
        bot_id = str(json_["bot_id"])
        model_name = str(json_["model_name"])
        # catch errors in case of missing parameters
        if not bot_id:
            return ("Please enter bot id")
        if not model_name:
            return ("Please enter model name")
        trainIntentClassifier(bot_id, model_name, json_)
        return ("Training finished successfully.")
    except:
        return jsonify({'trace': traceback.format_exc()})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        # get api parameters
        bot_id = str(json_["bot_id"])
        model_name = str(json_["model_name"])
        # catch errors in case of missing parameters
        if not bot_id:
            return ("Please enter bot id")
        if not model_name:
            return ("Please enter model name")
        # check if model has already been trained    
        model_dir = "resources/" + "models/" + model_name + "/" + bot_id
        if not Path(model_dir).is_dir():
            return ("No model has been trained. Please train a model first.")
        predictions = predictIntentClassifier(model_dir, json_)
        return jsonify({'prediction': str(predictions)})
    except:
        return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    app.run(port=port, debug=True)