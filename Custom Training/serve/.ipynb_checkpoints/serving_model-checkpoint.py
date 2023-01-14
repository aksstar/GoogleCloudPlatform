from flask import Flask, jsonify, request, Response
import pickle
import lightgbm as lgb
import os
import json
import joblib
app = Flask(__name__)

#Loading model from AIP_MODEL_DIR path.
model_f = "model.pkl"
model =  joblib.load(model_f)

@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
   return {"status": "healthy"}


@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def add_income():
    request_json = request.json
    request_instances = request_json['instances']
    prediction=model.predict(request_instances)
    prediction = prediction.tolist()
    output = {'predictions':
                   [
                       {
                           'result' : prediction
                       }
                   ]
               }
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)