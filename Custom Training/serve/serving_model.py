from flask import Flask, jsonify, request, Response
import pickle
import lightgbm as lgb
import os
import json
import joblib
app = Flask(__name__)


from google.cloud import storage
# Initalizing variables.
PROJECT_ID = 'your-project-name'
model_file_name="model.pkl" 
bucket_name = 'your-project-name'
REGION = 'us-central1'
ARTIFACT_URI=f"gs://{bucket_name}/model"
BLOB_NAME = 'model/' + model_file_name
print(ARTIFACT_URI)
# Initialise a client
storage_client = storage.Client()
# Create a bucket object for our bucket
bucket = storage_client.get_bucket(bucket_name)
# Create a blob object from the filepath
blob = bucket.blob(BLOB_NAME)
# Download the file to a destination
blob.download_to_filename(model_file_name)

model =  joblib.load(model_file_name)

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