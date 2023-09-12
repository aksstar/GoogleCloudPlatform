import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import lightgbm as lgb
from google.cloud import storage
import os
import joblib

# Initalizing variables.
PROJECT_ID = 'aakash-test-env'
model_file_name="model.pkl" 
bucket_name = 'aakash-test-env'
REGION = 'us-central1'
ARTIFACT_URI=f"gs://{bucket_name}/model"
BLOB_NAME = 'model/' + model_file_name
print(ARTIFACT_URI)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(url, names = attributes)
dataset.columns = attributes

# Take out a test set
train, test = train_test_split(dataset, test_size = 0.4, stratify = dataset['class'], random_state = 42)
X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_train = train['class']
X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_test = test['class']


model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
prediction=model.predict(X_test)
# clf.booster_.save_model(model_file_name)
joblib.dump(model, model_file_name)

# Initialise a client
storage_client = storage.Client(project=PROJECT_ID)

# Upload model artifact to Cloud Storage
model_directory = os.environ['AIP_STORAGE_URI']
print("AIP_MODEL_DIR==>>>", model_directory)
storage_path = os.path.join(model_directory, model_file_name)
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(model_file_name)

print("Successfully Uploaded Model !")


print('The accuracy of the LightGBM is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))

