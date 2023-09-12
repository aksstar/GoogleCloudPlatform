FROM python

# Installs additional packages
RUN pip install lightgbm flask scikit-learn google-cloud-storage

ENV AIP_STORAGE_URI=gs://your-project-name/model
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080
# Copies the API code to the docker image.
COPY . ./

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "serving_model.py"]