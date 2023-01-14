FROM python

# Installs additional packages
RUN pip install lightgbm pandas numpy scikit-learn google-cloud-aiplatform protobuf==3.20.3 google-cloud-storage


ENV AIP_STORAGE_URI=gs://aakash-test-env/model/

# COPY your-path-to/model.py /root/model.py
COPY . ./

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "task.py"]