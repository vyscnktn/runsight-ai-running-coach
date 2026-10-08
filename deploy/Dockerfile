# Official AWS Lambda base image for Python 3.13
FROM public.ecr.aws/lambda/python:3.13

# TensorFlow needs libgomp at runtime
RUN dnf install -y libgomp && dnf clean all

# Install dependencies first so Docker caches this layer when only the handler changes
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Model, fitted preprocessing objects (scalers) and handler code
COPY gru_model.keras ${LAMBDA_TASK_ROOT}/
COPY gru_preprocessing.pkl ${LAMBDA_TASK_ROOT}/
COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.lambda_handler"]
