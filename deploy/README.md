# Deployment — GRU on AWS Lambda

`handler.py` loads `gru_model.keras` and `gru_preprocessing.pkl` once per container and serves one workout per request. The image is based on `public.ecr.aws/lambda/python:3.13` (with `libgomp` for TensorFlow); dependency versions in `requirements.txt` are pinned to those used for training so the scalers and model stay compatible.

## Build and push

```bash
AWS_REGION=eu-central-1
aws ecr create-repository --repository-name runsight-gru --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin <account-id>.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t runsight-gru deploy/
docker tag runsight-gru:latest <account-id>.dkr.ecr.$AWS_REGION.amazonaws.com/runsight-gru:latest
docker push <account-id>.dkr.ecr.$AWS_REGION.amazonaws.com/runsight-gru:latest
```

Create a Lambda function from the image (give it enough memory for TensorFlow, e.g. 2–3 GB, and a timeout that covers the cold start) and enable a Function URL. **Protect the URL** (IAM auth, or a shared secret header checked in the workflow) — an unauthenticated URL lets anyone run your function.

## Request / response

```bash
python scripts/make_sample_event.py > sample_streams.json
curl -s -X POST "$LAMBDA_URL" -H 'Content-Type: application/json' -d @sample_streams.json
```

```json
{"training_zone": "anaerobik", "anaerobic_probability": 0.83, "confidence": 0.66, "hr_p10": 128.0, "hr_p95": 170.0}
```

Required keys: `heartrate`, `altitude`, `latlng`, `velocity_smooth` (each `{"data": [...]}`). A missing key returns HTTP 400 with an error message; there is no inference-time quality control. A single-element list (n8n's item wrapper) is accepted.

## Local run

```bash
pip install -r deploy/requirements.txt
cd deploy && python handler.py ../sample_streams.json
```
