import json
import logging
import os
import shutil
import tarfile
from pathlib import Path

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")
ASSETS_DIR = Path(os.environ["ASSETS_DIR"])


def inspect_dir(path):
    files = []
    path = Path(path)
    if not path.exists():
        logger.warn(f"{path.as_posix()} doesn't exist.")
    else:
        files = os.listdir(str(path))
    logger.info(f"{path.as_posix()}: {files}")


def lambda_handler(event, context):
    if "TrainingJobName" in event:
        job_name = event["TrainingJobName"]
    else:
        raise KeyError("TrainingJobName not found for event: {}.".format(json.dumps(event)))

    # Get the training job
    response = sm_client.describe_training_job(TrainingJobName=job_name)
    status = response["TrainingJobStatus"]
    logger.info("Training job:{} has status:{}.".format(job_name, status))

    # Get the metrics as a dictionary
    model_artifacts_s3_uri = response["ModelArtifacts"]["S3ModelArtifacts"]
    bucket, key = model_artifacts_s3_uri.split("/", 2)[-1].split("/", 1)
    model_artifact_local_path = "/tmp/model.tar.gz"
    logger.info(f"Download {model_artifacts_s3_uri}")
    s3_client.download_file(bucket, key, model_artifact_local_path)

    # Extract and copy scalers dir
    model_local_path = Path("/tmp/model")
    logger.info(f"Extract to {model_local_path}")
    model_tar = tarfile.open(model_artifact_local_path)
    model_tar.extractall(model_local_path)
    inspect_dir(model_local_path)

    src_scalers_path = model_local_path / "scalers"
    inspect_dir(src_scalers_path)

    dst_scalers_path = ASSETS_DIR / "scalers"
    inspect_dir(dst_scalers_path)

    logger.info(f"Remove old files in {dst_scalers_path}")
    dst_scalers_path.mkdir(parents=True, exist_ok=True)
    for f in dst_scalers_path.glob("*"):
        os.remove(f)
    inspect_dir(dst_scalers_path)

    logger.info(f"Copy new files to {dst_scalers_path}")
    for f in src_scalers_path.glob("*"):
        shutil.copy(f, dst_scalers_path)
    with open(os.path.join(dst_scalers_path, "training-job-description.txt"), "w") as f:
        f.write(str(response))
    inspect_dir(dst_scalers_path)

    return {
        "statusCode": 200,
        "results": {
            "TrainingJobName": job_name,
            "TrainingJobStatus": status,
        },
    }
