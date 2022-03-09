import json
import logging
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

TEMP_DIR = Path("/tmp")
# Fix write permission for librosa
# https://github.com/librosa/librosa/issues/1156#issuecomment-714381149
os.environ["NUMBA_CACHE_DIR"] = TEMP_DIR.as_posix()
from utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
logger.info(f"ENDPOINT_NAME {ENDPOINT_NAME}")
ASSETS_DIR = Path(os.environ["ASSETS_DIR"])

sm_runtime = boto3.client("runtime.sagemaker")
region = boto3.Session().region_name
boto_session = boto3.Session(region_name=region)


def format_results(results):
    return list(map(float, results.split("\n")[:-1]))


def get_payload_from_json_content(content):
    logger.info(f"get_payload")
    payload = ""
    return payload


def format_response(message, status_code, content_type):
    return {
        "statusCode": str(status_code),
        "body": json.dumps(message),
        "headers": {
            "Content-Type": content_type,
            "Access-Control-Allow-Origin": "*",
            "X-SageMaker-Endpoint": ENDPOINT_NAME,
        },
    }


def get_prediction(event):
    logger.info(f"get_prediction")
    content_type = event["headers"].get("Content-Type", "text/csv")
    custom_attributes = event["headers"].get("X-Amzn-SageMaker-Custom-Attributes", "")
    logger.info(f"content_type: {content_type}")
    logger.info(f"custom_attributes: {custom_attributes}")

    orig_payload = event["body"]
    if content_type.startswith("application/json"):
        orig_payload = json.loads(orig_payload)
        csv_payload = get_payload_from_json_content(orig_payload)
    elif content_type.startswith("text/csv"):
        csv_payload = orig_payload
    else:
        message = "bad content type: {}".format(content_type)
        logger.error()
        return format_response({"message": message}, 500)

    logger.info(f"payload len: {len(csv_payload)}")
    response = sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        Body=csv_payload,
        ContentType="text/csv",
    )
    results = response["Body"].read().decode()
    preds = format_results(results)
    return format_response({"prediction": preds}, 200, content_type)


def lambda_handler(event, context):
    logger.info("## CONTEXT")
    logger.info(f"Lambda function ARN: {context.invoked_function_arn}")
    logger.info(f"Lambda function memory limits in MB: {context.memory_limit_in_mb}")
    logger.info(f"Time remaining in MS: {context.get_remaining_time_in_millis()}")
    logger.info("## EVENT")
    logger.info(json.dumps(event, indent=2)[:100])
    try:
        response = get_prediction(event)
        logger.info(response)
        return response
    except ClientError as e:
        logger.error(
            "Unexpected sagemaker error: {}".format(e.response["Error"]["Message"])
        )
        logger.error(e)
        content_type = event["headers"].get("Content-Type", "text/csv")
        return format_response(
            {"message": "Unexpected sagemaker error"}, 500, content_type
        )
