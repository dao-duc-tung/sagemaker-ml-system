import json
import logging

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")


def lambda_handler(event, context):
    if "EvaluationJobName" in event:
        job_name = event["EvaluationJobName"]
    else:
        raise KeyError("EvaluationJobName not found for event: {}.".format(json.dumps(event)))

    # Get the Processing job
    response = sm_client.describe_processing_job(ProcessingJobName=job_name)
    status = response["ProcessingJobStatus"]
    logger.info("Processing job:{} has status:{}.".format(job_name, status))

    # Get the metrics as a dictionary
    evaluation_output_config = response["ProcessingOutputConfig"]
    for output in evaluation_output_config["Outputs"]:
        if output["OutputName"] == "evaluation":
            evaluation_s3_uri = "{}/{}".format(output["S3Output"]["S3Uri"], "eval.json")
            break

    bucket, key = evaluation_s3_uri.split("/", 2)[-1].split("/", 1)
    eval_local_path = "/tmp/eval.json"
    s3_client.download_file(bucket, key, eval_local_path)
    with open(eval_local_path, "r") as f:
        evaluation_output_dict = json.loads(f.read())
    logger.info("Evaluation result:{}".format(evaluation_output_dict))

    return {
        "statusCode": 200,
        "results": {
            "EvaluationJobName": job_name,
            "ProcessingJobStatus": status,
            "ProcessingMetrics": evaluation_output_dict,
        },
    }
