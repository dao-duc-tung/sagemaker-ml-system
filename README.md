# SageMaker MLOps

## Introduction

The system customizes the [AWS safe deployment pipeline for SM](https://github.com/aws-samples/amazon-sagemaker-safe-deployment-pipeline). This solution provides a _Blue/Green_, also known as a _Canary_ deployment, by creating a Lambda endpoint that calls into a SageMaker endpoint for real-time inference.

### Abbreviation and Terminology

1. SM: SageMaker
1. LD: Lambda
1. CF: CloudFormation
1. CW: CloudWatch
1. SF: A State machine that is responsible for managing the training pipeline using AWS Step Functions
1. EFS: Elastic File System
1. SM endpoint: Model endpoint created by SageMaker, only can be used from an LD function
1. LD endpoint: A LD function that is responsible for serving prediction service by calling SM endpoint
1. API endpoint: API Gateway endpoint that links with the LD endpoint. This endpoint is public and will be called by the end application (ie. mobile app, web app, etc.)
1. Dev deployment: Deployment in development stage
1. Prod deployment: Deployment in production stage
1. System pipeline: The CodePipeline that has sveral steps including fetch source code, build CF stacks, run SF, dev deployment, and prod deployment

### System functionalities

The system supports the following functionalities:

- Training
  - Training data is saved in any S3 bucket
  - Customizable docker image for pre-processing data, training model and model evaluation
- Deployment
  - Customizable docker image for serving model at SM endpoint
  - Support multi-core serving at SM endpoint instance using Nginx and Gunicorn
  - 2-stage deployment: dev and prod
  - Manual approve to deploy model from dev to prod stage
  - Customizable docker image for serving prediction service at LD endpoint
- Monitoring
  - Capture request data and response data of SM endpoint
  - Detect feature drift and alert
  - CloudWatch dashboard to monitor system metrics
  - Send alarm to admin email
- Automation
  - Auto rerun system pipeline monthly
  - Upload data to S3 bucket -> auto rerun system pipeline
  - Source code updated -> auto rerun system pipeline
  - Feature drift detected -> auto rerun system pipeline
  - Auto scaling SM endpoint
  - Auto shifting traffic from the old LD endpoint to the new LD endpoint
  - Canary deployment strategy with rollback on error

## Architecture

The architecture diagram below shows the entire MLOps pipeline at a high level. This pipeline uses the CF template `pipeline.yml` to build.

![architecture][architecture]

### Component Details

- Key Management Service (KMS): encrypts data and artifacts.
- Secrets Manager: stores your GitHub Access Token.
- Simple Notification Service (SNS): notifies you when CodeDeploy has successfully deployed the API, and to receive alerts for retraining and drift detection (signing up for these notifications is optional).
- Two Amazon CloudWatch event rules: one which schedules the pipeline to run every month, and one which triggers the pipeline to run when SageMaker Model Monitor detects certain metrics.
- SageMaker Jupyter notebook: to develop.
- S3: stores model artifacts.
- CodePipeline: defines several stages.

## Customize Docker images

This step creates several python scripts for steps including preparing data, training model, evaluating model, and serving model. This step will be done in an SM notebook instance.

The input of this step is the training code, evaluation code, and the model serving code of an ML model. The output of this step is the two docker images that contain all of these scripts.

The 1st docker image serves as a multi-functional docker image for preparing data, training model, evaluating model, and serving model. Combining all of these scripts will simplify the development process without creating several docker images for each individual script which is overkilled for this sample solution. Check [this tutorial](https://sagemaker-workshop.com/custom/containers.html) for more details.

The 2nd docker image serves as the runtime environment for the LD endpoint. This docker image uses the 1st docker image as the base image. The reason why we cannot use the 1st docker image directly for the LD endpoint is that the container-based LD function requires an additional library called `awslambdaric`. This LD function also requires a specific entry point and a specific command to run. Check [this article](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html#images-create-from-alt) for more details.
Let's start by creating an SM notebook instance and clone this repository to the notebook environment.

### Develop locally

The local environment here means the local SM notebook instance. Open `exp_nbs/exp-local-sm.ipynb` and run through the notebook. You might need to modify the scripts in the `container/code` folder.

The reason why we need to develop in the local environment first is to debug faster because running the SM processing job and SM training job takes a lot of time.

### Deploy to SM endpoint

Open `exp_nbs/exp-real-sm.ipynb` and run through the notebook. You might need to modify the scripts in the `container/code` folder. This step runs similar code to the previous step.

This notebook already included the code for testing the SM endpoint.

### Deploy to LD endpoint

Open `exp_nbs/exp-ml-gateway.ipynb` and run through the notebook. You might need to modify the script `container/code/lambda_handler.py`. This step runs similar code to the previous step.

This notebook already included the code for testing the LD endpoint and the API endpoint.

### Develop SF

Open `exp_nbs/exp-step-functions.ipynb` and run through the notebook.

## Setup SageMaker Studio project

<!-- MARKDOWN LINKS & IMAGES -->

[architecture]: /media/architecture.png
