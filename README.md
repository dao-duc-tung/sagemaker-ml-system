<!-- PROJECT LOGO -->
<br />
<p align="center">
  <img src="media/banner.png" alt="Logo" width="300" height="100">

  <h3 align="center">SageMaker ML System</h3>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#model-development">Model development</a></li>
    <li><a href="#model-deployment">Model deployment</a></li>
    <li><a href="#monitoring-and-maintenance">Monitoring and maintenance</a></li>
    <li><a href="#improvement">Improvement</a></li>
    <li><a href="#faq">FAQ</a></li>
    <li><a href="#other-solutions">Other solutions</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

## About the project

This solution customizes the project [Amazon SageMaker Safe Deployment Pipeline](https://github.com/aws-samples/amazon-sagemaker-safe-deployment-pipeline).

The purpose of the project is to build an end-to-end ML system. The system has some main functionalities such as:

- Customizable docker image for different steps of the ML lifecycle
- Support 2-stage deployment: dev and prod
- Enable data capturing at endpoints
- Feature drift detection at endpoints
- System metric monitoring and alarm
- Enable training and deployment pipelines auto-rerun
- Auto-scaling and auto-balance load at endpoints
- Support canary deployment strategy with rollback on error

Below is the summary of this project's requirements.

### Functional requirements

- Given a numerical dataset, build a multiclass classification model
- Build a training pipeline
- Build a deployment pipeline
- Store inference results in a structured database

### Non-functional requirements

- Explain
  - Steps to process data
  - Steps to train and select the best model
- Introduce various technology stacks to supplement the above code to run
- Suggest
  - Pipeline exception, weaknesses -> How to handle
  - Monitoring metrics + Troubleshooting steps

## Model development

Given a dataset in [data/full/train.csv](data/full/train.csv) file, train a model to predict the outcome in the `target` column. Note that the data in 2 files [data/train.csv](data/train.csv) and [data/test.csv](data/test.csv) are the smaller sets of the original data to quickly test the program code.

### Prerequisites

To run the model development notebooks, we need to set up the local environment by using Anaconda as below.

```bash
conda create --name sagemaker python=3.8
conda activate sagemaker
conda install -c conda-forge jupyterlab
pip install -r requirements.txt
```

### Model development notebooks

Please refer to the folder [model_dev_nbs](model_dev_nbs) for the process of developing the ML model and the analytics behind each decision. Below is the summary of the model development process.

- [model_dev_nbs/exp-01-explore-data.ipynb](model_dev_nbs/exp-01-explore-data.ipynb)
  - Sanitize data
  - Explore data by using feature statistics, multivariate statistics
  - Detect data issues: data consistency, labeling bias, correlated data, missing data
- [model_dev_nbs/exp-02-clean-data.ipynb](model_dev_nbs/exp-02-clean-data.ipynb)
  - Clean data and fix data issues: fill missing values, drop highly correlated features, perform under-sampling for majority classes and over-sampling for minority classes
- [model_dev_nbs/exp-03-training-01.ipynb](model_dev_nbs/exp-03-training-01.ipynb)
  - Use `PyCaret` to try different ML models quickly to have an idea of how they perform on our data
- [model_dev_nbs/exp-03-training-02.ipynb](model_dev_nbs/exp-03-training-02.ipynb)
  - Use `PyCaret` to perform data preprocessing on our data and train the final model
- [model_dev_nbs/exp-04-sagemaker-data.ipynb](model_dev_nbs/exp-04-sagemaker-data.ipynb)
  - Manipulate data for building the ML system.

## Model deployment

### Architecture

![architecture][architecture]

### Component Details

- **CodePipeline**: defines several stages to go from source code to the creation of the API endpoint.
- **CodeBuild**: builds artifacts like CloudFormation parameters and defines StepFunction.
- **S3**: stores created artifacts and model data.
- **CloudFormation**: creates resources in an infrastructure-as-code manner.
- **Step Functions**: orchestrates SageMaker training and processing jobs.
- **SageMaker**: trains and deploys ML model.
- **CodeDeploy**: automates shifting traffic between two Lambda functions.
- **API Gateway**: creates an HTTPS REST API endpoint for the Lambda functions that invoke deployed SageMaker endpoint.
- **Key Management Service (KMS)**: encrypts data and artifacts.
- **Secrets Manager**: stores your GitHub Access Token.
- **Simple Notification Service (SNS)**: notifies you when CodeDeploy has successfully deployed the API, and to receive alerts for retraining and drift detection (signing up for these notifications is optional).
- **CloudWatch event rules**: schedules the pipeline to run every month, and triggers the pipeline to run when SageMaker Model Monitor detects the feature drift.

### Architecture summary

In the folder [container/code](container/code), we developed all the necessary code for data preparation, model training, and model evaluation. When a new commit is pushed to the repository, the CodePipeline is triggered. The CodePipeline pulls the source code and runs the Build stage. This Build stage has 2 steps:

- Step 1: Build the CloudFormation templates and their parameters
- Step 2: Create the Step Functions workflow for the model training step, and at the same time, create a few lambda functions as the helper functions to support the Step Functions workflow

Next, CodePipeline runs the Step Functions workflow to train and evaluate the model. Then, the dev-stage deployment is triggered to deploy the model to a SageMaker endpoint for the development deployment. This stage waits for our manual approval to proceed. If we approve, the next step deploys the model to the production environment.

When deploying the model to the production environment, we also deploy some resources like:

- A Lambda deployment application to support Canary deployment strategy and rollback on error
- A RESTful API exposed over the internet
- Endpoint auto-scaling policies, monitoring configurations for feature drift detection, etc.

### Data preparation

This section uploads our data to S3.

1. Create a bucket in S3 to store our data. Throughout the entire project, I use `t5-engine-bucket` as the name of the S3 bucket. You need to change this name, and update it in every step where you are required to provide the data bucket.
1. Upload 2 files [data/train.csv](data/train.csv) and [data/test.csv](data/test.csv) to `t5-engine-bucket/training_data/dummy/train` and `t5-engine-bucket/training_data/dummy/test` folders in S3, respectively
1. Upload 2 files [data/full/train.csv](data/full/train.csv) and [data/test.csv](data/test.csv) to `t5-engine-bucket/training_data/full/train` and `t5-engine-bucket/training_data/full/test` folders in S3, respectively

### Docker image preparation

This section creates several python scripts for some steps including model training, model evaluation, and model serving. This section is done in a SageMaker notebook instance.

The input of this section is the training code, evaluation code, and the model serving code of our ML model. The output of this section is one docker image that wraps all of these scripts.

This docker image serves as a multi-functional docker image for model training, model evaluation, and model serving. Combining all of these scripts simplifies the development process without creating several docker images for each script for the demo purpose of this project. In reality, we should create different docker images for each step.

Firstly, we create a Docker repository in the AWS ECR service named `sagemaker-t5`.

Secondly, we create a SageMaker notebook instance and clone this repo to the notebook environment. The folder [sm_nbs](sm_nbs) contains all the notebooks we need. The folder [container](container) contains all the source code that is put into the docker image.

- [sm_nbs/exp-local-sm.ipynb](sm_nbs/exp-local-sm.ipynb): test the SageMaker jobs in the notebook local environment first, because running the real SageMaker jobs takes time to spin up the computing resources.
- [sm_nbs/exp-real-sm.ipynb](sm_nbs/exp-real-sm.ipynb): test the SageMaker jobs by using real SageMaker processing job, SageMaker training job, and SageMaker endpoint.

### SageMaker Studio Organization template deployment

This section deploys the ML model to the production environment. This section uses the SageMaker Studio project as the development environment.

The input of this section is the output of the previous section which is the docker image. The output of this section is the API endpoint that is consumed by the end application.

Let's start with the deployment of the SageMaker Studio Organization template. This template is used to create a SageMaker Studio project later.

1. Prepare your LINUX environment

   - Install `rsync`, `zip`
   - Install `aws-cli` in your python environment
   - Clone this repo to your LINUX system

1. [Optional] Update the [studio.yml](studio.yml) and [pipeline.yml](pipeline.yml) if needed

   - [studio.yml](studio.yml) creates the SageMaker Studio organization template and other resources.
   - [pipeline.yml](pipeline.yml) creates the following main resources for the SageMaker Studio project when you create a new SageMaker Studio project.
     - S3 Bucket and S3 Bucket Policy
     - SNS Topic, SNS Topic Policy, and SNS Subscription
     - A CodeCommit Repository, a CodeBuild Project, and a CodePipeline
     - Some Event Rules to retrain the model, schedule the CodePipeline

1. Validate the CloudFormation templates by running
   ```bash
   bash scripts/validate-tpl.sh
   ```
1. Build the SageMaker Studio Organization template
   ```bash
   bash scripts/build.sh sagemaker-safe-deployment-tpl sagemaker-safe-deployment-tpl ap-southeast-1 <STUDIO_ROLE_NAME>
   ```
   - `sagemaker-safe-deployment-tpl`: an S3 bucket used to store the organization template artifacts
   - `sagemaker-safe-deployment-tpl`: CloudFormation stack name that creates this SageMaker Studio organization template
   - `ap-southeast-1`: the region that we are working in
   - <STUDIO_ROLE_NAME>: the SageMaker Execution Role. This role is usually created for the first time when you use SageMaker Studio

### SageMaker Studio project creation

Open SageMaker Studio, and create a new SageMaker Studio project using the Organization template named SageMaker Safe Deployment template created in the previous section with the following information. Note that this step creates a CloudFormation stack named `SC-<account id>-pp-<...>` by using [pipeline.yml](pipeline.yml) template.

- Name (16 chars): `sd-t5-01`
- Model name (10 chars): `sd-t5`
- S3 Bucket for Dataset: `t5-engine-bucket` (the S3 bucket created in the section [Data preparation](#data-preparation))
- Image repo: `sagemaker-t5` (the ECR repo to store docker image)
- Unique prefix to bind the components (10 chars): `smsd` (MUST be the same as PREFIX in [scripts/build.sh](scripts/build.sh))
- Git Branch: `main`
- Email: `example@example.com` (to receive alert)
- Train and build timeout: `45` (minutes)

In the newly created SageMaker Studio project, clone the CodeCommit repository to the SageMaker Studio project environment.

### System pipeline

Open [notebook/mlops.ipynb](notebook/mlops.ipynb) notebook to run through the system pipeline. Below is its description. The system pipeline consists of several stages.

1. **Source stage**. When a new commit is pushed to the main branch or `data-source.zip` is uploaded to the pre-defined S3 folder, the system pipeline will be triggered.

![source-stage][source-stage]

2. **Build stage**. This stage consists of two steps (see [pipeline.yml](pipeline.yml)).

   1. **Step 1**: Build templates. This step runs [model/run_pipeline.yml] to do several tasks (see [model/buildspec.yml](model/buildspec.yml)).

      - Create `workflow-graph.yml`: this CloudFormation stack creates the StepFunction to prepare data, train, and evaluate the model. See **Train stage** below.
      - Create `workflow-graph.json`: Parameters of the CloudFormation stack `workflow-graph.yml`.
      - Create `sagemaker-custom-resource.json`: Parameters of the CloudFormation stack [custom_resource/sagemaker-custom-resource.yml](custom_resource/sagemaker-custom-resource.yml). See **Step 2** below.
      - Create `deploy-model-dev.json`: Parameters of the CloudFormation stack [assets/deploy-model-dev.yml](assets/deploy-model-dev.yml). See **Dev deployment stage** below.
      - Create `deploy-model-prd.json`: Parameters of the CloudFormation stack [assets/deploy-model-prd.yml](assets/deploy-model-prd.yml). See **Prod deployment stage** below.
      - Package CloudFormation stack `workflow-graph.yml` with its parameters.
      - Package CloudFormation stack [custom_resource/sagemaker-custom-resource.yml](custom_resource/sagemaker-custom-resource.yml) with its parameters.

   1. **Step 2**: This step updates the `workflow-graph.yml` and [custom_resource/sagemaker-custom-resource.yml](custom_resource/sagemaker-custom-resource.yml) packaged CloudFormation stacks. The [custom_resource/sagemaker-custom-resource.yml](custom_resource/sagemaker-custom-resource.yml) CloudFormation stack creates the following main resources.

      - An Lambda function to prepend header to a batch transform job. See [custom_resource/sagemaker_add_transform_header.py](custom_resource/sagemaker_add_transform_header.py).
      - An Lambda function to create a SageMaker experiment and trial. See [custom_resource/sagemaker_create_experiment.py](custom_resource/sagemaker_create_experiment.py).
      - An Lambda function to query evaluation job to return results. See [custom_resource/sagemaker_query_evaluation.py](custom_resource/sagemaker_query_evaluation.py).
      - An Lambda function to query processing job to return drift. See [custom_resource/sagemaker_query_drift.py](custom_resource/sagemaker_query_drift.py).

![build-stage][build-stage]

3. **Train stage**. The **Build stage** just creates the StepFunction. This **Train stage** will run that StepFunction that has the following steps (see [model/run_pipeline.yml](model/run_pipeline.yml)).

   - Create a _baseline_ for the model monitor using a SageMaker processing job
   - Train a model using a SageMaker training job
   - Save the trained model
   - Query the evaluation results using the query-evaluation Lambda function created by [custom_resource/sagemaker-custom-resource.yml](custom_resource/sagemaker-custom-resource.yml) CloudFormation stack
   - Verify if the evaluation results meet some criteria

![train-stage][train-stage]

Below are the steps of the defined StepFunction.

![detailed-sf][detailed-sf]

4. **Dev deployment stage**. This stage consists of two steps.

   1. Step 1: Deploy Model Dev. This step updates the `deploy-model-dev.yml` CloudFormation stack. This CloudFormation stack creates the following resources.

      - An SageMaker model
      - An SageMaker endpoint configuration
      - An SageMaker endpoint

   1. Step 2: After the SageMaker endpoint is deployed, this step waits for you to manually approve the changes to move to the next stage.

![dev-deployment-stage][dev-deployment-stage]

5. **Prod deployment stage**. This stage updates the `deploy-model-prd.yml` CloudFormation stack. This CloudFormation stack creates the following main resources.

   - An SageMaker model
   - An SageMaker endpoint configuration
   - An SageMaker endpoint
   - An Lambda function (Lambda endpoint). See [api/app.py]. This Lambda function supports gradual deployment. This gradual deployment creates some resources like a `CodeDeploy::Application`, a `CodeDeploy::DeploymentGroup`, and an implicit API Gateway endpoint. Read more [here](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/automating-updates-to-serverless-apps.html).
   - An Lambda function to perform the checking process of pre-shifting traffic to the Lambda endpoint. See [api/pre_traffic_hook.py](api/pre_traffic_hook.py). Read more about `hooks` section for an AWS Lambda deployment [here](https://docs.aws.amazon.com/codedeploy/latest/userguide/reference-appspec-file-structure-hooks.html#appspec-hooks-lambda)
   - An Lambda function to perform the checking process of post-shifting traffic to the Lambda endpoint. See [api/post_traffic_hook.py](api/post_traffic_hook.py)
   - An SageMaker monitoring schedule to run on requests' data
   - An CloudWatch alarm to track for the feature drift on requests' data
   - Two CloudWatch alarms to track the Lambda endpoint deployment's status
   - An `ApplicationAutoScaling::ScalableTarget` for the SageMaker endpoint
   - An `ApplicationAutoScaling::ScalingPolicy` for the SageMaker endpoint
   - This CloudFormation stack outputs the API Gateway endpoint URL

![prod-deployment-stage][prod-deployment-stage]

CodeDeploy will perform a canary deployment with rollback on error. The error is watched by the CloudWatch alarms.

![traffic-shifting-progress][traffic-shifting-progress]

The CodePipeline instance is configured with CloudWatch Events to start the pipeline for retraining when the drift detection triggers specific metric alarms.

### System pipeline running time

- Full Pipeline: 35 minutes
- Source stage: Few seconds
- Build stage: 5 minutes
- Train stage: 15 minutes
- Dev Deployment stage: 10 minutes
- Prod Deployment stage: 10 minutes
- Monitoring schedule: every hour

### SageMaker environment cleanup

1. Delete all the CF stacks one by one starting from the top one. Don't delete them all at once.
1. Delete all related S3 buckets
1. Delete the SM Studio project
   ```bash
   bash scripts/rm_prj.sh <sm-studio-project-name>
   ```

## Monitoring and maintenance

An ML system can fail because of many reasons such as:

- Model failure
- Deliberate system abuse
- Resource overload
- Excessive latency
- Poor security
- Downtime or crashing

To prevent, detect and fix the failure as fast as possible, we need to perform monitoring and maintenance in our ML system. Monitoring is the process of tracking statistics about an ML system to understand its environment and behavior. Maintenance is the process of updating a deployed ML system to improve performance or correct for failure.

### Monitoring

In the monitoring context, we usually need to monitor system infrastructure, data pipelines, and model performance.

- System infrastructure
  - Are we prepared if code dependency changes? Solution: subscribe to the dependencies to update for the most critical changes, use tools like GitHub's Dependabot to monitor and track dependency changes.
  - Are we meeting uptime, latency, auditing, and compliance requirements? Solution: log for different services/applications, log as much information as we can (date/time, input data, output prediction, true label, model version, model hyperparameters, number of requests served, historical data used as training data, min/max/average serving times, the threshold to convert probabilities to label, etc.)
- Data pipelines
  - Data validation: log data structure, check NaN in data.
  - Data distribution: log data statistics.
  - Data dependencies: track data dependencies (input sources), versioning preprocessing steps.
- Model performance
  - Output performance statistics.
  - Model auditing and interpretability (if reasoning the model matters) by using Local interpretable model-agnostic explanations (LIME), SHapley Additive exPlanation (SHAP).

### Maintenance

Maintenance is a cycle that consists of model monitoring, model training, model offline evaluation, and model online evaluation steps. In the maintenance context, we usually need to watch for performance validation, shadow release, and model health.

- Performance validation: answer the question "Why are we deploying a new model?". The reasons can be latency concerns, computing resource concerns, more powerful architecture needs, feature drift, etc.
- Model health: apply deployment strategies with rollback on error.

## Improvement

Currently, we need to perform data preparation manually and upload preprocessed data to S3 to retrain the model. To automate this data preparation task, we can create a pipeline in CodePipeline. This pipeline has 2 stages:

- The first stage detects the change in an S3 folder where we store the data
- The second stage runs a StepFunction. The StepFunction has steps as below:
  - Step 1: Preprocess data by running a SageMaker processing job. The input is the S3 folder where we store the data. The output is an S3 folder that stores preprocessed data.
  - Step 2: Run a Lambda function to trigger the retraining process. This Lambda function firstly zips the preprocessed data with the model hyperparameters JSON file into a file named `data-source.zip`. This file will then be uploaded to a pre-defined S3 folder of the system pipeline to trigger the retraining process. The inputs are an S3 folder that stores the preprocessed data and the model hyperparameters in JSON format. The output is an S3 folder used to trigger the system pipeline.

Finally, to trigger the retraining process, we just need to upload the raw data to an S3 folder without preprocessing it manually or on the local machine.

## FAQ

### How to use the API endpoint in production?

The end-user (web app, mobile app, etc.) sends a request to the API endpoint in JSON format. Please use the Postman collection in [postman](postman) folder.

### How to customize the docker image for the model training, model evaluation, and model serving processes?

The docker images used for training, evaluation, and serving are identical. Please check the file [container/Dockerfile](container/Dockerfile).

### How to customize the code for the model training, model evaluation, and model serving processes?

The training code is stored in the [container/code/train](container/code/train) file.

The evaluation code is stored in the [container/code/evaluate.py](container/code/evaluate.py) file.

The serving code is stored in the [container/code/predictor.py](container/code/predictor.py) file.

## Other solutions

The alternative solution for this SageMaker end-to-end ML system is using Kubeflow built on top of a Kubernetes cluster. Kubeflow is dedicated to making deployments of ML workflows on Kubernetes simple, portable and scalable. It provides a straightforward way to deploy ML systems to a Kubernetes infrastructure.

Similarly to SageMaker, Kubeflow includes services to create and manage Jupyter notebooks, provides a custom TensorFlow training job operator, TensorFlow Serving to deploy ML models, provides custom ML pipelines (same as StepFunction workflow or SageMaker training pipeline) to manage end-to-end ML workflows, supports multiple ML frameworks (PyTorch, XGBoost, etc.).

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

Tung Dao - [LinkedIn](https://www.linkedin.com/in/tungdao17/)

Project Link: [https://github.com/dao-duc-tung/sagemaker-ml-system](https://github.com/dao-duc-tung/sagemaker-ml-system)

## Acknowledgements

- [PyCaret](https://pycaret.gitbook.io/docs/)
- [SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [Kubeflow](https://www.kubeflow.org/docs/)

<!-- MARKDOWN LINKS & IMAGES -->

[architecture]: media/architecture.png
[source-stage]: media/source-stage.png
[build-stage]: media/build-stage.png
[train-stage]: media/train-stage.png
[detailed-sf]: media/detailed-sf.png
[dev-deployment-stage]: media/dev-deployment-stage.png
[prod-deployment-stage]: media/prod-deployment-stage.png
[traffic-shifting-progress]: media/traffic-shifting-progress.png
