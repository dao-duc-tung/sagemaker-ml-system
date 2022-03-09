import argparse
import json
import os

import boto3
import sagemaker
from sagemaker.image_uris import retrieve
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    Processor,
    ScriptProcessor,
)
from stepfunctions import steps
from stepfunctions.inputs import ExecutionInput
from stepfunctions.workflow import Workflow

processing_instance_count = 1
processing_instance_type = "ml.m5.2xlarge"
training_instance_type = "ml.m5.2xlarge"


def create_experiment_step(create_experiment_function_name):
    create_experiment_step = steps.compute.LambdaStep(
        "Create Experiment",
        parameters={
            "FunctionName": create_experiment_function_name,
            "Payload": {
                "ExperimentName.$": "$.ExperimentName",
                "TrialName.$": "$.TrialName",
            },
        },
        result_path="$.CreateTrialResults",
    )
    return create_experiment_step


def create_baseline_step(input_data, execution_input, region, role):
    # Define the enviornment
    dataset_format = DatasetFormat.csv()
    env = {
        "dataset_format": json.dumps(dataset_format),
        "dataset_source": "/opt/ml/processing/input/baseline_dataset_input",
        "output_path": "/opt/ml/processing/output",
        "publish_cloudwatch_metrics": "Disabled",  # Have to be disabled from processing job?
    }

    # Define the inputs and outputs
    inputs = [
        ProcessingInput(
            source=input_data["BaselineUri"],
            destination="/opt/ml/processing/input/baseline_dataset_input",
            input_name="baseline_dataset_input",
        ),
    ]
    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=execution_input["BaselineOutputUri"],
            output_name="monitoring_output",
        ),
    ]

    # Get the default model monitor container
    monor_monitor_container_uri = retrieve(
        region=region, framework="model-monitor", version="latest"
    )

    # Create the processor
    monitor_analyzer = Processor(
        image_uri=monor_monitor_container_uri,
        role=role,
        instance_count=1,
        instance_type=processing_instance_type,
        max_runtime_in_seconds=1800,
        env=env,
    )

    # Create the processing step
    baseline_step = steps.sagemaker.ProcessingStep(
        "Baseline Job",
        processor=monitor_analyzer,
        job_name=execution_input["BaselineJobName"],
        inputs=inputs,
        outputs=outputs,
        experiment_config={
            "ExperimentName": execution_input["ExperimentName"],  # '$.ExperimentName',
            "TrialName": execution_input["TrialName"],
            "TrialComponentDisplayName": "Baseline",
        },
        tags={
            "GitBranch": execution_input["GitBranch"],
            "GitCommitHash": execution_input["GitCommitHash"],
            "DataVersionId": execution_input["DataVersionId"],
        },
    )

    # Add the catch
    baseline_step.add_catch(
        steps.states.Catch(
            error_equals=["States.TaskFailed"],
            next_step=steps.states.Fail(
                "Baseline failed",
                cause="SageMakerBaselineJobFailed",
            ),
        )
    )
    return baseline_step


def create_training_step(
    image_uri,
    hyperparameters,
    input_data,
    output_data,
    execution_input,
    query_evaluation_function_name,
    query_training_function_name,
    region,
    role,
):
    # Create the estimator
    est = sagemaker.estimator.Estimator(
        image_uri,
        role,
        instance_count=1,
        instance_type=training_instance_type,
        hyperparameters=hyperparameters,
        output_path=output_data[
            "ModelOutputUri"
        ],  # NOTE: Can't use execution_input here
        max_run=1200,  # timeout in seconds
        use_spot_instances=True,
        max_wait=1200,  # <= max_run
    )

    # Specify the data source
    s3_input_train = sagemaker.inputs.TrainingInput(s3_data=input_data["TrainingUri"])
    data = {"train": s3_input_train}

    # Create the training step
    training_step = steps.TrainingStep(
        "Training Job",
        estimator=est,
        data=data,
        job_name=execution_input["TrainingJobName"],
        experiment_config={
            "ExperimentName": execution_input["ExperimentName"],
            "TrialName": execution_input["TrialName"],
            "TrialComponentDisplayName": "Training",
        },
        tags={
            "GitBranch": execution_input["GitBranch"],
            "GitCommitHash": execution_input["GitCommitHash"],
            "DataVersionId": execution_input["DataVersionId"],
        },
        result_path="$.TrainingResults",
    )

    # Add the catch
    training_step.add_catch(
        steps.states.Catch(
            error_equals=["States.TaskFailed"],
            next_step=steps.states.Fail(
                "Training failed",
                cause="SageMakerTrainingJobFailed",
            ),
        )
    )

    # Must follow the training test
    model_step = steps.sagemaker.ModelStep(
        "Save Model",
        input_path="$.TrainingResults",
        model=training_step.get_expected_model(),
        model_name=execution_input["TrainingJobName"],
        result_path="$.ModelStepResults",
    )

    # Evaluation step
    model_evaluation_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python"],
        role=role,
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        max_runtime_in_seconds=1200,
    )
    model_artifacts_uri = training_step.output()["TrainingResults"]["ModelArtifacts"][
        "S3ModelArtifacts"
    ]
    evaluation_step = steps.sagemaker.ProcessingStep(
        "Model Evaluation Job",
        processor=model_evaluation_processor,
        job_name=execution_input["EvaluationJobName"],
        # the evaluate.py file is already inside the image
        container_entrypoint=["python", "/opt/program/evaluate.py"],
        inputs=[
            ProcessingInput(
                input_name="model_artifacts_input",
                source=model_artifacts_uri,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                input_name="test_dataset_input",
                source=input_data["TestUri"],
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=execution_input["EvaluationOutputUri"],
            ),
        ],
        experiment_config={
            "ExperimentName": execution_input["ExperimentName"],  # '$.ExperimentName',
            "TrialName": execution_input["TrialName"],
            "TrialComponentDisplayName": "Evaluation",
        },
        tags={
            "GitBranch": execution_input["GitBranch"],
            "GitCommitHash": execution_input["GitCommitHash"],
            "DataVersionId": execution_input["DataVersionId"],
        },
        result_path="$.EvaluationResults",
    )
    evaluation_step.add_catch(
        steps.states.Catch(
            error_equals=["States.TaskFailed"],
            next_step=steps.states.Fail(
                "Evaluation failed",
                cause="SageMakerEvaluationJobFailed",
            ),
        )
    )

    # Query the evaluation step
    evaluation_query_step = steps.compute.LambdaStep(
        "Query Evaluation Results",
        parameters={
            "FunctionName": query_evaluation_function_name,
            "Payload": {"EvaluationJobName.$": "$.EvaluationJobName"},
        },
        result_path="$.QueryEvaluationResults",
    )

    check_accuracy_fail_step = steps.states.Fail(
        "Model Performance Is Not Acceptable", comment="AUC is smaller than threshold"
    )

    # Query the training step to store artifacts
    training_query_step = steps.compute.LambdaStep(
        "Query Training Results",
        parameters={
            "FunctionName": query_training_function_name,
            "Payload": {"TrainingJobName.$": "$.TrainingJobName"},
        },
        result_path="$.QueryTrainingResults",
    )
    training_query_step.add_retry(
        steps.states.Retry(
            error_equals=["States.ALL"],
            interval_seconds=3,
            max_attempts=3,
            backoff_rate=2.0,
        )
    )

    # TODO: Update query method to query validation error using better result path
    threshold_rule = steps.choice_rule.ChoiceRule.NumericGreaterThan(
        variable=evaluation_query_step.output()["QueryEvaluationResults"]["Payload"][
            "results"
        ]["ProcessingMetrics"]["binary_classification_metrics"]["auc"]["value"],
        value=0.7,
    )

    check_accuracy_step = steps.states.Choice("AUC > 0.7")

    check_accuracy_step.add_choice(rule=threshold_rule, next_step=training_query_step)
    check_accuracy_step.default_choice(next_step=check_accuracy_fail_step)

    # Return the chain of these steps
    return steps.states.Chain(
        [
            training_step,
            model_step,
            evaluation_step,
            evaluation_query_step,
            check_accuracy_step,
        ]
    )


def create_graph(create_experiment_step, baseline_step, training_step):
    sagemaker_jobs = steps.states.Parallel("SageMaker Jobs")
    sagemaker_jobs.add_branch(baseline_step)
    sagemaker_jobs.add_branch(training_step)

    # Do we need specific failure for the jobs for group?
    sagemaker_jobs.add_catch(
        steps.states.Catch(
            error_equals=["States.TaskFailed"],
            next_step=steps.states.Fail(
                "SageMaker Jobs failed",
                cause="SageMakerJobsFailed",
            ),
        )
    )

    # Return the workflow graph
    return steps.states.Chain([create_experiment_step, sagemaker_jobs])


def get_custom_resource_config(sagemaker_project_id, project_prefix):
    return {
        "Parameters": {
            "SageMakerProjectId": sagemaker_project_id,
            "ProjectPrefix": project_prefix,
        },
        "Tags": {
            "SageMakerProjectId": sagemaker_project_id,
        },
    }


def get_dev_config(
    model_name, job_id, role, image_uri, kms_key_id, sagemaker_project_id
):
    return {
        "Parameters": {
            "ImageRepoUri": image_uri,
            "ModelName": model_name,
            "TrainJobId": job_id,
            "DeployRoleArn": role,
            "ModelVariant": "dev",
            "KmsKeyId": kms_key_id,
        },
        "Tags": {
            "mlops:model-name": model_name,
            "mlops:stage": "dev",
            "SageMakerProjectId": sagemaker_project_id,
        },
    }


def get_prd_config(
    model_name,
    job_id,
    role,
    image_uri,
    kms_key_id,
    notification_arn,
    sagemaker_project_id,
    project_prefix,
    image_lambda_uri,
):
    dev_config = get_dev_config(
        model_name, job_id, role, image_uri, kms_key_id, sagemaker_project_id
    )
    prod_params = {
        "SageMakerProjectId": sagemaker_project_id,
        "ProjectPrefix": project_prefix,
        "ImageLambdaUri": image_lambda_uri,
        "ModelVariant": "prd",
        "ScheduleMetricName": "feature_baseline_drift_total_amount",
        "ScheduleMetricThreshold": str("0.20"),
        "NotificationArn": notification_arn,
        "FormattedJobId": str(job_id).replace("-", ""),
    }
    prod_tags = {"mlops:stage": "prd", "SageMakerProjectId": sagemaker_project_id}
    return {
        "Parameters": dict(dev_config["Parameters"], **prod_params),
        "Tags": dict(dev_config["Tags"], **prod_tags),
    }


def get_pipeline_execution_id(pipeline_name, codebuild_id):
    codepipeline = boto3.client("codepipeline")
    response = codepipeline.get_pipeline_state(name=pipeline_name)
    for stage in response["stageStates"]:
        for action in stage["actionStates"]:
            # Return the matching stage with the same external id
            if (
                "latestExecution" in action
                and "externalExecutionId" in action["latestExecution"]
                and action["latestExecution"]["externalExecutionId"] == codebuild_id
            ):
                return stage["latestExecution"]["pipelineExecutionId"]


def get_pipeline_revisions(pipeline_name, execution_id):
    codepipeline = boto3.client("codepipeline")
    response = codepipeline.get_pipeline_execution(
        pipelineName=pipeline_name, pipelineExecutionId=execution_id
    )
    return dict(
        (r["name"], r["revisionId"])
        for r in response["pipelineExecution"]["artifactRevisions"]
    )


def main(
    git_branch,
    codebuild_id,
    pipeline_name,
    model_name,
    deploy_role,
    sagemaker_role,
    sagemaker_bucket,
    data_dir,
    output_dir,
    kms_key_id,
    workflow_role_arn,
    notification_arn,
    sagemaker_project_id,
    project_prefix,
    tags,
):
    # Define the function names
    create_experiment_function_name = "mlops-create-experiment"
    query_evaluation_function_name = "mlops-query-evaluation"
    query_training_function_name = "mlops-query-training"

    # Get the region
    region = boto3.Session().region_name
    print("region: {}".format(region))

    # Load the image uri and input data config
    with open(os.path.join(data_dir, "imageDetail.json"), "r") as f:
        image_detail = json.load(f)
        image_uri = image_detail["ImageUri"]
        image_lambda_uri = image_detail["ImageLambdaUri"]
    print("image_uri: {}".format(image_uri))
    print("image_lambda_uri: {}".format(image_lambda_uri))

    with open(os.path.join(data_dir, "inputData.json"), "r") as f:
        input_data = json.load(f)
        print("training uri: {}".format(input_data["TrainingUri"]))
        print("test uri: {}".format(input_data["TestUri"]))
        print("baseline uri: {}".format(input_data["BaselineUri"]))

    # Get the job id and source revisions
    job_id = get_pipeline_execution_id(pipeline_name, codebuild_id)
    revisions = get_pipeline_revisions(pipeline_name, job_id)
    git_commit_id = revisions["ModelSourceOutput"]
    data_verison_id = revisions["DataSourceOutput"]
    print("job id: {}".format(job_id))
    print("git commit: {}".format(git_commit_id))
    print("data version: {}".format(data_verison_id))

    # Set the output Data
    output_data = {
        "ModelOutputUri": "s3://{}/{}".format(sagemaker_bucket, model_name),
        "BaselineOutputUri": f"s3://{sagemaker_bucket}/{model_name}/monitoring/baseline/{model_name}-pbl-{job_id}",
        "EvaluationOutputUri": f"s3://{sagemaker_bucket}/{model_name}/evaluation/{model_name}-pbl-{job_id}",
    }
    print("output_data: {}".format(output_data))

    # Pass these into the training method
    hyperparameters = {}
    if os.path.exists(os.path.join(data_dir, "hyperparameters.json")):
        with open(os.path.join(data_dir, "hyperparameters.json"), "r") as f:
            hyperparameters = json.load(f)
            for i in hyperparameters:
                hyperparameters[i] = str(hyperparameters[i])
    print("hyperparameters: {}".format(hyperparameters))

    # Define the step functions execution input schema
    execution_input = ExecutionInput(
        schema={
            "GitBranch": str,
            "GitCommitHash": str,
            "DataVersionId": str,
            "ExperimentName": str,
            "TrialName": str,
            "BaselineJobName": str,
            "BaselineOutputUri": str,
            "TrainingJobName": str,
            "EvaluationJobName": str,
            "EvaluationOutputUri": str,
        }
    )

    # Create experiment step
    experiment_step = create_experiment_step(create_experiment_function_name)
    baseline_step = create_baseline_step(
        input_data, execution_input, region, sagemaker_role
    )
    training_step = create_training_step(
        image_uri,
        hyperparameters,
        input_data,
        output_data,
        execution_input,
        query_evaluation_function_name,
        query_training_function_name,
        region,
        sagemaker_role,
    )
    workflow_definition = create_graph(experiment_step, baseline_step, training_step)

    # Create the workflow as the model name
    workflow = Workflow(model_name, workflow_definition, workflow_role_arn)
    print("Creating workflow: {0}-{1}".format(model_name, sagemaker_project_id))

    # Create output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Write the workflow graph to json
    with open(os.path.join(output_dir, "workflow-graph.json"), "w") as f:
        f.write(workflow.definition.to_json(pretty=True))

    # Write the workflow graph to yml
    with open(os.path.join(output_dir, "workflow-graph.yml"), "w") as f:
        f.write(workflow.get_cloudformation_template())

    # Write the workflow inputs to file
    with open(os.path.join(output_dir, "workflow-input.json"), "w") as f:
        workflow_inputs = {
            "ExperimentName": "{}".format(model_name),
            "TrialName": "{}-{}".format(model_name, job_id),
            "GitBranch": git_branch,
            "GitCommitHash": git_commit_id,
            "DataVersionId": data_verison_id,
            "BaselineJobName": "{}-pbl-{}".format(model_name, job_id),
            "BaselineOutputUri": output_data["BaselineOutputUri"],
            "TrainingJobName": "{}-{}".format(model_name, job_id),
            "EvaluationJobName": "{}-evl-{}".format(model_name, job_id),
            "EvaluationOutputUri": output_data["EvaluationOutputUri"],
        }
        json.dump(workflow_inputs, f)

    # Write the dev & prod params for CFN
    with open(os.path.join(output_dir, "sagemaker-custom-resource.json"), "w") as f:
        config = get_custom_resource_config(sagemaker_project_id, project_prefix)
        json.dump(config, f)
    with open(os.path.join(output_dir, "deploy-model-dev.json"), "w") as f:
        config = get_dev_config(
            model_name, job_id, deploy_role, image_uri, kms_key_id, sagemaker_project_id
        )
        json.dump(config, f)
    with open(os.path.join(output_dir, "deploy-model-prd.json"), "w") as f:
        config = get_prd_config(
            model_name,
            job_id,
            deploy_role,
            image_uri,
            kms_key_id,
            notification_arn,
            sagemaker_project_id,
            project_prefix,
            image_lambda_uri,
        )
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="sagemaker_role",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )
    parser.add_argument("--codebuild-id", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--deploy-role", required=True)
    parser.add_argument("--sagemaker-role", required=True)
    parser.add_argument("--sagemaker-bucket", required=True)
    parser.add_argument("--kms-key-id", required=True)
    parser.add_argument("--git-branch", required=True)
    parser.add_argument("--workflow-role-arn", required=True)
    parser.add_argument("--notification-arn", required=True)
    parser.add_argument("--sagemaker-project-id", required=True)
    parser.add_argument("--project-prefix", required=True)
    args = vars(parser.parse_args())
    print("args: {}".format(args))
    main(**args)
