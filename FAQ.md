# FAQ

## How to use the API endpoint in production?

The end service (web app, etc.) needs to send a request to the API endpoint in JSON format. The binary data must be formatted in a base64 string. Please refer to section `Test API endpoint` for more detail.

## How to customize the API endpoint request's format?

Whatever is sent to the API endpoint is forwarded to the LD endpoint. To customize the API endpoint request's format, we need to change the LD endpoint request's format.

The LD endpoint request's processing procedure is defined in the `container/code/lambda_handler.py` file. You need to update this file to adapt to the new request format.

Please refer to section `Customize Docker images`/`Deploy to LD endpoint` to develop, debug and test the LD endpoint before deployment.

## How to customize the docker image for the data preparation, model training, model evaluation, and model serving processes?

The docker images used for training, evaluation and serving are identical. Please check the file `container/Dockerfile`.

## How to customize the code for the data preparation, model training, model evaluation, and model serving processes?

The data preparation code is stored in the `container/code/prepare_data.py` file.

The training code is stored in the `container/code/train` file.

The evaluation code is stored in the `container/code/evaluate.py` file.

The serving code is stored in the `container/code/predictor.py` file.

Please refer to section `Customize Docker images`/`Develop locally` for more detail.
