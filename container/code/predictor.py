import io
from pathlib import Path

import flask
import pandas as pd

from utils import *


class ServeConfig:
    OPT_ML_DIR = Path("/opt/ml")
    MODEL_DIR = OPT_ML_DIR / "model"
    MODELS_DIR = MODEL_DIR / "models"
    SCALERS_DIR = MODEL_DIR / "scalers"

    ASSETS_PATH = Path("./assets")
    ASSETS_PATH.mkdir(parents=True, exist_ok=True)
    AUDIO_PATH = ASSETS_PATH / "audio"
    AUDIO_PATH.mkdir(parents=True, exist_ok=True)
    META_PATH = ASSETS_PATH / "metadata"
    META_PATH.mkdir(parents=True, exist_ok=True)


class ScoringService(object):
    """
    A singleton for holding the model. This simply loads the model and holds it.
    It has a predict function that does a prediction based on the model and the input data.
    """

    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        cls.model = None
        return cls.model

    @classmethod
    def predict(cls, input: pd.DataFrame):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        model = cls.get_model()
        input_features = input.values
        # model.predict(input_features)
        output = None
        return output


app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(
            response="This predictor only supports CSV data",
            status=415,
            mimetype="text/plain",
        )

    print("Invoked with {} records".format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(data)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")
