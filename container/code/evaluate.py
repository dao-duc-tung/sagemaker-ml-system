import json
from pathlib import Path
import tarfile
import sys

import pandas as pd
from pycaret.regression import load_model, predict_model, load_config
from sklearn.metrics import accuracy_score, f1_score


class EvalConfig:
    ROOT_DIR = Path("/opt/ml/processing")

    IN_MODEL_DIR = ROOT_DIR / "model"
    IN_MODEL_TAR = IN_MODEL_DIR / "model.tar.gz"

    IN_TEST_DIR = ROOT_DIR / "test"
    IN_TEST_CSV = IN_TEST_DIR / "test.csv"

    OUT_EVAL_DIR = ROOT_DIR / "evaluation"
    OUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_EVAL_JSON = OUT_EVAL_DIR / "eval.json"


def inspect_input():
    logger.info(f"Start inspect_input")
    files = list_dir(EvalConfig.IN_MODEL_DIR)
    logger.info(f"{EvalConfig.IN_MODEL_DIR.as_posix()}: {files}")

    files = list_dir(EvalConfig.IN_TEST_DIR)
    logger.info(f"{EvalConfig.IN_TEST_DIR.as_posix()}: {files}")


def evaluate():
    logger.info(f"Start evaluate")
    test_df = pd.read_csv(EvalConfig.IN_TEST_CSV)

    logger.info(f"Pycaret load_config")
    config_path = EvalConfig.IN_MODEL_DIR / "final-config"
    load_config(config_path.as_posix())

    logger.info(f"Pycaret load_model")
    model_path = EvalConfig.IN_MODEL_DIR / "final-model"
    saved_model = load_model(model_path.as_posix())

    logger.info(f"Pycaret predict_model")
    pred_df = predict_model(saved_model, data=test_df)

    logger.info(f"Compute f1, accuracy")
    y_test = pred_df["target"]
    y_pred = pred_df["Label"]
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)

    # MUST follow Sagemaker convention
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {"value": acc, "standard_deviation": "NaN"},
            "weighted_f1": {"value": f1, "standard_deviation": "NaN"},
        }
    }
    logger.info(f"report_dict: {report_dict}")
    logger.info(f"Save report_dict to: {EvalConfig.OUT_EVAL_JSON}")
    with open(EvalConfig.OUT_EVAL_JSON.as_posix(), "w") as f:
        f.write(json.dumps(report_dict))


def inspect_output():
    logger.info(f"Start inspect_output")
    files = list_dir(EvalConfig.OUT_EVAL_DIR)
    logger.info(f"{EvalConfig.OUT_EVAL_DIR.as_posix()}: {files}")


if __name__ == "__main__":
    sys.path.append("/opt/program")
    from utils import *

    model_dir = EvalConfig.IN_MODEL_DIR
    model_path = EvalConfig.IN_MODEL_TAR
    if not model_path.exists():
        logger.error(f"{model_path} does not exist!")
        sys.exit(1)

    logger.info(f"Extracting model from path: {model_path}")
    with tarfile.open(model_path.as_posix()) as tar:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=EvalConfig.IN_MODEL_DIR)

    inspect_input()
    evaluate()
    inspect_output()
    sys.exit(0)
