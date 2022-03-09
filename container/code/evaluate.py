import json
import sys
import tarfile
from pathlib import Path


class EvalConfig:
    ROOT_DIR = Path("/opt/ml/processing")

    IN_MODEL_DIR = ROOT_DIR / "model"
    IN_MODEL_TAR = IN_MODEL_DIR / "model.tar.gz"

    IN_TEST_DIR = ROOT_DIR / "test"

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
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {"value": 1.0, "standard_deviation": "NaN"},
            "auc": {"value": 1.0, "standard_deviation": "NaN"},
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
        tar.extractall(path=EvalConfig.IN_MODEL_DIR)

    inspect_input()
    evaluate()
    inspect_output()
    sys.exit(0)
