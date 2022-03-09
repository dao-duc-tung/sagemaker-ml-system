import sys
from pathlib import Path


class PrepareDataConfig:
    ROOT_DIR = Path("/opt/ml/processing")
    INPUT_DIR = ROOT_DIR / "input"
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    IN_TRAIN_DIR = INPUT_DIR / "train"
    IN_TEST_DIR = INPUT_DIR / "test"
    TMP_DIR = ROOT_DIR / "tmp"

    OUT_TRAIN_DIR = ROOT_DIR / "train"
    OUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    OUT_TEST_DIR = ROOT_DIR / "test"
    OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)


def inspect_input():
    logger.info(f"Start inspect_input")
    files = list_dir(PrepareDataConfig.INPUT_DIR)
    logger.info(f"{PrepareDataConfig.INPUT_DIR.as_posix()}: {files}")

    files = list_dir(PrepareDataConfig.IN_TRAIN_DIR)
    logger.info(f"{PrepareDataConfig.IN_TRAIN_DIR.as_posix()}: {files}")

    files = list_dir(PrepareDataConfig.IN_TEST_DIR)
    logger.info(f"{PrepareDataConfig.IN_TEST_DIR.as_posix()}: {files}")


def prepare_dataset():
    logger.info(f"Start prepare_dataset")


def inspect_output():
    logger.info(f"Start inspect_output")
    files = list_dir(PrepareDataConfig.OUT_TRAIN_DIR)
    logger.info(f"{PrepareDataConfig.OUT_TRAIN_DIR.as_posix()}: {files}")

    files = list_dir(PrepareDataConfig.OUT_TEST_DIR)
    logger.info(f"{PrepareDataConfig.OUT_TEST_DIR.as_posix()}: {files}")


if __name__ == "__main__":
    sys.path.append("/opt/program")
    from utils import *

    inspect_input()
    prepare_dataset()
    inspect_output()
    sys.exit(0)
