# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:28:22 2024

@author: Yonatan
"""

from pathlib import Path

import typer
from mimic_dev_main import make_submission


def main(inference_path: Path):
    inference_path.mkdir(exist_ok=True)
    (inference_path / "data").mkdir(exist_ok=True)
    (inference_path / "assets").mkdir(exist_ok=True)
    make_submission(submission_path=inference_path, no_check=True)


if __name__ == "__main__":
    typer.run(main)
