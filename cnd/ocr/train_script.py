import argparse
import os

from argus.callbacks import Checkpoint
from torch.utils.data import DataLoader
from cnd.ocr.dataset import OcrDataset
from cnd.ocr.argus_model import CRNNModel
from cnd.config import OCR_EXPERIMENTS_DIR, CONFIG_PATH, Config
from cnd.ocr.transforms import get_transforms
from cnd.ocr.metrics import StringAccuracy, LevenshteinDistance, JaroDistance
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--experiment_name", help="Save folder name", required=True)
args = parser.parse_args()

EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_DIR = OCR_EXPERIMENTS_DIR / EXPERIMENT_NAME

CV_CONFIG = Config(CONFIG_PATH)

DATASET_PATHS = [Path(CV_CONFIG.get("data_path"))]

BATCH_SIZE = 150
NUM_EPOCHS = 400

alphabet = "ABEKMHOPCTYX" + "".join([str(i) for i in range(10)]) + "-"

CRNN_PARAMS = {"image_height": CV_CONFIG.get("ocr_image_size")[0],
               "number_input_channels": CV_CONFIG.get("num_input_channels"),
               "number_class_symbols": len(alphabet),
               "rnn_size": CV_CONFIG.get("rnn_size")}

MODEL_PARAMS = {"nn_module": ("CRNN", CRNN_PARAMS),
                "alphabet": alphabet,
                "loss": {"reduction":"mean"},
                "optimizer": ("Adam", {"lr": 0.0001}),
                "device": "cpu",
                }


if __name__ == "__main__":
    if EXPERIMENT_DIR.exists():
        print(f"Folder 'EXPERIMENT_DIR' already exists")
    else:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    transforms = get_transforms(image_size=CV_CONFIG.get("ocr_image_size"))
    data_path = CV_CONFIG.get("data_path")

    all_pics = [data_path + '/' + name for name in os.listdir(data_path)]
    train, val = train_test_split(all_pics, test_size=0.2)
    train_dataset = OcrDataset(train, transforms)
    val_dataset = OcrDataset(val, transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    model = CRNNModel(MODEL_PARAMS)
    metrics = [StringAccuracy(),
               LevenshteinDistance(),
               JaroDistance()]
    callbacks = [Checkpoint(EXPERIMENT_DIR)]

    model.fit(
        train_loader,
        val_loader=val_loader,
        max_epochs=NUM_EPOCHS,
        metrics=metrics,
        metrics_on_train=True,
        callbacks=callbacks
    )
