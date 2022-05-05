from tensorflow.keras.losses import MeanSquaredError 
from tensorflow.keras.optimizers import Adam
from utils.dataset import dataset
from utils.common import PSNR
from model import VDSR
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",           type=int,   default=80,             help='-')
parser.add_argument("--batch-size",       type=int,   default=64,             help='-')
parser.add_argument("--save-best-only",   type=int,   default=0,              help='-')
parser.add_argument("--ckpt-dir",         type=str,   default="checkpoint",   help='-')


FLAG, unparsed = parser.parse_known_args()
epochs = FLAG.epochs
batch_size = FLAG.batch_size
ckpt_dir = FLAG.ckpt_dir
model_path = os.path.join(ckpt_dir, f"VDSR.h5")
save_best_only = (FLAG.save_best_only == 1)


# -----------------------------------------------------------
#  Init datasets
# -----------------------------------------------------------

dataset_dir = "dataset"
crop_size = 41 

train_set = dataset(dataset_dir, "train")
train_set.generate(crop_size, transform=True)
train_set.load_data()

valid_set = dataset(dataset_dir, "validation")
valid_set.generate(crop_size)
valid_set.load_data()


# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------

vdsr = VDSR()
vdsr.setup(optimizer=Adam(learning_rate=1e-3),
            loss=MeanSquaredError(),
            model_path=model_path,
            metric=PSNR)

vdsr.load_checkpoint(ckpt_dir)
vdsr.train(train_set, valid_set, 
            epochs=epochs, batch_size=batch_size,
            save_best_only=save_best_only)

