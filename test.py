import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.common import *
from model import VDSR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=int, default=2, help='-')

FLAGS, _ = parser.parse_known_args()

scale = FLAGS.scale

ckpt_path = f"checkpoint/VDSR.h5"

model = VDSR()
model.load_weights(ckpt_path)

ls_data = sorted_list(f"dataset/test/x{scale}/data")
ls_labels = sorted_list(f"dataset/test/x{scale}/labels")

sum_psnr = 0
for i in range(0, len(ls_data)):
    hr_image = read_image(ls_labels[i])
    lr_image = read_image(ls_data[i])
    bicubic_image = upscale(lr_image, scale)

    hr_image = rgb2ycbcr(hr_image)
    bicubic_image = rgb2ycbcr(bicubic_image)

    hr_image = norm01(hr_image)
    bicubic_image = norm01(bicubic_image)

    bicubic_image = tf.expand_dims(bicubic_image, axis=0)
    sr_image = model.predict(bicubic_image)[0]

    sum_psnr += PSNR(hr_image, sr_image, max_val=1)

print(sum_psnr / len(ls_data))

