# SPDX-FileCopyrightText: 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
# SPDX-License-Identifier: GPL-3.0
#
# SPDX-FileCopyrightText: 2019 Intel ISL (Intel Intelligent Systems Lab)
# SPDX-License-Identifier: MIT

# This code is adapted from https://github.com/isl-org/MiDaS

"""Compute depth maps for images in the input folder.
"""
import os
import sys
#TODO: add the path to MiDaS if not running the script from within the folder
sys.path.append(r'C:\Users\isxzl\OneDrive\Code\MiDaS')

import glob
import torch
import utils
import cv2
from tqdm import tqdm
from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet


def extract_depth(input_path, output_path, model_path, model_type="large", optimize=True, img_list="ALL"):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large": # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False
    
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()
    
    if optimize==True:
        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)  
            model = model.half()

    model.to(device)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    print(f'\n Working on folder {input_path}')
    # Get sub-folders

    img_names = glob.glob(os.path.join(input_path, "*"))
    if img_list!="ALL":
        img_names=[name for name in img_names if os.path.basename(name) in str(img_list)]

    num_images = len(img_names)
    print(num_images)
    for ind, img_name in tqdm(enumerate(img_names)):

        # input
        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize==True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # create output sub-folder
        os.makedirs(output_path, exist_ok=True)

        # output
        filename = os.path.join(
            output_path,
            os.path.splitext(os.path.basename(img_name))[0]
        )
        utils.write_depth(filename, prediction,grayscale=None, bits=2)

    print("finished")

