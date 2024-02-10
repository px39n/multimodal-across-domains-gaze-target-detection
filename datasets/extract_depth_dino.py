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
sys.path.append(r'..')
import glob
import torch
import utils
import cv2
from tqdm import tqdm
from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from transformers import AutoImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
import matplotlib
from torchvision import transforms
from PIL import Image    

def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])

def render_depth(values, colormap_name="magma_r"):
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return colors #Image.fromarray(colors)
def extract_depth_dino(input_path, output_path, model_path, model_type="large", optimize=True, img_list="ALL"):

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
    image_processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-large-nyu")
    model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-large-nyu")


    # forward pass
    transform = make_depth_transform()

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
        img = Image.open(img_name)
        
        # compute
        # compute
        rescaled_image = img.resize((img.width, img.height))
        transformed_image = transform(rescaled_image)
        inputs = image_processor(images=transformed_image, return_tensors="pt")
        
        # Prediction
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = -outputs.predicted_depth.squeeze().cpu().numpy()
        prediction=np.array(Image.fromarray(prediction_dino).resize((img.width, img.height)))
        #display(depth_image)
    
        
        # create output sub-folder
        os.makedirs(output_path, exist_ok=True)

        # output
        filename = os.path.join(
            output_path,
            os.path.splitext(os.path.basename(img_name))[0]
        )
        utils.write_depth(filename, prediction,grayscale=None, bits=2)

    print("finished")
