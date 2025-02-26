import Auto_img_Cap
import os
import numpy as np
import pandas as pd
import pickle
import random
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import models, transforms
from img_caption_model import ImageCaptionModel
from flicker_dataset import FlickerDataResnet
from customdata import CustomData
# from Auto_img_Cap import remove_single_character_ward
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
image_folder = '/home/aishanya/Desktop/Img_Caption/archive/Images'
dir_captions = '/home/aishanya/Desktop/Img_Caption/archive/captions.txt'
vocab_path = '/home/aishanya/Desktop/Img_Caption/archive/vocab.txt'
features_path = '/home/aishanya/Desktop/Img_Caption/archive/image_features.pkl'
df = pd.read_csv(dir_captions, sep=',')
train = df.iloc[:int(0.9 * len(df))]
valid = df.iloc[int(0.9 * len(df)):]
unique_train_img = train[['image']].drop_duplicates()
unique_valid_img = valid[['image']].drop_duplicates()
# Load image filenames and captions
result = Auto_img_Cap.generate_caption(3, unique_valid_img.iloc[112]['image'])
print(result)