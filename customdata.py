import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
from collections import Counter
from tensorflow import keras
import sys, os, time, warnings
import pandas as pd
import pickle
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from torch.utils.data import DataLoader ,Dataset
import torch.nn as nn
from PIL import Image
class CustomData():
    def __init__(self , data):
      self.data = data
      self.scaler = transforms.Resize([224,224])
      self.normalize = transforms.Normalize(mean = [0.485 , 0.456 , 0.406],std = [0.229 , 0.224,0.225]) # Average pixel intensity in RGB channels.,Standard deviation of pixel intensities.
      self.to_tensor= transforms.ToTensor()#cale data to a standard range (like -1 to 1)

    def __len__(self):
      return len(self.data)
    def __getitem__(self,idx):
       image_name = self.data.iloc[idx]['image']
       img_loc = '/home/aishanya/Desktop/Img_Caption/archive/Images/'+str(image_name)
       img = Image.open(img_loc)
       tensor_img = self.normalize(self.to_tensor(self.scaler(img)))
       return image_name , tensor_img