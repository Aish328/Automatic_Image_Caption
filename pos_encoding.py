import torch
import random
import torch.nn as nn
import math
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
from Auto_img_Cap import remove_single_character_word
from torchvision import models
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_folder = '/home/aishanya/Desktop/Img_Caption/archive/Images'
dir_captions = '/home/aishanya/Desktop/Img_Caption/archive/captions.txt'

jpgs = os.listdir(image_folder)
df = pd.read_csv(dir_captions, sep = ',')

#making cleaned word dataset
df['cleaned'] = df['caption'].apply(lambda caption : ['<start>'] + [word.lower() if word.isalpha() else ' ' for word in caption.split(" ")] + ["<end>"])
df['cleaned'] = df['cleaned'].apply(lambda x : remove_single_character_word(x) )
df['len_of_sequence'] = df['cleaned'].apply(lambda x : len(x))
max_s= df['len_of_sequence'].max()

class PositionalEncoding(nn.Module):
    def __init__(self,d_model, dropout = 0.1 , max_len= max_s):
        super(PositionalEncoding , self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len , d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.pe.size(0) < x.size(0):
            self.pe = self.pe.repeat(x.size(0),1,1).to(device)
        self.pe = self.pe[:x.size(0),:,:]
        x = x+self.pe
        return self.dropout(x)