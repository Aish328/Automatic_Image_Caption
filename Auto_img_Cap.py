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
def generate_caption(K, img_nm):
    image = Image.open(os.path.join(image_folder, img_nm)).convert("RGB")
    plt.imshow(image)
    model.eval()
    img_embed = torch.tensor(valid_img_embed[img_nm]).unsqueeze(0).unsqueeze(0).to(device)
    input_seq = torch.tensor([word_to_index['<start>']] + [word_to_index['<pad>']] * (max_s - 1)).unsqueeze(0).to(device)
    predicted_sentence = []
    
    with torch.no_grad():
        for _ in range(max_s):
            output, _ = model(img_embed, input_seq)
            next_word = index_to_word[torch.topk(output[_, 0, :], K).indices.tolist()[0]]
            input_seq[:, _ + 1] = word_to_index[next_word]
            if next_word == '<end>':
                break
            predicted_sentence.append(next_word)
    
    print(f"Predicted caption: {' '.join(predicted_sentence)}.")
def remove_single_character_ward(word_list):
    return [word for word in word_list if len(word)>1]
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
image_folder = '/home/aishanya/Desktop/Img_Caption/archive/Images'
dir_captions = '/home/aishanya/Desktop/Img_Caption/archive/captions.txt'
vocab_path = '/home/aishanya/Desktop/Img_Caption/archive/vocab.txt'
features_path = '/home/aishanya/Desktop/Img_Caption/archive/image_features.pkl'

# Load image filenames and captions
df = pd.read_csv(dir_captions, sep=',')
df['cleaned'] = df['caption'].apply(lambda caption: ['<start>'] + 
    [word.lower() if word.isalpha() else ' ' for word in caption.split()] + ['<end>'])
df['cleaned'] = df['cleaned'].apply(lambda x : remove_single_character_ward(x) )
# df['cleaned'] = df['cleaned'].apply(lambda x: [word for word in x if len(word) > 1])

# Pad sequences
# max_s = df['cleaned'].apply(len).max()
df['len_of_sequence'] = df['cleaned'].apply(lambda x : len(x))
max_s= df['len_of_sequence'].max()
df.drop(['len_of_sequence'], axis = 1, inplace=True)
df['cleaned'] = df['cleaned'].apply(lambda caption: caption + ['<pad>'] * (max_s - len(caption)))

# # Build vocabulary
# word_dict = Counter(" ".join(df['cleaned'].apply(lambda x: " ".join(x))).split())
# word_dict = sorted(word_dict, key=word_dict.get, reverse=True)
list_word = df['cleaned'].apply(lambda x : " ".join(x)).str.cat(sep = ' ').split(' ')
word_dict = Counter(list_word)
word_dict = sorted(word_dict, key=word_dict.get , reverse= True)
with open(vocab_path, 'wb') as f:
    pickle.dump(word_dict, f)

with open(vocab_path, 'rb') as f:
    word_dict = pickle.load(f)
word_dict[:10]
vocab_size = len(word_dict)
word_to_index = {word: idx for idx, word in enumerate(word_dict)}
index_to_word = {idx: word for word, idx in word_to_index.items()}
df['text_seq'] = df['cleaned'].apply(lambda caption: [word_to_index[word] for word in caption])

# Split dataset
df = df.sort_values(by='image')
train = df.iloc[:int(0.9 * len(df))]
valid = df.iloc[int(0.9 * len(df)):]
unique_train_img = train[['image']].drop_duplicates()
unique_valid_img = valid[['image']].drop_duplicates()

# Create DataLoaders
train_ds = CustomData(unique_train_img)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
valid_ds = CustomData(unique_valid_img)
valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False)

# Load ResNet model
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract image features
features = {}
for jpg in os.listdir(image_folder):
    img_path = os.path.join(image_folder, jpg)
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        x = resnet(img).squeeze().detach().cpu().numpy()
    
    features[jpg] = x.flatten()

# Save features
with open(features_path, 'wb') as f:
    pickle.dump(features, f)

# Load training data
train_ds = FlickerDataResnet(train, features_path)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_ds = FlickerDataResnet(valid, features_path)
valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=True)

# Define model
model = ImageCaptionModel(16, 4, vocab_size, 512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2, verbose=True)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
min_val_loss = np.inf

# Training loop
EPOCHS = 50
for epoch in tqdm(range(EPOCHS)):
    model.train()
    total_epoch_train_loss, total_epoch_valid_loss = 0, 0
    total_train_words, total_valid_words = 0, 0
    
    for caption_seq, target_seq, image_embed in train_dl:
        optimizer.zero_grad()
        image_embed, caption_seq, target_seq = image_embed.squeeze(1).to(device), caption_seq.to(device), target_seq.to(device)
        output, padding_mask = model(image_embed, caption_seq)
        loss = torch.mul(criterion(output.permute(1, 2, 0), target_seq), padding_mask)
        final_batch_loss = torch.sum(loss) / torch.sum(padding_mask)
        final_batch_loss.backward()
        optimizer.step()
        total_epoch_train_loss += torch.sum(loss).detach().item()
        total_train_words += torch.sum(padding_mask)
    
    model.eval()
    with torch.no_grad():
        for caption_seq, target_seq, image_embed in valid_dl:
            image_embed, caption_seq, target_seq = image_embed.squeeze(1).to(device), caption_seq.to(device), target_seq.to(device)
            output, padding_mask = model(image_embed, caption_seq)
            loss = torch.mul(criterion(output.permute(1, 2, 0), target_seq), padding_mask)
            total_epoch_valid_loss += torch.sum(loss).detach().item()
            total_valid_words += torch.sum(padding_mask)
    
    total_epoch_train_loss /= total_train_words
    total_epoch_valid_loss /= total_valid_words
    print(f"Epoch {epoch}: Train Loss {total_epoch_train_loss:.4f}, Valid Loss {total_epoch_valid_loss:.4f}")
    
    if min_val_loss > total_epoch_valid_loss:
        print(f"Saving model at epoch {epoch}")
        torch.save(model, './BestModel.pth')
        min_val_loss = total_epoch_valid_loss
    
    scheduler.step(total_epoch_valid_loss)
model = torch.load('/home/aishanya/Desktop/Img_Caption/archive/model.h5')
start_token = word_to_index['<start>']
end_token = word_to_index['<end>']
pad_token = word_to_index['<pad>']
valid_img_embed = pd.read_pickle('/home/aishanya/Desktop/Img_Caption/archive/image_features.pkl')

max_seq_len = 33
# print(start_token, end_token , pad_token)
# Caption generation

generate_caption(3, unique_valid_img.iloc[112]['image'])