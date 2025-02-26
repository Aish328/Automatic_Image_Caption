import pandas as pd
import torch
class FlickerDataResnet():
    def __init__(self, data, pkl_file):
        self.data = data
        self.encodedImg = pd.read_pickle(pkl_file)

        # Debugging: Check keys
        none_count = sum(1 for v in self.encodedImg.values() if v is None)
        print(f"Total None values in encodedImg: {none_count}")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption_seq = self.data.iloc[idx]['text_seq']
        target_seq = caption_seq[1:] + [0]
        image_name = self.data.iloc[idx]['image']

        # Debugging: Check if image_name exists
        if image_name not in self.encodedImg:
            raise ValueError(f"❌ Error: No valid embedding found for image: {image_name}")

        image_tensor = self.encodedImg[image_name]

        # Debugging: Check if image_tensor is None
        if len(image_tensor.shape) == 3:  # If [C, H, W], permute to [H, W, C]
            image_tensor = image_tensor.permute(1, 2, 0)

        return torch.tensor(caption_seq), torch.tensor(target_seq), image_tensor