from torch.utils.data import Dataset
import os
from PIL import Image
from utils import renderedLaTeXLabelstr2Formula, crop_to_formula
import json

class renderedLaTeXDataset(Dataset):
    def __init__(self, image_folder, image_label_file, processor, device):
        self.image_folder = image_folder
        with open(image_label_file, 'r') as f:
            self.image_label = json.load(f)
        self.file_name = os.listdir(self.image_folder)
        self.device = device
        self.processor = processor
        self.num_image = len(self.image_label)
        
    def __len__(self):
        return self.num_image

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.file_name[idx])
        image = Image.open(img_name).convert('RGBA')
        image = crop_to_formula(image)
        inputs = self.processor(images = image, padding = "max_length", return_tensors="pt").to(self.device)
        for key in inputs:
            inputs[key] = inputs[key].squeeze() # Get rid of batch dimension since the dataloader will batch it for us.

        formula_idx = self.image_label[(self.file_name[idx]).split(".")[0]]
        # print(formula_idx)
        
        caption = renderedLaTeXLabelstr2Formula(formula_idx)
        # print(caption)
        # print(self.processor.tokenizer.tokenize(caption))
        caption = self.processor.tokenizer.encode(
            caption, return_tensors="pt", padding = "max_length", max_length = 512, truncation = True, # Tweak this
            ).to(self.device).squeeze()
        
        return inputs, caption