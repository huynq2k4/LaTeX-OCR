import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import renderedLaTeXDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = 'microsoft/trocr-large-handwritten'
processor = TrOCRProcessor.from_pretrained(path)
model = VisionEncoderDecoderModel.from_pretrained(path).to(device)

NUM_EPOCHS = 5
LEARNING_RATE = 1e-5
BATCH_SIZE = 2
SHUFFLE_DATASET = True

set_seed(0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

train_ds = renderedLaTeXDataset(image_folder='latex-ocr/training-data',
                               image_label_file='latex-ocr/training-data-label.json',
                                processor=processor,
                               device=device)

train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = SHUFFLE_DATASET, num_workers = 0)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.to(device)
model.train()

history = []; val_history = []; val_timesteps = []
ema_loss = None; ema_alpha = 0.95
scaler = torch.amp.GradScaler(enabled = True)
for epoch in range(NUM_EPOCHS):
    with tqdm(train_dl, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}") as pbar:
        for batch, captions in pbar:
            pixel_values = batch["pixel_values"]
            
            optimizer.zero_grad()
            with torch.autocast(device_type = "cuda", dtype = torch.float16, enabled = True):
                outputs = model(pixel_values = pixel_values,
                                labels = captions)
                loss = outputs.loss
                history.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema_loss is None: ema_loss = loss.item()
            else: ema_loss = ema_loss * ema_alpha + loss.item() * (1 - ema_alpha)
            pbar.set_postfix(loss=ema_loss)

model.save_pretrained("models/latexocr-finetuned")
processor.save_pretrained("models/latexocr-finetuned")