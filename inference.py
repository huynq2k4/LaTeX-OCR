import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import argparse
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument("--img_url", type=str, required=True, help="Path to the image file.")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = 'models/latexocr-finetuned'
processor = TrOCRProcessor.from_pretrained(path)
model = VisionEncoderDecoderModel.from_pretrained(path).to(device)

img_url = args['img_url']
image = Image.open(img_url).convert('RGB')

pixel_values = processor(images = image, return_tensors="pt").pixel_values.to(device)
out = model.generate(pixel_values, max_new_tokens = 256)
pred = processor.decode(out[0], skip_special_tokens=True).replace("\\ ", "\\")

print("Prediction:", pred)