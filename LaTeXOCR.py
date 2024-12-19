import torch
from PIL import Image
import re
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


import pandas as pd
import os, sys

# from PIL import Image
# from torchvision import transforms
# from sklearn.model_selection import train_test_split
# import sklearn as skl
# import torch as t

from ultralytics import YOLO
import matplotlib.pyplot as plt

import gradio as gr
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''YOLO'''
localize_model = YOLO("30_best.pt")
def localize_image(image):
	with torch.no_grad():
		predictions = localize_model.predict(
			source=image,
			conf=0.1,
			iou=0.1,
			stream=True
		)

	for prediction in predictions:
		if len(prediction.boxes.xyxy):
			boxes = prediction.boxes.xyxy.cpu().numpy()
			scores = prediction.boxes.conf.cpu().numpy()

			# print(boxes)
			# print(scores)
			# print(bbox)
			idx = np.argsort(scores)
			x1, y1, x2, y2 = boxes[idx[-1]]

			print(boxes)
			result = image.crop([x1, y1, x2, y2])

			# plt.imshow(result)
			# plt.axis('off')  # Hide the axis
			# plt.show()

			return result

	return image

'''TrOCR'''
# path = "/kaggle/input/latex-ocr-implementation/models/trocr-large-finetuned-math-captions"

path = './models/latexocr-finetuned'
# path = 'microsoft/trocr-large-handwritten'
processor = TrOCRProcessor.from_pretrained(path)
model = VisionEncoderDecoderModel.from_pretrained(path).to(device)
def predict_ocr(image):
	with torch.no_grad():
		pixel_values = processor(images = image, return_tensors="pt").pixel_values.to(device)
		out = model.generate(pixel_values, max_new_tokens = 256)
		pred = processor.decode(out[0], skip_special_tokens=True).replace("\\ ", "\\")
		symbols = ['lim', 'sin', 'cos', 'tan']
		for sym in symbols:
			pred = pred.replace(sym, f"\\{sym}")
		print(pred)
	return f'${pred}$'


'''FULL PIPELINE'''
def predict_latex(image):
	print('localizing image')
	cropped = localize_image(image)
	print('done localizing')
	print('predicting')
	prediction = predict_ocr(cropped)
	print('done prediction')

	return prediction

# Create Gradio interface
with gr.Blocks() as demo:
	gr.Markdown("# Image to LaTeX Converter")
	gr.Markdown("Upload an image containing a mathematical expression, and this tool will convert it to LaTeX.")

	with gr.Row():
		with gr.Column():
			image_input = gr.Image(type="pil", label="Upload Image")
		with gr.Column():
			latex_output = gr.Markdown(height=100, latex_delimiters=[ {"left": "$", "right": "$", "display": True }])

	convert_button = gr.Button("Convert")

	convert_button.click(
		predict_latex,
		inputs=image_input,
		outputs=latex_output
	)

demo.launch(share=True, debug=True)