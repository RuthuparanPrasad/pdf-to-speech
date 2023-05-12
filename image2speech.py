from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import streamlit as st
from io import BytesIO
from gtts import gTTS
from PIL import Image
import numpy as np
import pytesseract
import fitz
import cv2
import re
import os

# model details
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# function to generate image captions using BLIP model
def image_captioning(image, language = 'en'):
  text = "A picture of"
  inputs = processor(image, text, return_tensors="pt")
  out = model.generate(**inputs, max_new_tokens=30)
  final_text = processor.decode(out[0], skip_special_tokens=True)
  return final_text

# function to check if ocr is required
def ocr_or_captioning(img):
  # Converting images to grayscale
  gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
  # OCR recognition using Tesseract
  text = pytesseract.image_to_string(gray)
  # Determine if meaningful text information has been extracted
  if len(re.sub(r"\s+", "", text)) > 5:
    return text
  else:
    return image_captioning(img)
  
# function to convert text to audio
def text2audio(image_text, language = 'en'):
  audio_output = gTTS(text=image_text, lang=language, slow=False)
  audio_output.save('output.mp3')
  return 'output.mp3'

# function to read pdf and extract images/text
def pdf2text(pdf_file):

  pdf_doc = fitz.open(pdf_file)

  # Initialize the output filename
  output_file = 'output.txt'

  output_string = ''

  # Open the output file
  with open(output_file, 'w') as f:
      # Loop through the pages of the PDF file
      for page_num in range(pdf_doc.page_count):
          page = pdf_doc[page_num]
          # Get the images on the page together
          images = page.get_images()
          for i, img in enumerate(images):
              # Get the image data from the PDF page
              xref = img[0]
              pix = fitz.Pixmap(pdf_doc, xref)
              image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)          
              # check if the image needs to be captioned as text using BLIP or read as text using OCR
              image_text = ocr_or_captioning(image)
              f.write(f'Image {i+1} on page {page_num+1}:\n{image_text}\n')
          # Extract the text from the page
          text = page.get_text()
          if text:
              # Write the text to the output file
              f.write(text)

  # Close the PDF file
  pdf_doc.close() 
  return output_file

# main function that generates the final output
def generate_output(pdf_file):
  output_file = pdf2text(pdf_file)
  with open(output_file, 'r') as f:
    output1 = f.read()
  output2 = text2audio(output1)
  return output1, output2
