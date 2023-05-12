from image2speech import pdf2text, text2audio, ocr_or_captioning, image_captioning, generate_output
import streamlit as st
# import gdown
import time
import os

# url_dict = {"https://drive.google.com/file/d/1vzDFTZRaz2N4Pg6Bghcz04Xf6b8l56zE/view?usp=sharing" : "./data/sample_file1.pdf", 
#             "https://drive.google.com/file/d/1iUxIgdHNRwbZYF_TSUsBi0Ekn7IG0zp_/view?usp=sharing" : "./data/sample_file2.pdf"}

# for u, o in url_dict.items():
#   gdown.download(u, o, quiet=False, fuzzy = True)

cwd = os.getcwd()
pdf1 = cwd + "/data/sample_file1.pdf"
pdf2 = cwd + "/data/sample_file2.pdf"
pdf3 = cwd + "/data/sample_file3.pdf"


pdf_examples = [pdf1, pdf2, pdf3]
st.markdown("<h1 style='text-align: center;'>PDF to Speech</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Converting Images and Text in PDFs to Audio using the BLIP Model and OCR</h5>", unsafe_allow_html=True)

st.write("<h6 style = 'text-align: center;'>Text Analytics Group 9</h6>", unsafe_allow_html=True) 
st.divider()

# Select PDF file
selected_pdf = st.selectbox("Choose an example file", options=pdf_examples)

#display empty outputs

text_heading = st.empty()
output_text = st.empty()
audio_heading = st.empty()
output_audio = st.empty()

 # placeholder running status
run_status = st.empty()

div_line = st.empty()

with st.spinner("Converting..."):
    if st.button('Convert'):
        start_time = time.time()
        # run_status.text("Running....")
        converted_text, converted_audio = generate_output(selected_pdf)
        text_heading.write("Converted Text")
        output_text.text(converted_text)
        audio_heading.write("Converted Audio")
        output_audio.audio(converted_audio)
        run_status.empty()
        end_time = time.time()
        # total time
        elapsed_time = round(end_time - start_time, 2)
        st.write(f"Elapsed time: {elapsed_time} seconds")




