'''
The main UI for the application

To run this application locally with GUI as web app,
run this command in cmd if the path of venv is already set correctly:
streamlit run UI\main.py
'''

# region libraries
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import streamlit as st
from PIL import Image
import cv2 as cv
import time
import glob
from PREPROCESSING.detection_models import text_detection_ocr, text_detection
from CLASSIFICATION_MODEL.processing import classify
# endregion




# app layout
st.set_page_config(page_title='Medical Prescription Text Detection App', layout='wide')
st.markdown("<h1 style='text-align: center; color: lightblue;'>Medical Prescription Text Detection</h1>", unsafe_allow_html=True)

# models
MODELS = {
    'Detector 1': text_detection_ocr,
    'Detector 2': text_detection
}

# initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

if 'radio_disabled' not in st.session_state:
    st.session_state.radio_disabled = False

# columns
col1, col2 = st.columns(2, gap='medium', border=True)


# region components
with col1:
    st.subheader(':blue[Input Settings]', divider='gray')
        
    # model selection
    selected_model = st.selectbox(
        ':blue[Select detection method]',
        list(MODELS.keys()),
        index=0
    )
    st.caption('NOTE: _Detector 1 usually works better._')
    
    # file uploader
    uploaded_file = st.file_uploader(':blue[Choose an image]', type=['jpg', 'jpeg', 'png'])

    # process button
    if st.button(':blue[**Detect Text Regions**]', width='stretch'):
        with st.spinner():
            if not uploaded_file:
                st.warning('Please choose an image!')

            image = Image.open(uploaded_file)

        # progress bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        
        # save as temp
        file_path = os.path.join('TEMP', uploaded_file.name)
        with open(file_path, 'wb') as file:
            file.write(uploaded_file.getbuffer())

        # main process for detection
        detection_func = classify(file_path, 'boosted_model.pkl', MODELS[selected_model])
        st.session_state.processed_image= Image.fromarray(cv.cvtColor(detection_func, cv.COLOR_BGR2RGB))
        progress_bar.empty()

with col2:
    st.subheader(':blue[Result]', divider='gray') 
    
    # show results
    if st.session_state.processed_image:
        st.image(
            st.session_state.processed_image
        )
        st.success('Detection Completed!')

        files = glob.glob('TEMP/*')
        for f in files:
            os.remove(f)


st.markdown("NOTE: _If a detection method didn't work properly or an error was raised, please switch to another one._")
# endregion


