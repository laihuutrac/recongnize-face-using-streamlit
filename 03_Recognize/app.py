
from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter.filedialog import Open, SaveAs
from turtle import heading, width
import numpy as np
import cv2
import joblib

from sklearn.svm import LinearSVC
import streamlit as st 
from PIL import Image
from keras_preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model
import base64

detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2022mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)
detector.setInputSize((320, 320))

recognizer = cv2.FaceRecognizerSF.create(
            "face_recognition_sface_2021dec.onnx","")
svc = joblib.load('svc.pkl')
mydict = ['Ban Dac','Ban Duc','Ban Khac Huy','Ban Kiet','Ban Ky','Ban Nguyen','Ban Nhat Huy','Ban Ninh','Ban Quang','Ban Thanh','Ban Trac','Ban Truong','Thay Duc']

@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
img = get_img_as_base64("background.jpg")
page_bg_img = f"""
<style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{img}");
    background-size: 110%;
    background-position: top left;
    background-repeat: no-repeat;
    background-attachment: local;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
def Recognition(img_path):
        img=load_img(img_path,target_size=(320,320,3))
        imgin=img_to_array(img)
        faces = detector.detect(imgin)
        face_align = recognizer.alignCrop(imgin, faces[1][0])
        face_feature = recognizer.feature(face_align)
        test_prediction = svc.predict(face_feature)
        result = mydict[test_prediction[0]]
        return result

def run():
    st.markdown("<h1 style='text-align: Center; color: red;'>Nhận diện khuôn mặt</h1>", unsafe_allow_html=True)
    img_file = st.file_uploader("", type=["jpg", "png","bmp"])
    if img_file is not None:
        img = Image.open(img_file).resize((320,320))
        st.image(img,use_column_width=False)
        
        save_image_path = '../upload/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        if st.button("Recognition"):
            if img_file is not None:
                result= Recognition(save_image_path)
                # st.success("**Recognition : "+result+'**')
                txt ="Recognition :" + result
                htmlresult=f"""<p style='background-color:#00FF00;
                                           color:back;
                                           font-size:18px;
                                           border-radius:3px;
                                           line-height:60px;
                                           padding-left:17px;
                                           opacity:0.6'>
                                           {txt}</style>
                                           <br></p>""" 
                st.markdown(htmlresult,unsafe_allow_html=True) 
           
           
run()