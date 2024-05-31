import streamlit as st
from PIL import Image
# from io import BytesIO
import cv2
import numpy as np
import onnxruntime as ort

path_yolo = "weights/best_ckpt.onnx"
path_cls = "weights/id_crop_cls_v2_regnetY.onnx"

sess_yolo = ort.InferenceSession(path_yolo)
input_name_yolo = sess_yolo.get_inputs()[0].name
input_shape_yolo = sess_yolo.get_inputs()[0].shape

sess_cls = ort.InferenceSession(path_cls)
input_name_cls = sess_cls.get_inputs()[0].name
input_shape_cls = sess_cls.get_inputs()[0].shape

def tranform_img_yolo(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
    return img

def detect_id(img):
    input_tensor = img.reshape(input_shape_yolo)
    output = sess_yolo.run(None, {input_name_yolo: input_tensor})
    output = np.array(output[0][0])
    return output

def transform_img_cls(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    return img

def cls_id(img):
    input_tensor = img.reshape(input_shape_cls)
    output = sess_cls.run(None, {input_name_cls: input_tensor})
    output = np.array(output[0][0])
    return output

def softmax(x):
    exp_x = np.exp(x)
    exp_x = exp_x / exp_x.sum(axis=0)
    exp_x = np.round(exp_x*100, 2)
    return exp_x

def process(img_raw):
    h_raw, w_raw = img_raw.shape[:2]
    h_yolo, w_yolo = 640, 640

    img_yolo_tf = tranform_img_yolo(img_raw)
    yolo_result = detect_id(img_yolo_tf)

    max_index = np.argmax(yolo_result[:, 5])
    box = yolo_result[max_index, :4]

    pt1 = (box[0]-box[2]/2)/w_yolo*w_raw, (box[1]-box[3]/2)/h_yolo*h_raw
    pt2 = (box[0]+box[2]/2)/w_yolo*w_raw, (box[1]+box[3]/2)/h_yolo*h_raw
    pt1, pt2 = np.int16(pt1), np.int16(pt2)
    img_crop = img_raw[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]

    img_cls = transform_img_cls(img_crop)
    cls_result = cls_id(img_cls)
    cls_result_sm = softmax(cls_result)
    if cls_result_sm[0]>cls_result_sm[1]:
        return([img_crop, 'live :thumbsup:', str(cls_result_sm[0])+'%'])
    else:
        return([img_crop, 'spoof :thumbsdown:', str(cls_result_sm[1])+'%'])


st.set_page_config(layout="wide", page_title="ID_CCCD_Spoof_Cls")

st.write("## ID Citizen Identification Card Spoofing-Classifier")
st.write("Demo version")
st.sidebar.write("## Upload an image of ID Card :gear:")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# # Download the fixed image
# def convert_image(img):
#     buf = BytesIO()
#     img.save(buf, format="JPG")
#     byte_im = buf.getvalue()
#     return byte_im


def fix_image(upload):
    image = Image.open(upload).convert('RGB')
    image = np.array(image)[:,:,::-1]
    col1.write("#### Original Image :camera:")
    col1.image(image, channels='BGR')

    img_crop, res, conf = process(image)
    col2.write("#### Classification result :clipboard:")
    col2.write("Crop-ID Image :scissors:")
    col2.image(img_crop, channels='BGR')
    col2.write('Label: '+res)
    col2.write('Confident: '+conf)
    # st.sidebar.markdown("\n")
    # st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.jpg", "image/jpg")

col1, col2 = st.columns(2, gap='large')
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 10MB.")
    else:
        fix_image(upload=my_upload)