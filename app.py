import streamlit as st
import altair as altc
import pandas as pd
import numpy as np
import os, urllib, cv2
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,UpSampling2D,Conv2DTranspose

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate, add
from skimage.transform import resize
import plotly.express as px
from keras import backend as K

from u_net import build_unet


st.cache(allow_output_mutation=True)
def load_model():


    H = 768   #to keep the original ratio
    W = 1152
    num_classes = 23

    model = build_unet((H,W, 3), num_classes)
    model.load_weights('unet(w)_44.h5')
    return model


def main():
    class_dict = pd.read_csv('class_dict_seg.csv')
    cla= []
    for iteration in range(24):
        cla.append(list(class_dict.iloc[iteration][1:]))

    model = load_model()

    st.sidebar.header("Upload an Image File")
    uploaded_file = st.sidebar.file_uploader("")
    if uploaded_file is not None:


        st.sidebar.success('Image uploaded successfully')
        image = Image.open(uploaded_file)

        image= image.resize((1152,768))

        arr = np.array(image)


        x = arr.reshape((1,768,1152,3))

        plt.imshow(image)
        plt.axis("off")
        plt.show()
        plt.savefig('WC.jpg')
        img= Image.open("WC.jpg")
        st.header("Input Image")
        st.image(img)


        x = x/255
        p = model.predict(x)[0]
        p = np.argmax(p, axis=-1)
        l = np.zeros(shape = (768, 1152,3),dtype = np.uint8)
        v = []
        for i in range(768) :
            for j in range(1152) :
                k = p[i,j]
                v.append(cla[k])
                l[i,j,:] = cla[k]

        plt.imshow(l)
        plt.savefig('se.jpg')
        img= Image.open("se.jpg")
        st.header("Segmented Image")
        st.image(img)
        # fig = px.imshow(img)
        # fig.show()




if __name__=='__main__':
    main()
