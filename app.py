import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def carrega_modelo():

    url = 'https://drive.google.com/uc?id=1W-IivGWd_xNppno26cVmGQvDb9yR-bAG'


    gdown.download(url, "modelo_quantizado16bits.tflite")
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    

def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte sua imagem:', type = ['png', 'jpg','jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success("Imagem carregada com sucesso!")

        image = np.array(image, dtype=np.float32)
        image = image/ 255.0
        image = np.expand_dims(image, axis=0)


        return image



def previsao(interpreter, image):
    input_details = interpreters.get_input_details()
    output_details = interpreters.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['BlackMeasles','BlackRot','Healthy','LeafBlight']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]

    fig = px.bar(df, x = 'Probabilidades (%)',
                 title = 'Probabilidade de classes de Doenças de Folhas de Videira')
    st.plotly_chart(fig) 




def main():

    st.set_page_config(
        page_title="Classificação de imagens de folhas de videira",
        page_icon="🍇"
    )

    st.write("# Classificação de folhas de Videiras! 🍇")


    #Carrega o modelo
    interpreter = carrega_modelo()


    #Carrega a imagem
    image = carrega_imagem()


    #Classifica a imagem
    if image:
        previsao(interpreter, image)

if __name__ == '__main__':
    main()