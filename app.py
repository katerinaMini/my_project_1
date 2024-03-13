import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
import pickle


st.set_page_config(
    page_title="Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

uploaded_file = None

if 'df_input' not in st.session_state:
    st.session_state['df_input'] = pd.DataFrame()

model_file_path = 'models\lr_model_churn_prediction.sav'
model = pickle.load(open(model_file_path, 'rb'))

encoding_model_file_path = 'models\encoding_model2.sav'
encoding_model = pickle.load(open(encoding_model_file_path, 'rb'))


def predict(df_input):
    df_original = df_input.copy()
    scaler = MinMaxScaler()
    dicts = df_input.to_dict(orient='records')
    X = encoding_model.transform(dicts)
    X_encoded = scaler.fit_transform(X)
    y_pred = model.predict(X_encoded)
    df_original['predicted'] = np.expm1(y_pred)
    return df_original

def convert_df(df):
    # Функция для конвертации датафрейма в csv
    return df.to_csv(index=False).encode('utf-8')

st.image('https://www.ibik.ru/img/glav_/solushion/health.png', width=400)
st.title('Сумма затрат человека на медицинские расходы')

with st.sidebar:
    st.title('Загрузите файл')
    uploaded_file = st.file_uploader("Выбрать CSV файл", type=['csv'])
    if uploaded_file is not None:
        try:
            st.session_state['df_input'] = pd.read_csv(uploaded_file)
            st.success("Файл успешно загружен.")
        except Exception as e:
          st.error(f"Ошибка при чтении файла CSV: {str(e)}")

    prediction_button = st.button('Предсказать', type='primary', use_container_width=True, key='button1')

    if prediction_button and 'df_input' in st.session_state:
        st.session_state['df_predicted'] = predict(st.session_state['df_input'])

if 'df_predicted' in st.session_state and len(st.session_state['df_predicted']) > 0:
    st.subheader('Результаты прогнозирования')
    st.write(st.session_state['df_predicted'])
    res_all_csv = convert_df(st.session_state['df_predicted'])
    st.download_button(
        label="Скачать предсказания",
        data=res_all_csv,
        file_name='df-churn-predicted-all.csv',
        mime='text/csv',
    )