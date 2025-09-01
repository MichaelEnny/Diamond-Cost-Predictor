# app.py
import streamlit as st
import pandas as pd
import os

from src.pipeline.prediction_pipeline import PredictPipeline, CustomData
from src.utils.utils import load_dataframe, load_object


data = load_dataframe("data", "raw.csv")
data.drop(['price'], axis=1, inplace=True)

st.write("""
# Diamond Price Predictor

Fill the diamond feature details in the sidebar and get its market price estimation !
""")
st.write('---')

# Sidebar Header:
st.sidebar.header('Specify Diamond Features')

def user_input_features(data):
    
    CARAT = st.sidebar.slider('CARAT', data.carat.min(), data.carat.max(), data.carat.mean())
    CUT = st.sidebar.selectbox('CUT', options=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    COLOR = st.sidebar.selectbox('COLOR', options=['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    CLARITY = st.sidebar.selectbox('CLARITY', options=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    DEPTH = st.sidebar.slider('DEPTH', data.depth.min(), data.depth.max(), data.depth.mean())
    TABLE = st.sidebar.slider('TABLE', data.table.min(), data.table.max(), data.table.mean())
    X = st.sidebar.slider('X', data.x.min(), data.x.max(), data.x.mean())
    Y = st.sidebar.slider('Y', data.y.min(), data.y.max(), data.y.mean())
    Z = st.sidebar.slider('Z', data.z.min(), data.z.max(), data.z.mean())


    data = {'carat': float(CARAT),
            'cut': str(CUT),
            'color': str(COLOR),
            'clarity': str(CLARITY),
            'depth': float(DEPTH),
            'table': float(TABLE),
            'x': float(X),
            'y': float(Y),
            'z': float(Z)
            }
    
    features = pd.DataFrame(data, index=[0])

    return features


df = user_input_features(data)

# Main Panel

# Print specified input parameters in a table:
st.header('Specified Diamond Features')
st.table(df)
st.write('---')

# Prediction using our Model:
predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(df)

# Displaying the prediction:
st.header('Prediction of Diamond Price')
with st.columns(3)[1]:
    st.subheader(prediction[0])
    # st.metric(label="Price", value=prediction[0])

st.write('---')
