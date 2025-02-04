import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import seaborn as sns


filename='model.pkl'
with open(filename,'rb') as f:
    model=pickle.load(f)

st.title('tips prediction app')
st.subheader('enter the data')

df=sns.load_dataset('tips')
column_list=df[[ 'total_bill', 'time', 'size']]

uploaded_file=st.file_uploader('upload a csv file',type=['csv'])

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    object_coluumn=df.select_dtypes(include='object').columns.to_list()
    
le=LabelEncoder()
df['time']=le.fit_transform(df['time'])

prediction =model.predict(df)
prediction_text=np.where(prediction==1,'yes','no')
st.write(prediction_text)