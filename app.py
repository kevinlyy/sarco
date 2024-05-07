import pandas as pd
import numpy as np
import streamlit as st
import pickle
st.image("logo.png",width=700,caption='Logo of the AITIS project', use_column_width=True) # use_column_width=True可以让图像变清晰
st.write('''
# An online application for test-free screening and surveillance of sarcopenia
*Note*: This app predicts sarcopenia risk (as defined by the Asian Working Group for Sarcopenia 2019 criteria) 
in middle-aged and older adults (using information on age, sex, weight and function performance related questions).
''')
st.write('''
***Version 1.0.0 by Liangyu Yin, MD, PhD; Email: liangyuyin1988@qq.com or liangyuyin1988@tmmu.edu.cn***
        ''')
st.sidebar.header('Module 1: Batch Prediction')
st.sidebar.markdown("""
[Example CSV input file](https://github.com/kevinlyy/sarco/data_example.csv)
""")
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.header('Module 2: Individual Prediction')
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        age = st.sidebar.slider('Age, years', 1, 100, 40)
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        height = st.sidebar.slider('Body height, m', 0.50, 2.50, 1.70)
        weight = st.sidebar.slider('Body weight, kg', 10, 150, 70)
        dressing = st.sidebar.selectbox('Dressing', ('yes','no'))
        bathing = st.sidebar.selectbox('Bathing', ('yes','no'))
        eating = st.sidebar.selectbox('Eating', ('yes','no'))
        bed = st.sidebar.selectbox('Bed', ('yes','no'))
        toilet = st.sidebar.selectbox('Toilet', ('yes','no'))
        urination = st.sidebar.selectbox('Urination', ('yes','no'))
        money = st.sidebar.selectbox('Money',('yes','no'))
        medication = st.sidebar.selectbox('Medication',('yes','no'))
        shopping = st.sidebar.selectbox('Shopping',('yes','no'))
        meal = st.sidebar.selectbox('Meal',('yes','no'))
        housework = st.sidebar.selectbox('Housework',('yes','no'))
        jogging_1km = st.sidebar.selectbox('Jogging 1km',('yes','no'))
        walking_1km = st.sidebar.selectbox('Walking 1km',('yes','no'))
        walking_100m = st.sidebar.selectbox('Walking 100m',('yes','no'))
        chair = st.sidebar.selectbox('Chair',('yes','no'))
        climbing = st.sidebar.selectbox('Climbing',('yes','no'))
        stooping = st.sidebar.selectbox('Stooping',('yes','no'))
        lifting_5kg = st.sidebar.selectbox('Lifting 5kg',('yes','no'))
        picking = st.sidebar.selectbox('Picking',('yes','no'))
        arms = st.sidebar.selectbox('Arms',('yes','no'))
        data = {'age':age,
                'sex':sex,
                'height':height,
                'weight':weight,
                'dressing':dressing,
                'bathing':bathing,
                'eating':eating,
                'bed':bed,
                'toilet':toilet,
                'urination':urination,
                'money':money,
                'medication':medication,
                'shopping':shopping,
                'meal':meal,
                'housework':housework,
                'jogging_1km':jogging_1km,
                'walking_1km':walking_1km,
                'walking_100m':walking_100m,
                'chair':chair,
                'climbing':climbing,
                'stooping':stooping,
                'lifting_5kg':lifting_5kg,
                'picking':picking,
                'arms':arms
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
local_data = pd.read_csv('data_example.csv')
df = pd.concat([input_df,local_data],axis=0)
encode = [  'sex',
            'dressing',
            'bathing',
            'eating',
            'bed',
            'toilet',
            'urination',
            'money',
            'medication',
            'shopping',
            'meal',
            'housework',
            'jogging_1km',
            'walking_1km',
            'walking_100m',
            'chair',
            'climbing',
            'stooping',
            'lifting_5kg',
            'picking',
            'arms']

for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]
columns_to_normalize = ['age', 'height', 'weight']
index = ['age', 'height', 'weight']
values = [57.954545, 1.582009, 58.987817]
means = pd.Series(values, index=index)
index = ['age', 'height', 'weight']
values = [9.529645, 0.094185, 11.662986]
stds = pd.Series(values, index=index)
df_zscore = df.copy()
df_zscore[columns_to_normalize] = (df[columns_to_normalize] - means) / stds
df = df_zscore[:1]
st.subheader('User Input Parameters')
if  uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using input parameters of Module 2 (as shown below)')
    st.write(df)
with open('model.pkl', 'rb') as file:
    load_model = pickle.load(file)
prediction = load_model.predict(df)
prediction_proba = load_model.predict_proba(df)
st.subheader('Class labels and their corresponding index number')
target_names = {0:'Not sarcopenia',
                1:'Sarcopenia'}
target = pd.DataFrame(target_names, index=[0])
st.write(target)
st.subheader('Prediction')
st.write(target[prediction])
st.subheader('Predicted probability of each class')
st.write('''This model uses 0.285 as the optimal threshold to indicate the positive class''')
st.write(prediction_proba)
