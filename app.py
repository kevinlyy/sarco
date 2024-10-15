from pycaret.classification import *
import streamlit as st
import pandas as pd
import numpy as np
st.image("logo.png",width=700,caption='Logo of the AITIS project', use_column_width=True)
st.write('''
# An online application for test-free screening and surveillance of sarcopenia
*Note*: This is a web app to predict the risk of sarcopenia in middle-aged and older adults. The prediction is based on age, sex, height, weight and function performance-related questions. Please answer each question in the sidebar to see the prediction.
''')
st.write('''
***Version 1.0.0 by Liangyu Yin, MD, PhD; Email: liangyuyin1988@qq.com or liangyuyin1988@tmmu.edu.cn***
        ''')
st.sidebar.header('Module 1: Batch Prediction')
st.sidebar.markdown("""
[Example CSV input file](https://github.com/kevinlyy/sarco/data_example.csv)
""")
# st.set_option('deprecation.showfileUploaderEncoding', False)
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
        st.sidebar.write('''***Do you have some difficulty to do the following tasks?***''')
        dressing = st.sidebar.selectbox('Dressing', ('yes','no'))
        bathing = st.sidebar.selectbox('Bathing', ('yes','no'))
        eating = st.sidebar.selectbox('Eating', ('yes','no'))
        bed = st.sidebar.selectbox('Bed (getting in and out of bed)', ('yes','no'))
        toilet = st.sidebar.selectbox('Toilet (using the toilet)', ('yes','no'))
        urination = st.sidebar.selectbox('Urination (controlling urination and defecation)', ('yes','no'))
        money = st.sidebar.selectbox('Money (managing money)',('yes','no'))
        medication = st.sidebar.selectbox('Medication (taking medications)',('yes','no'))
        shopping = st.sidebar.selectbox('Shopping (shopping for groceries)',('yes','no'))
        meal = st.sidebar.selectbox('Meal (preparing meals)',('yes','no'))
        housework = st.sidebar.selectbox('Housework (cleaning house)',('yes','no'))
        jogging_1km = st.sidebar.selectbox('Jogging 1km (running or jogging 1km)',('yes','no'))
        walking_1km = st.sidebar.selectbox('Walking 1km',('yes','no'))
        walking_100m = st.sidebar.selectbox('Walking 100m',('yes','no'))
        chair = st.sidebar.selectbox('Chair (getting up from a chair after sitting for long periods)',('yes','no'))
        climbing = st.sidebar.selectbox('Climbing (climbing several flights of stairs without resting)',('yes','no'))
        stooping = st.sidebar.selectbox('Stooping (stooping, kneeling, or crouching)',('yes','no'))
        lifting_5kg = st.sidebar.selectbox('Lifting 5kg (lifting or carrying weights over 5kg)',('yes','no'))
        picking = st.sidebar.selectbox('Picking (picking up a coin from the table)',('yes','no'))
        arms = st.sidebar.selectbox('Arms (reaching arms above shoulder level)',('yes','no'))
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
load_model = load_model('model')
result = predict_model(load_model,data=df, raw_score=True, probability_threshold=0.285)
prediction = result.iloc[0,]['prediction_label']
prediction_proba = result.iloc[0,][['prediction_score_0','prediction_score_1']]
st.subheader('Class labels and their corresponding index number')
target_names = {0:'Not sarcopenia',
                1:'Sarcopenia'}
target = pd.DataFrame(target_names, index=[0])
st.write(target)
st.subheader('Predicted probability of each class')
st.write('''*Note*: This model uses 0.285 as the optimal threshold to indicate the positive class.''')
st.write(prediction_proba)
st.subheader('Prediction')
target[prediction]
