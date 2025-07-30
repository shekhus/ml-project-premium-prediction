# codebasics ML course: codebasics.io, all rights reserverd

import streamlit as st
from prediction_helper import predict

# Define the page layout
st.title('Health Insurance Cost Predictor')




# Button to make prediction
if st.button('Predict'):
    prediction = predict(input_dict)
    st.success(f'Predicted Health Insurance Cost: {prediction}')
