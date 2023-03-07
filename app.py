import time
import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.write("""
# Analisis Sentimen Perkuliahan daring di Indonesia berdasarkan Tweet Twitter Menggunakan LSTM

# """)

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")