import streamlit as st
from PIL import Image


def home():
    st.title("iBert")
    st.write("####  Automating Topic Discovery and Classification!")
    st.write("##### Created by Akshara Shukla")
    st.write('''Welcome, to the home page! Here, you can select the task you want to perform by choosing from the list on the left.''')

    img = Image.open('ibert.PNG')
    st.image(img)
    st.markdown('##')
    st.markdown('##')
    st.markdown('##')
    st.markdown('##')
    st.write('### Source Code and github link')
    st.write('You can access the portfolio which describes the reasoning for building this application on this [link](https://www.notion.so/bigintelintern/Reading-Guide-731e616c9cf14fba8ffdb32652b11ef0).')
    st.write('For accessing the source code for this application, you can click on this [link]()')



