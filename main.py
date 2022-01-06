import streamlit as st
from multiapp import MultiApp
import topmodel
import home
import topclassification

# Instiantiate the app
app = MultiApp()

# Adding all pages here
app.add_app("Home", home.home)
app.add_app("Topic Modelling", topmodel.app)
app.add_app("Topic Classification", topclassification.classify)


# Run the main app
app.run()
