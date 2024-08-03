# main Python app
import streamlit as st
import streamlit.components.v1 as stc

#import our app
from eda_app import run_eda_app
from ml_app import run_ml_app

html_temp = """
<div style="background: linear-gradient(90deg, rgba(0,100,0,1) 0%, rgba(34,139,34,1) 100%); padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
    <h1 style="color: white; text-align: center; font-family: 'Arial', sans-serif; font-size: 2.5em; margin: 0;">Song Popularity Prediction</h1>
    <h4 style="color: white; text-align: center; font-family: 'Arial', sans-serif; font-size: 1.2em; margin-top: 10px;">Made by: <span style="font-weight: bold;">Dinda Pratiwi</span></h4>
</div>

            """

desc_temp = """
            ### Song Popularity Prediction 
            Want to know if your latest track has hit potential?
            This Streamlit application employs a Random Forest Model to predict song popularity. 
            Users can input audio features and visualize the model's prediction. 
            The app is built using Python and Streamlit, providing a robust and interactive experience.
            Have fun experimenting with different audio features and see how they impact your song's predicted popularity.
            #### Data Source
            - https://www.kaggle.com/datasets/yasserh/song-popularity-dataset/data
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning Section
            """


def main():
    # st.title("Main App")
    stc.html(html_temp)

    menu = ["Home","Exploratory Data Analysis", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Exploratory Data Analysis":
        run_eda_app()    
    elif choice == "Prediction":
        run_ml_app()

if __name__ == '__main__':
    main()
