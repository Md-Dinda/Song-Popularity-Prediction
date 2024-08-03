import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Fungsi untuk memuat data
@st.cache
def load_data(data):
    df = pd.read_csv(data)
    df = df.iloc[:, 1:]
    return df

# Fungsi untuk menjalankan aplikasi EDA
def run_eda_app():
    st.subheader("Exploratory Data Analysis")
    df = load_data("song_data.csv")

    # Menu di sidebar
    submenu = st.sidebar.selectbox("SubMenu", ["Description", "Plots"])
    
    if submenu == "Description":
        st.dataframe(df)

        with st.expander("Dataset Summary"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
        
        with st.expander("Descriptive Summary"):
            st.dataframe(df.describe())

    elif submenu == "Plots":
        st.subheader("Plots")

        with st.expander("Correlation Plot"):
            columns = df.select_dtypes(include=['int64','float64']).columns.to_list()
            corr_matrix = df[columns].corr()
            fig = plt.figure(figsize=(20,10))
            sns.heatmap(corr_matrix, annot=True, cmap="crest", linewidth=.5, annot_kws={"size":15})
            st.pyplot(fig)
            
        with st.expander("Histogram"):
            column = st.selectbox("Select Column for Histogram", columns)
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)

        with st.expander("Box Plot"):
            column = st.selectbox("Select Column for Box Plot", columns, key='box_plot')
            fig, ax = plt.subplots()
            sns.boxplot(df[column], ax=ax)
            st.pyplot(fig)

        with st.expander("Scatter Plot"):
            x_axis = st.selectbox("Select X-axis", columns)
            y_axis = st.selectbox("Select Y-axis", columns)
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
            st.pyplot(fig)

    else:
        st.write("Option not available.")
