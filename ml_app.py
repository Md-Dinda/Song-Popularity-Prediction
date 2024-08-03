import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Memuat encoder dan scaler
encoder_path = 'encoded_df.pkl'
scaler_path = 'scaler.pkl'
model_path = 'rf.pkl.bz2'

if os.path.exists(encoder_path) and os.path.exists(scaler_path) and os.path.exists(model_path):
    df_encoded = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
else:
    st.error("File encoder, scaler, atau model tidak ditemukan.")
    st.stop()

def run_ml_app():
    st.subheader("Welcome to Prediction Section")
    st.subheader("Input your Song Data to Predict Song Popularity")
    song_duration_ms = st.number_input('Song Durations (ms)', 0, 500000)
    acousticness = st.number_input('acousticness', format="%.6f")
    danceability = st.number_input('danceability', format="%.3f")
    energy = st.number_input('energy', format="%.3f")
    instrumentalness = st.number_input('instrumentalness', format="%.6f")
    key = st.selectbox('Key', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    liveness = st.number_input('liveness', format="%.4f")
    loudness = st.number_input('loudness', format="%.3f")
    audio_mode = st.selectbox('audio mode', [0, 1])
    speechiness = st.number_input('speechiness', format="%.4f")
    tempo = st.number_input('tempo', format="%.3f")
    time_signature = st.selectbox('time signature', [1, 2, 3, 4])
    audio_valence = st.number_input('audio_valence', format="%.3f") 

    with st.expander("Your Selected Options"):
        result = {
            'song_duration_ms': song_duration_ms,
            'acousticness': acousticness,
            'danceability': danceability,
            'energy': energy,
            'instrumentalness': instrumentalness,
            'key': key,
            'liveness': liveness,
            'loudness': loudness,
            'audio_mode': audio_mode,
            'speechiness': speechiness,
            'tempo': tempo,
            'time_signature': time_signature,
            'audio_valence': audio_valence,
        }
        st.write(result)

    # Convert input to DataFrame for encoding
    input_df = pd.DataFrame([result])

    # Perform the same encoding as training
    input_encoded = pd.get_dummies(input_df, columns=['audio_mode', 'key', 'time_signature'], drop_first=False)
    
    # Align the input with the encoder columns
    input_encoded = input_encoded.reindex(columns=df_encoded.columns, fill_value=0)

    # Remove 'song_popularity' if it exists
    if 'song_popularity' in input_encoded.columns:
        input_encoded = input_encoded.drop(columns=['song_popularity'])

    # Scale the encoded input data
    input_scaled = scaler.transform(input_encoded)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display prediction result
    with st.expander("Prediction Result"):
        def display_prediction_result(prediction):
            predicted_popularity = round(prediction[0])
            st.subheader('Drumroll, please...')
            st.balloons()
            
            st.subheader('Prediction result:')
            st.write(predicted_popularity)
            
            if predicted_popularity >= 80:
                st.success(f"Your song has a **fantastic** chance of being a hit! With a predicted popularity score of {predicted_popularity}, you're on the right track!")
            elif predicted_popularity >= 60: 
                st.info(f"Your song has a **good** chance of being popular. With a predicted popularity score of {predicted_popularity}, there's potential for success!")
            elif predicted_popularity >= 40:
                st.warning(f"Your song has a **moderate** chance of being popular. With a predicted popularity score of {predicted_popularity}, consider refining your song or targeting a specific audience.")
            else:
                st.error(f"Unfortunately, your song has a **low** predicted popularity score of {predicted_popularity}. Don't give up! Keep experimenting and improving your music.")

        display_prediction_result(prediction)
