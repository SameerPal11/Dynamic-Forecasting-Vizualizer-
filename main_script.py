import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import chardet  
import hashlib
import os
import pickle  
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from QnA import *

# Page Configuration
st.set_page_config(page_title="Dynamic Forecasting Visualizer", layout="wide" )

# Custom CSS for BW Theme
st.markdown(
    """
    <style>
        body {
            background-color: #000;
            color: #FFF;
            font-family: Arial, sans-serif;
        }
        .stTextInput, .stSelectbox, .stButton button {
            background-color: #222;
            color: white;
            border-radius: 10px;
            border: 1px solid #555;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #FFF;
        }
        .stSidebar {
            background-color: #111;
        }
                div[data-baseweb="select"] > div {
            text-align: center !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("# 📊 Dynamic Forecasting Visualizer")
st.markdown("### Upload your dataset and explore future trends with AI models!\n---")

# Define storage directories
STORAGE_DIR = "saved_forecasts"
MODEL_DIR = "saved_models"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
# Dictionary to store multiple datasets

uploaded_file = st.sidebar.file_uploader("📂 Upload your dataset(CSV)", type=["csv"], accept_multiple_files=True)

if uploaded_file:
        file_names = [file.name for file in uploaded_file]
        selected_file_name = st.sidebar.selectbox("📜 Select a dataset to ask a question:", file_names)
        selected_file = next(file for file in uploaded_file if file.name == selected_file_name)

        file_name = selected_file.name  # Store filename
        selected_file.seek(0)  # Reset file pointer for reading

        # Read file content once and store it
        file_content = selected_file.read()  
        file_hash = get_file_hash(file_content)  # Pass raw bytes

        selected_file.seek(0)  # Reset again after reading

            
        cached_data, trained_model, date_col, target_values = load_cached_data(file_hash)
        
        if cached_data is not None and trained_model is not None:
            st.write("### Using Cached Data & Model (No Retraining)")
            df, forecast_index, forecast = cached_data
        
            target_col = st.sidebar.selectbox("Select the Target Column:", list(target_values.keys()))
            df[target_col] = target_values[target_col]
        else:
            raw_data = selected_file.read()
            result = chardet.detect(raw_data)
            encoding_type = result["encoding"]
            selected_file.seek(0)
            df = pd.read_csv(selected_file, encoding=encoding_type, encoding_errors="replace")
            df.dropna(inplace=True)


            st.write(f"Detected encoding: {encoding_type}")
            st.write("Dataset Preview:")
            st.write(df.head())


            for col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce') 
                    
                    if df[col].notna().sum() > 0:
                        date_col = col
                        break
                except Exception:
                    continue
          

            if date_col is None:
                df.reset_index(inplace=True)
                date_col = 'index'
            st.write(f"Detected Date Column: {date_col}")
            df.dropna(subset=[date_col], inplace=True)
            df.sort_values(by=date_col, inplace=True)
            # st.write(f"This is the df column ({date_col}):")
            st.write(df[col].head(10))  

            target_col = st.selectbox( "Select the Target Column (to forecast):", [col for col in df.columns if col != date_col]  )
            if not np.issubdtype(df[target_col].dtype, np.number):
                df[target_col], _ = pd.factorize(df[target_col])
            
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df.dropna(subset=[target_col], inplace=True)
            
            if df[target_col].nunique() <= 1:
                st.write("Target variable needs more variation. Choose Another!")
                st.stop()

            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df[date_col], df[target_col], label="Actual Data")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)
            
            forecast_index = pd.date_range(pd.to_datetime(df[date_col].iloc[-1]) + pd.Timedelta(days=1), periods=10, freq="D")
            
            df["timestamp"] = (df[date_col] - df[date_col].min()).dt.days
            X = df[["timestamp"]].values
            y = df[target_col].values

    # only for random forest regressor 


            future_timestamps = (forecast_index - df[date_col].min()).days.values.reshape(-1, 1)
            future_timestamps = future_timestamps.reshape((future_timestamps.shape[0], future_timestamps.shape[1], 1))
  
            try:
                    st.spinner("Training the model...")
                    model = ARIMA(y, order=(5, 1, 0))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=10)
                    trained_model = model_fit
                    st.write("### Forecasting ")

            except:
                    try:
                        st.spinner("Training the model...")
                        X_2d = X.reshape(X.shape[0], -1)

                        model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42)

                        model.fit(X_2d, y)
                        future_timestamps_2d = future_timestamps.reshape(future_timestamps.shape[0], -1)
                        forecast = model.predict(future_timestamps_2d)
                        
                        trained_model = model
                        st.write("### Forecasting")
                    except Exception as e:
                        st.write("All models failed. Error:", str(e))
                        forecast = None
                        trained_model = None
            
            if trained_model is not None:
                target_values = {col: df[col].tolist() for col in df.columns if col != date_col}
                save_trained_model(file_hash, trained_model)
                save_cached_data(file_hash, (df, forecast_index, forecast), date_col, target_values)

        if forecast is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df[date_col], df[target_col], label="Actual Data")
            ax.plot(forecast_index, forecast, label="Forecast", color="red")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(fig)
            
            dataset_text = df.head(10).to_json(index=False)

            st.sidebar.header("Chat with Your Dataset")
                    
            if "chat_history" not in st.session_state:
                if os.path.exists(CHAT_HISTORY_FILE):
                    with open(CHAT_HISTORY_FILE, "r") as f:
                      st.session_state.chat_history = json.load(f)
                else:
                    st.session_state.chat_history = []

            
            user_input = st.sidebar.text_input(
                f"Ask a question about {selected_file.name}:", key=f"user_question_{selected_file.name}"
            )           
          
            # if st.sidebar.button("Submit",key=f"submit_button_{selected_file.name}"):  # Process only when the button is clicked
            if st.sidebar.button("Submit", key=f"submit_button_{selected_file.name}"):
 
             if user_input and selected_file:
                        response = query_bedrock(user_input, dataset_text)
                        new_chat_entry = {"question": user_input, "answer": response}
                        st.session_state.chat_history.insert(0, new_chat_entry)  # Store in session
                        # Save to JSON file for persistence
                        with open(CHAT_HISTORY_FILE, "w") as f:
                            json.dump(st.session_state.chat_history, f)
            # Display chat history

            for chat in st.session_state.chat_history:
                st.sidebar.write(f"**{chat['question']}**")
                st.sidebar.write(f" {chat['answer']}")

