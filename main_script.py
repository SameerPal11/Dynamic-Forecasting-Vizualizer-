import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
import json
import os
import chardet
import hashlib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
# from pmdarima import auto_arima
from QnA import *

# 📌 Set Streamlit to full-page width
st.set_page_config(layout="wide")


# 🔥 App Title
st.title("📊 AI-Powered Multi-File Data Analysis & Forecasting")

# 🗄 Define storage directories
STORAGE_DIR = "saved_forecasts"
MODEL_DIR = "saved_models"
CHAT_HISTORY_FILE = "chat_history/chat_history.json"  # File to store chat history
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# 📥 Sidebar for Multiple File Uploads
st.sidebar.header("📂 Upload Your Datasets")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

# Store DataFrames
dataframes = []
file_hashes = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_hash = get_file_hash(uploaded_file)
        file_hashes.append(file_hash)

        # 🔍 Detect file encoding

        raw_data = uploaded_file.read()
        encoding_type = chardet.detect(raw_data)["encoding"]
        uploaded_file.seek(0)

        # Load CSV into DataFrame
        df = pd.read_csv(uploaded_file, low_memory=False, encoding=encoding_type, encoding_errors="replace")
        df.dropna(inplace=True)
        dataframes.append(df)

        st.sidebar.write(f"✅ {uploaded_file.name} (Encoding: `{encoding_type}`)")
        st.write(f"### 📌 Dataset Preview ({uploaded_file.name})")
        st.dataframe(df.head())

# Combine all uploaded datasets

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    # st.write("### 🔄 Combined Dataset")
    # st.dataframe(combined_df.head())

    # 🔎 Detect Date Column
    date_col = None
    for col in combined_df.columns:
        try: 
            combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce')
            if combined_df[col].notna().sum() > 0:
                date_col = col
                break
        except Exception:
            continue

    if date_col is None:
        st.write("⚠ No valid date column found. Using index as time series.")
        combined_df.reset_index(inplace=True)
        date_col = 'index'
    else:
        st.write(f"📅 Detected Date Column: `{date_col}`")

    combined_df.dropna(subset=[date_col], inplace=True)
    combined_df = combined_df.sort_values(by=date_col)

    # 🎯 Select Target Column
    target_col = st.sidebar.selectbox("Select Target Column (to forecast):", combined_df.columns)

    if not np.issubdtype(combined_df[target_col].dtype, np.number):
        st.sidebar.write("🔢 Target column is categorical. Encoding it numerically.")
        combined_df[target_col], _ = pd.factorize(combined_df[target_col])

    # 📊 Initial Data Plot
    st.write("### 📈 Actual Data Visualization")
    fig, ax = plt.subplots(figsize=(12, 6),dpi=80)
    ax.plot(combined_df[date_col], combined_df[target_col], label="📊 Actual Data", marker="o", linestyle="-")
    ax.set_title("Actual Data Trend", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(target_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)

    # 🔮 Forecasting Preparation
    forecast_index = pd.date_range(pd.to_datetime(combined_df[date_col].iloc[-1]) + pd.Timedelta(days=1), periods=10, freq="D")
    
    # 🔍 Model Selection (ARIMA / RandomForest)
    try:
        model = ARIMA(combined_df[target_col], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)
        st.write("### 🔮 Forecasting ")
    except Exception:
        st.write("🔮 Forecasting.")
        try:
            combined_df["timestamp"] = (combined_df[date_col] - combined_df[date_col].min()).dt.days
            X = combined_df[["timestamp"]]
            y = combined_df[target_col]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            future_timestamps = (forecast_index - combined_df[date_col].min()).days.values.reshape(-1, 1)
            forecast = model.predict(future_timestamps)
            st.write("### 🌲 Forecasting with Random Forest")
        except Exception as e:
            st.write("❌ Both ARIMA and Random Forest failed.")
            st.write("Error:", str(e))
            forecast = None

    # 📉 Forecast Visualization
    if forecast is not None:
        st.write("### 📊 Forecast vs Actual Data")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(combined_df[date_col], combined_df[target_col], label="📊 Actual Data", marker="o", linestyle="-")
        ax.plot(forecast_index, forecast, label="🔮 Forecast", marker="x", linestyle="--", color="red")
        ax.set_title("Forecast vs Actual Data", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(target_col, fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

    # 🗨 Chatbot with AWS Bedrock (Claude Instant) + Chat History
    st.sidebar.header("🤖 Chat with Your Dataset")
    
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            chat_history = json.load(f)
    else:
        chat_history = []

    user_input = st.sidebar.text_input("Ask a question about your datasets:")

    def query_bedrock(user_input, combined_df):
        df_sample = combined_df[[date_col, target_col]].head(50).to_json()

        prompt = f"""
        You are an AI assistant analyzing multiple datasets:
        {df_sample}

        User's question: {user_input}

        Provide a structured response.
        """

        payload = {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",

            "max_tokens_to_sample": 500,
            "temperature": 0.7,
            "top_k": 250,
            "top_p": 0.9,
            "stop_sequences": ["\n\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31"
        }
        
        try:
            response = bedrock_client.invoke_model(
                body=json.dumps(payload),
                modelId="anthropic.claude-instant-v1",
                accept="application/json",
                contentType="application/json"
            )

            response_body = json.loads(response["body"].read())
            return response_body["completion"]

        except Exception as e:
            return f"❌ Error: {str(e)}"

    if user_input:
        if uploaded_files:
            response = query_bedrock(user_input, combined_df)
            chat_history.insert(0, {"question": user_input, "answer": response})  # Latest message at top
            with open(CHAT_HISTORY_FILE, "w") as f:
                json.dump(chat_history, f)

    # Display Chat History
    for chat in chat_history:
        st.sidebar.write(f"**🗨 {chat['question']}**")
        st.sidebar.write(f"🤖 {chat['answer']}")


