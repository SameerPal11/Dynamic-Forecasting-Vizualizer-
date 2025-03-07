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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Page Configuration
st.set_page_config(page_title="Dynamic Forecasting Visualizer", layout="wide")
st.markdown("# 📊 Dynamic Forecasting Visualizer")
st.markdown("### Upload your dataset and explore future trends with AI models!\n---")

# Define storage directories
STORAGE_DIR = "saved_forecasts"
MODEL_DIR = "saved_models"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def get_file_hash(uploaded_file):
    file_content = uploaded_file.getvalue()
    return hashlib.md5(file_content).hexdigest()

def load_cached_data(file_hash):
    cache_path = os.path.join(STORAGE_DIR, f"{file_hash}.pkl")
    model_path = os.path.join(MODEL_DIR, f"{file_hash}_model.pkl")
    
    if os.path.exists(cache_path) and os.path.exists(model_path):
        with open(cache_path, "rb") as f:
            cached_data, date_col, target_values = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return cached_data, model, date_col, target_values  
    return None, None, None, None

def save_cached_data(file_hash, data, date_col, target_values):
    cache_path = os.path.join(STORAGE_DIR, f"{file_hash}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump((data, date_col, target_values), f)

def save_trained_model(file_hash, model):
    model_path = os.path.join(MODEL_DIR, f"{file_hash}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

def train_lstm_model(X, y):
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X, y, epochs=50, verbose=0)
    return model

uploaded_file = st.sidebar.file_uploader("📂 Upload your dataset(CSV)", type=["csv"])

if uploaded_file is not None:
    file_hash = get_file_hash(uploaded_file)
    cached_data, trained_model, date_col, target_values = load_cached_data(file_hash)
    
    if cached_data is not None and trained_model is not None:
        st.write("### Using Cached Data & Model (No Retraining)")
        df, forecast_index, forecast = cached_data
        target_col = st.sidebar.selectbox("Select the Target Column:", list(target_values.keys()))
        df[target_col] = target_values[target_col]
    else:
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding_type = result["encoding"]
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding=encoding_type, encoding_errors="replace")
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
        # try:
            # X_train, y_train = X[:-10], y[:-10]
            # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        
            # model = train_lstm_model(X_train, y_train)
            # future_timestamps = (forecast_index - df[date_col].min()).days.values.reshape(-1, 1)
            # future_timestamps = future_timestamps.reshape((future_timestamps.shape[0], future_timestamps.shape[1], 1))
            # forecast = model.predict(future_timestamps).flatten()
            # trained_model = model
            # st.write("### Forecasting with LSTM")
        # except Exception:

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

                    model = RandomForestRegressor(n_estimators=100, random_state=42)
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
