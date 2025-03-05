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

st.title("Dynamic Forecasting Visualizer")

STORAGE_DIR = "saved_forecasts"
MODEL_DIR = "saved_models"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def get_file_hash(uploaded_file):
    """Generate a unique hash for the uploaded file."""
    file_content = uploaded_file.getvalue()
    return hashlib.md5(file_content).hexdigest()

def load_cached_data(file_hash):
    cache_path = os.path.join(STORAGE_DIR, f"{file_hash}.pkl")
    model_path = os.path.join(MODEL_DIR, f"{file_hash}_model.pkl")
    
    cached_data = None
    model = None
    date_col = None  
    target_col = None  

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached_data, date_col, target_col = pickle.load(f)  

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    
    return cached_data, model, date_col, target_col  

def save_cached_data(file_hash, data, date_col, target_col):
    cache_path = os.path.join(STORAGE_DIR, f"{file_hash}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump((data, date_col, target_col), f)

def save_trained_model(file_hash, model):
    """Save the trained model to local storage."""
    model_path = os.path.join(MODEL_DIR, f"{file_hash}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    file_hash = get_file_hash(uploaded_file)
    
    # Load cached data and model if available
    cached_data, trained_model, date_col, target_col = load_cached_data(file_hash)
    
    if cached_data and trained_model:
        st.write("### Using Cached Data & Model (No Retraining)")
        df, forecast_index, forecast = cached_data
    else:
        # Read and detect file encoding
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding_type = result["encoding"]
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding=encoding_type, encoding_errors="replace")
        df.dropna(inplace=True)

        st.write(f"Detected encoding: {encoding_type}")
        st.write("Dataset Preview:")
        st.write(df.head())

        date_col = None
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce') 
                if df[col].notna().sum() > 0:
                    date_col = col
                    break
            except Exception:
                continue

        if date_col is None:
            st.write("No valid date column found. Using index as time series.")
            df.reset_index(inplace=True)
            date_col = 'index'
        else:
            st.write(f"Detected Date Column: {date_col}")

        df.dropna(subset=[date_col], inplace=True)
        df = df.sort_values(by=date_col)

        target_col = st.selectbox("Select the Target Column (to forecast):", df.columns)
        
        if not np.issubdtype(df[target_col].dtype, np.number):
            st.write("Target column is categorical. Encoding it numerically.")
            df[target_col] = pd.factorize(df[target_col])[0]

        if df[target_col].empty or len(df[target_col]) < 2:
            st.write("Not enough data points to fit a model. Please upload a larger dataset.")
        else:
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df.dropna(subset=[target_col], inplace=True)

            if df[target_col].nunique() == 1:
                st.write("Target variable has only one unique value. Model requires variation in data.")
            else:
                fig, ax = plt.subplots()
                ax.plot(df[date_col], df[target_col], label="Actual Data")
                plt.xticks(rotation=45)
                plt.legend()
                st.pyplot(fig)

                forecast_index = pd.date_range(pd.to_datetime(df[date_col].iloc[-1]) + pd.Timedelta(days=1), periods=10, freq="D")

                if trained_model:
                    st.write("### Using Previously Trained Model for Forecasting")
                    df["timestamp"] = (df[date_col] - df[date_col].min()).dt.days
                    future_timestamps = (forecast_index - df[date_col].min()).days.values.reshape(-1, 1)
                    forecast = trained_model.predict(future_timestamps)
                else:
                    try:
                        model = ARIMA(df[target_col], order=(5, 1, 0))
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=10)
                        trained_model = model_fit
                        st.write("### Forecasting with ARIMA")
                    except Exception:
                        st.write("ARIMA model failed. Falling back to Random Forest.")
                        try:
                            df["timestamp"] = (df[date_col] - df[date_col].min()).dt.days
                            X = df[["timestamp"]]
                            y = df[target_col]
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(X, y)
                            trained_model = model
                            
                            future_timestamps = (forecast_index - df[date_col].min()).days.values.reshape(-1, 1)
                            forecast = model.predict(future_timestamps)
                            st.write("### Forecasting with Random Forest")
                        except Exception as e:
                            st.write("Both ARIMA and Random Forest models failed to fit.")
                            st.write("Error:", str(e))
                            forecast = None

                    if trained_model:
                        save_trained_model(file_hash, trained_model)

                if forecast is not None:
                    save_cached_data(file_hash, (df, forecast_index, forecast), date_col, target_col)

    if forecast is not None:
        fig, ax = plt.subplots()
        ax.plot(df[date_col], df[target_col], label="Actual Data")
        ax.plot(forecast_index, forecast, label="Forecast", color="red")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)
