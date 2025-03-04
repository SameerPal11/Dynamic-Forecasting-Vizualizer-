import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

st.title("Dynamic Forecasting Visualizer")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
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

    df = df.sort_values(by=date_col)

    target_col = st.selectbox("Select the Target Column (to forecast):", df.columns)

    if not np.issubdtype(df[target_col].dtype, np.number):
        st.write("Target column is categorical. Encoding it numerically.")
        df[target_col] = pd.factorize(df[target_col])[0]

    fig, ax = plt.subplots()
    ax.plot(df[date_col], df[target_col], label="Actual Data")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig)
        
    model_choice = st.selectbox("Select Forecasting Model:", ["ARIMA", "Random Forest"])

    if model_choice == "ARIMA":
        model = ARIMA(df[target_col], order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)

        st.write("### Forecasting with ARIMA")
        fig, ax = plt.subplots()
        ax.plot(df[date_col], df[target_col], label="Actual Data")
        ax.plot(pd.date_range(df[date_col].iloc[-1], periods=10, freq="D"), forecast, label="Forecast", color="red")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

    elif model_choice == "Random Forest":
        df["timestamp"] = (df[date_col] - df[date_col].min()).dt.days
        X = df[["timestamp"]]
        y = df[target_col]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        future_dates = pd.date_range(df[date_col].iloc[-1], periods=10, freq="D")
        future_timestamps = (future_dates - df[date_col].min()).days.values.reshape(-1, 1)
        forecast = model.predict(future_timestamps)

        st.write("### Forecasting with Random Forest")
        fig, ax = plt.subplots()
        ax.plot(df[date_col], df[target_col], label="Actual Data")
        ax.plot(future_dates, forecast, label="Forecast", color="red")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)
else:
    print("upload your file")
