import hashlib
import os
import pickle  
import boto3
import json
import streamlit as st
# Define storage directories
STORAGE_DIR = "saved_forecasts"
MODEL_DIR = "saved_models"
CHAT_HISTORY_FILE = "chat_history/chat_history.pkl"  # Local file to store chat history

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)# Initialize AWS Bedrock client


bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def query_claude(prompt, dataset_text=""):
    model_id = "anthropic.claude-instant-v1"

    # Limit dataset snippet to 2000 characters
    dataset_snippet = dataset_text[:2000] if dataset_text else "No dataset provided."

    # Construct the API payload dynamically
    payload = {
        "prompt": f"\n\nHuman: Here is a dataset snippet:\n{dataset_snippet}\n\nNow answer this question:\n{prompt}\n\nAssistant:",
        "max_tokens_to_sample": 500,
        "temperature": 0.7,
        "top_k": 250,
        "top_p": 0.9,
        "stop_sequences": ["\n\nHuman:"],
        "anthropic_version": "bedrock-2023-05-31"
    }

    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="*/*",
            body=json.dumps(payload)  # Convert dictionary to JSON string
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        return response_body.get("completion", "").strip()
    
    except boto3.exceptions.Boto3Error as e:
        return f"Error processing query: AWS Boto3 Error - {str(e)}"
    
    except Exception as e:
        return f"Error processing query: {str(e)}"



@st.cache_resource
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "rb") as file:
            return pickle.load(file)
    return []

@st.cache_resource
# Function to save chat history to a local file
def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, "wb") as file:
        pickle.dump(chat_history, file)

# @st.cache_resource
# def load_qa_model():
#     return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")




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

# def train_lstm_model(X, y):
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#     model = Sequential([
#         LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
#         LSTM(50, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
#     model.fit(X, y, epochs=50, verbose=0)
#     return model
