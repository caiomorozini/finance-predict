import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def create_sequences(data, seq_length=60):
    X, y = [], []

    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i])
        y.append(data[i, 3])  # Preço de fechamento (índice 3)

    return np.array(X), np.array(y)


def denormalize_price(normalized_price, scaler, close_index=3):
    dummy = np.zeros((len(normalized_price), scaler.n_features_in_))
    dummy[:, close_index] = normalized_price.flatten()
    denormalized = scaler.inverse_transform(dummy)
    return denormalized[:, close_index]


def prepare_data_for_prediction(df, scaler, features, seq_length=60):
    data = df[features].tail(seq_length).values
    scaled_data = scaler.transform(data)
    X_input = scaled_data.reshape(1, seq_length, len(features))
    return X_input


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {"mae": mae, "rmse": rmse, "mape": mape}
