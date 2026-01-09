import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from fastapi import HTTPException

from api.schemas import PredictionResponse
from api.config import model_config


def make_prediction(
    df: pd.DataFrame, data_source: str = "manual"
) -> PredictionResponse:
    """
    Executa a predição a partir de um DataFrame.

    Args:
        df: DataFrame com dados históricos (já no formato correto)
        data_source: Fonte dos dados para incluir na resposta

    Returns:
        PredictionResponse com a predição
    """
    config = model_config.config
    model = model_config.model
    scaler = model_config.scaler

    seq_length = config["seq_length"]

    # Garantir ordem correta das colunas
    df = df[config["features"]]

    # Normalizar dados
    scaled_data = scaler.transform(df.values)

    # Reshape para (1, seq_length, n_features)
    X_input = scaled_data.reshape(1, seq_length, len(config["features"]))

    # Fazer predição
    scaled_prediction = model.predict(X_input, verbose=0)

    # Desnormalizar predição
    dummy = np.zeros((1, len(config["features"])))
    dummy[0, 3] = scaled_prediction[0, 0]  # Index 3 = Close
    predicted_price = scaler.inverse_transform(dummy)[0, 3]

    # Último preço real
    last_price = float(df["Close"].iloc[-1])

    # Calcular variação esperada
    change_percent = ((predicted_price / last_price) - 1) * 100

    # Montar informações do modelo
    model_info = {
        "symbol": config["symbol"],
        "mae": config["metrics"]["test_mae"],
        "rmse": config["metrics"]["test_rmse"],
        "mape": config["metrics"]["test_mape"],
    }

    if data_source != "manual":
        model_info["data_source"] = data_source
        model_info["days_used"] = seq_length

    return PredictionResponse(
        predicted_price=round(float(predicted_price), 2),
        last_actual_price=round(float(last_price), 2),
        expected_change_percent=round(float(change_percent), 2),
        prediction_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_info=model_info,
    )


def fetch_stock_data(symbol: str, days_needed: int) -> pd.DataFrame:
    """
    Coleta dados históricos de uma ação do Yahoo Finance.

    Args:
        symbol: Símbolo da ação (ex: GOOGL)
        days_needed: Quantidade de dias úteis necessários

    Returns:
        DataFrame com os dados históricos

    Raises:
        HTTPException: Em caso de erro na coleta
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(
            days=days_needed * 2
        )  # 2x para garantir dias úteis

        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Nenhum dado encontrado para o símbolo {symbol.upper()}. Verifique se o símbolo está correto.",
            )

        if len(df) < days_needed:
            raise HTTPException(
                status_code=400,
                detail=f"Dados insuficientes. Necessário {days_needed} dias, obtido {len(df)} dias.",
            )

        return df.tail(days_needed)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao coletar dados do Yahoo Finance: {str(e)}",
        )
