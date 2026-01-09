from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import pandas as pd

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    AutoPredictionRequest,
    HealthResponse,
)
from api.config import model_config
from api.services import make_prediction, fetch_stock_data


app = FastAPI(
    title="Tech Challenge - Fase 4 FIAP",
    description="API para predição de preços de ações usando modelo LSTM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Carrega modelo e artefatos na inicialização"""
    model_config.load_artifacts()


@app.get("/", response_model=Dict)
async def root():
    """Endpoint raiz"""
    return {
        "message": "Stock Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (com dados históricos)",
            "predict_auto": "/predict-auto (coleta automática)",
            "model_info": "/model-info",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica saúde da API"""
    return {
        "status": "healthy" if model_config.is_loaded() else "unhealthy",
        "model_loaded": model_config.is_loaded(),
        "config": model_config.config if model_config.config else {},
    }


@app.get("/model-info", response_model=Dict)
async def get_model_info():
    """Retorna informações sobre o modelo carregado"""
    if not model_config.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    config = model_config.config
    model = model_config.model

    return {
        "symbol": config["symbol"],
        "features": config["features"],
        "sequence_length": config["seq_length"],
        "metrics": config["metrics"],
        "training_date": config["training_date"],
        "model_architecture": {
            "total_params": model.count_params(),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    """
    Prediz o preço de ações usando dados históricos fornecidos manualmente
    """
    if not model_config.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    try:
        seq_length = model_config.config["seq_length"]
        if len(request.historical_data) < seq_length:
            raise HTTPException(
                status_code=400,
                detail=f"Necessário pelo menos {seq_length} dias de dados históricos. Recebido: {len(request.historical_data)}",
            )

        data_list = [item.dict() for item in request.historical_data[-seq_length:]]
        df = pd.DataFrame(data_list)

        return make_prediction(df, data_source="manual")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@app.post("/predict-auto", response_model=PredictionResponse)
async def predict_stock_price_auto(request: AutoPredictionRequest):
    """
    Prediz o preço de fechamento do próximo dia coletando dados automaticamente

    Coleta automaticamente os últimos 60 dias de dados históricos do Yahoo Finance
    e faz a predição. Mais conveniente que /predict mas requer conexão com Yahoo Finance.
    """
    if not model_config.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    try:
        config = model_config.config
        if request.symbol.upper() != config["symbol"].upper():
            raise HTTPException(
                status_code=400,
                detail=f"Modelo treinado para {config['symbol']}, mas recebeu {request.symbol.upper()}. Use o símbolo correto ou retreine o modelo.",
            )

        df = fetch_stock_data(request.symbol, config["seq_length"])

        return make_prediction(
            df, data_source="Yahoo Finance (coletado automaticamente)"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro na predição automática: {str(e)}"
        )


@app.get("/predict-example")
async def get_prediction_example():
    """Retorna um exemplo de como usar o endpoint /predict"""
    if not model_config.is_loaded():
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    config = model_config.config

    return {
        "info": "Exemplo de requisição para /predict",
        "required_days": config["seq_length"],
        "features": config["features"],
        "example_request": {
            "historical_data": [
                {
                    "Open": 150.0,
                    "High": 152.5,
                    "Low": 149.0,
                    "Close": 151.5,
                    "Volume": 50000000,
                }
            ]
            *config["seq_length"]
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
