from pydantic import BaseModel, Field
from typing import List, Dict


class StockData(BaseModel):
    """Dados de um dia de negociação"""

    Open: float = Field(..., description="Preço de abertura")
    High: float = Field(..., description="Preço máximo")
    Low: float = Field(..., description="Preço mínimo")
    Close: float = Field(..., description="Preço de fechamento")
    Volume: float = Field(..., description="Volume negociado")

    class Config:
        schema_extra = {
            "example": {
                "Open": 150.0,
                "High": 152.5,
                "Low": 149.0,
                "Close": 151.5,
                "Volume": 50000000,
            }
        }


class PredictionRequest(BaseModel):
    """Request para predição com dados históricos"""

    historical_data: List[StockData] = Field(
        ..., description="Dados históricos dos últimos dias", min_items=1
    )

    class Config:
        schema_extra = {
            "example": {
                "historical_data": [
                    {
                        "Open": 150.0,
                        "High": 152.5,
                        "Low": 149.0,
                        "Close": 151.5,
                        "Volume": 50000000,
                    }
                ]
                * 60  # 60 dias de exemplo
            }
        }


class AutoPredictionRequest(BaseModel):
    """Request para predição automática"""

    symbol: str = Field(
        ..., description="Símbolo da ação (ex: GOOGL, AAPL, MSFT)", example="GOOGL"
    )


class PredictionResponse(BaseModel):
    """Response da predição"""

    predicted_price: float = Field(..., description="Preço predito")
    last_actual_price: float = Field(..., description="Último preço real")
    expected_change_percent: float = Field(..., description="Variação esperada (%)")
    prediction_date: str = Field(..., description="Data da predição")
    model_info: Dict = Field(..., description="Informações do modelo")


class HealthResponse(BaseModel):
    """Response do health check"""

    status: str
    model_loaded: bool
    config: Dict


class AutoPredictionRequest(BaseModel):
    """Request para predição automática"""

    symbol: str = Field(
        ..., description="Símbolo da ação (ex: GOOGL, AAPL, MSFT)", example="GOOGL"
    )
