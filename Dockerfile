# Usar imagem oficial do Python 3.12
FROM python:3.12-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas arquivos necessários primeiro (para cache)
COPY pyproject.toml ./

# Instalar uv e dependências
RUN pip install --no-cache-dir uv && \
    uv pip install --system -r pyproject.toml

# Copiar código da API e src
COPY api/ ./api/
COPY src/ ./src/

# Copiar modelos treinados
COPY models/ ./models/

# Expor porta
EXPOSE 8000

# Variáveis de ambiente
ENV MODEL_PATH=/app/models/lstm_stock_predictor.keras
ENV SCALER_PATH=/app/models/scaler.pkl
ENV CONFIG_PATH=/app/models/model_config.json
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para iniciar a API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
