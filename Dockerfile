# 1. Usa a imagem oficial estável do Python 3.12
FROM python:3.12-slim

# 2. Define o diretório de trabalho interno do container
WORKDIR /app

# 3. Instala dependências essenciais do sistema operacional para o LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copia e instala os requerimentos do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia as pastas de código e o repositório de logs do MLflow para dentro do container
COPY src/ /app/src/
COPY mlruns/ /app/mlruns/

# 6. Expõe a porta padrão que a API usará
EXPOSE 8000

# 7. Comando para iniciar o FastAPI via Uvicorn no container
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
