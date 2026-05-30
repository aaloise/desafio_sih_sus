import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc

# Silencia logs desnecessários do Git Portable no ecossistema Python
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# 1. Inicialização do Microsserviço com Metadados de Produção
app = FastAPI(
    title="Radar de Custo Catastrófico Hospitalar - SIH/SUS",
    description="API de produção para inferência em tempo real de risco orçamentário hospitalar.",
    version="2.0.0"
)

# 2. Camada de Validação de Dados (Pydantic)
class PayloadPaciente(BaseModel):
    MUNIC_RES: str
    MUNIC_MOV: str
    IDADE: str
    SEXO: str
    DIAG_PRINC: str
    CAR_INT: str
    INSTRU: str
    GESTAO: str

# 3. Carregamento Seguro e Auto-Detectável do Modelo
try:
    print("-> Tentando carregar o modelo via Model Registry...")
    model_uri = "models:/modelo_sih_sus_catastrofico@production"
    modelo_producao = mlflow.pyfunc.load_model(model_uri)
    print("Sucesso! Modelo em @production acoplado.")
except Exception as e:
    print(f"Alerta: Carregamento por alias exigiu varredura física no container ({e})")
    
    # BUSCA IDENTIFICADORA DE ARTEFATOS: Localiza o modelo pelos arquivos obrigatórios do MLflow
    pastas_modelos = []
    for raiz, diretorios, arquivos in os.walk("."):
        if "MLmodel" in arquivos or "model.pkl" in arquivos:
            pastas_modelos.append(raiz)
            
    if pastas_modelos:
        pastas_modelos.sort()
        modelo_producao = mlflow.pyfunc.load_model(pastas_modelos[-1])
        print(f"Modelo localizado e acoplado com sucesso via varredura: {pastas_modelos[-1]}")
    else:
        # MAPEAMENTO DE SEGURANÇA: Se não achar nada, lista o conteúdo para auditoria visual imediata
        mapeamento_pastas = []
        if os.path.exists("mlruns"):
            for r, d, f in os.walk("mlruns"):
                if f:
                    mapeamento_pastas.append(f"{r}: {f}")
        raise RuntimeError(
            f"Erro Crítico: Os arquivos físicos do modelo (model.pkl ou MLmodel) não foram encontrados em mlruns. "
            f"Estrutura de arquivos existente no container: {mapeamento_pastas}"
        )

# 4. Endpoint de Health Check (Monitoramento de Saúde da API)
@app.get("/")
def checar_saude():
    return {
        "status": "operacional",
        "modelo": "modelo_sih_sus_catastrofico",
        "ambiente": "production"
    }

# 5. Endpoint de Inferência (Onde a predição acontece)
@app.post("/predict")
def predizer_risco_custo(paciente: PayloadPaciente):
    try:
        dados_entrada = pd.DataFrame([{
            'MUNIC_RES': paciente.MUNIC_RES,
            'MUNIC_MOV': paciente.MUNIC_MOV,
            'IDADE': paciente.IDADE,
            'SEXO': paciente.SEXO,
            'DIAG_PRINC': paciente.DIAG_PRINC,
            'CAR_INT': paciente.CAR_INT,
            'INSTRU': paciente.INSTRU,
            'GESTAO': paciente.GESTAO
        }])
        
        predicao = modelo_producao.predict(dados_entrada)
        classe_final = int(predicao[0])
        
        if classe_final == 1:
            diagnostico_radar = "ALTO RISCO: Internação com alto potencial de Custo Catastrófico (Estouro do P90)."
        else:
            diagnostico_radar = "Custo Controlado: Previsão de gastos dentro do teto financeiro padrão."
            
        return {
            "alvo_binario": classe_final,
            "radar_diagnostico": diagnostico_radar
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno no motor de inferência do modelo: {str(e)}"
        )
