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
# Mapeia exatamente o JSON esperado pelo ColumnTransformer do nosso Pipeline
class PayloadPaciente(BaseModel):
    MUNIC_RES: str
    MUNIC_MOV: str
    IDADE: str
    SEXO: str
    DIAG_PRINC: str
    CAR_INT: str
    INSTRU: str
    GESTAO: str

# 3. Carregamento Seguro do Modelo do Model Registry via Alias @production
try:
    print("-> Carregando o modelo campeão a partir do Model Registry...")
    # O MLflow resolve a URI nativamente buscando a Versão 4 marcada como estável
    model_uri = "models:/modelo_sih_sus_catastrofico@production"
    modelo_producao = mlflow.pyfunc.load_model(model_uri)
    print("Sucesso! Modelo em @production acoplado e pronto para operação.")
except Exception as e:
    print(f"Alerta de Caminho: Carregamento por alias exigiu fallback físico ({e})")
    import glob
    # Fallback de segurança buscando a última pasta de artefatos gerada localmente
    pastas_modelos = glob.glob("mlruns/**/artifacts/model", recursive=True)
    if pastas_modelos:
        modelo_producao = mlflow.pyfunc.load_model(pastas_modelos[-1])
        print(f"Modelo carregado via fallback local: {pastas_modelos[-1]}")
    else:
        raise RuntimeError("Erro Crítico: O artefato do modelo não foi localizado no servidor MLflow.")

# 4. Endpoint de Health Check (Monitoramento de Saúde da API)
@app.get("/")
def checar_saude():
    """Retorna o status operacional da API para ferramentas de monitoramento."""
    return {
        "status": "operacional",
        "modelo": "modelo_sih_sus_catastrofico",
        "ambiente": "production"
    }

# 5. Endpoint de Inferência (Onde a predição acontece)
@app.post("/predict")
def predizer_risco_custo(paciente: PayloadPaciente):
    """
    Recebe os dados cadastrais da admissão de uma internação via JSON 
    e infere se o custo final estourará o teto orçamentário do P90.
    """
    try:
        # Converte o payload do JSON recebido em um DataFrame estruturado de 1 linha
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
        
        # O DataFrame bruto passa direto pelo Pipeline do scikit-learn carregado
        predicao = modelo_producao.predict(dados_entrada)
        classe_final = int(predicao[0])
        
        # Mapeamento semântico da saída para facilitar a tomada de decisão do gestor do SUS
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