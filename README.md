# Radar de Custo Catastrófico Hospitalar (SIH-SUS)

Este projeto implementa uma esteira completa de engenharia de machine learning (MLOps) para a identificação precoce de internações com risco de **Custo Catastrófico Hospitalar** no âmbito do Sistema Único de Saúde (SUS), utilizando os dados de AIH (Autorização de Internação Hospitalar) de dezembro de 2024.

O objetivo do negócio é atuar como um radar preventivo na recepção do hospital. Para evitar **Data Leakage (vazamento de dados)**, o modelo foi blindado para utilizar exclusivamente características cadastrais disponíveis **no exato momento da admissão do paciente**.

---

## 📅 Fonte de Dados e Licença

* **Origem:** Ministério da Saúde – Departamento de Informática do SUS (DataSUS).
* **Dataset:** Sistema de Informações Hospitalares (SIH/SUS) – Arquivos de AIH Reduzida.
* **Filtro Utilizado:** Competência de dezembro de 2024 (arquivo `RDAM2412.csv`).
* **Endereço de Disseminação:** Mecanismo de extração automatizado via Kaggle [SIH/SUS - Hospital Admissions Municipalities 2024](https://www.kaggle.com/datasets/andersonfranca/sistema-de-informaes-hospitalares-sus).
* **Licença:** Domínio público / Dados Abertos Governamentais (Lei nº 12.527/2011).

---

## 📊 Resumo dos Resultados Técnicos

* **Definição do Alvo (Target):** Binarização baseada no corte rigoroso do 90º percentil (P90) dos custos nacionais da base do SIH.
* **Proporção de Classes:** Classe 0 (Custo Controlado) = 90% | Classe 1 (Custo Catastrófico) = 10%.
* **Métrica Central de Avaliação:** **F1-Macro**, escolhida devido ao desbalanceamento severo das classes para mitigar o viés em favor da classe majoritária.
* **Abordagem Competitiva (Fase de Seleção):** Avaliação estatística via Validação Cruzada Estratificada em 5 dobras (5-Fold Stratified CV), comparando três algoritmos:

  * *Decision Tree* → F1-Macro CV: **0,6147 ± 0,0023** (Vencedora)
  * *XGBoost* → F1-Macro CV: **0,6052 ± 0,0013**
  * *LightGBM* → F1-Macro CV: **0,5915 ± 0,0055**
* **Otimização (Tuning):** RandomizedSearchCV refinou a árvore final para `max_depth=20` e `min_samples_split=5`.
* **Performance no Teste de Produção (Holdout Invisível):** **F1-Macro de 0,5771** sobre mais de 218 mil linhas inéditas, alcançando um **Recall de 60,18%** na captura dos custos catastróficos.

---

## 🧠 Decisões de Engenharia e Pré-Processamento

Para garantir a reprodutibilidade integral sem vazamento de dados, todas as transformações foram encapsuladas em um `ColumnTransformer` do Scikit-Learn:

### 1. Variáveis Numéricas (`IDADE`)

* Tratamento de dados ausentes via `SimpleImputer(strategy='median')`.
* Padronização por `StandardScaler()`.

### 2. Engenharia de Alta Cardinalidade (`DIAG_PRINC`)

* O código CID-10 original registrou mais de 7.000 valores únicos.
* Foi aplicado um `FunctionTransformer` customizado para extrair apenas a letra inicial (Capítulo do CID-10), reduzindo drasticamente o espaço dimensional antes da aplicação do `OneHotEncoder(handle_unknown='ignore')`.

### 3. Variáveis Categóricas

* Preenchimento por frequência via `SimpleImputer(strategy='most_frequent')`.
* Codificação via `OneHotEncoder(handle_unknown='ignore')`.

---

## 📁 Estrutura do Repositório

```text
desafio_sih_sus/
├── data/
│   ├── processed/
│   │   └── sih_processed.csv
│   └── raw/
│       └── RDAM2412.csv.csv
├── mlruns/
├── notebooks/
│   ├── 01_eda.html
│   ├── 01_eda.ipynb
│   └── 01_eda.pdf
├── src/
│   ├── app.py
│   ├── baixar_dados.py
│   ├── gerar_eda_notebook.py
│   └── train.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Como Executar o Projeto Nativamente (Ambiente Local)

### 1. Preparar o Ambiente Virtual (Python 3.12)

```powershell
# Clonar o repositório
git clone https://github.com/aaloise/desafio_sih_sus.git

# Entrar na pasta do projeto
cd desafio_sih_sus

# Criar e ativar o ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1

# Instalar dependências
python -m pip install -r requirements.txt
```
## 2. Versionamento de Dados com DVC

Este projeto utiliza o **DVC (Data Version Control)** para versionar o dataset sem comitar arquivos grandes no Git. O dataset bruto (>65MB) não está no repositório — apenas os arquivos `.dvc` de metadados.

### Por que usar DVC?
- Evita estourar o limite de 100MB do GitHub
- Mantém histórico de versões do dataset
- Permite reprodução exata do ambiente de treino
- Separa código (Git) de dados (DVC)

### Como restaurar o dataset localmente

```bash
# 1. Instalar o DVC (se ainda não tiver)
pip install dvc

# 2. Inicializar o DVC no repositório (apenas na primeira vez)
dvc init

# 3. Configurar o remote local para armazenamento dos dados
# Windows:
dvc remote add -d local C:/dvcstore
# Linux/Mac:
dvc remote add -d local ~/dvcstore

# 4. Baixar os dados versionados
dvc pull
```

## 3. Pipeline de Dados e Treinamento

### Passo A: Baixar os dados brutos
```
python src/baixar_dados.py
```

### Passo B: Executar a esteira completa
```
python src/train.py
```

**Observação:** a promoção automática do modelo campeão para o estágio de produção é realizada ao final do treinamento por meio da API do MLflow.

### 4. Interface de Governança (MLflow)

```powershell
mlflow ui
```

Acesse: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🐳 Como Executar via Docker Compose (Ambiente de Produção)

A aplicação está totalmente containerizada, garantindo independência de infraestrutura.

```powershell
docker compose up --build -d
```

Acesse a documentação interativa da API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ⚠️ Limitações Conhecidas e Próximos Passos

### Limitações do Modelo Atual

#### 1. Janela Temporal Estática

O modelo foi treinado com dados de uma única competência mensal (dezembro/2024), impossibilitando a captura de sazonalidades clínicas, como surtos de arboviroses ou doenças respiratórias.

#### 2. Volumetria de Falsos Positivos

O modelo exibe uma taxa elevada de falsos alarmes (64.181 casos no conjunto de teste). Embora aceitável para auditoria preventiva de riscos financeiros, pode gerar fadiga de alertas em ambiente operacional.

### Próximos Passos Recomendados

* **Janela Histórica Expandida:** reexecutar o pipeline consumindo uma série histórica de 24 meses via DVC.
* **Tuning Massivo de Ensembles:** expandir o espaço de busca dos hiperparâmetros dos modelos XGBoost e LightGBM.
* **Monitoramento de Data Drift:** implementar monitoramento contínuo utilizando Evidently AI.
* **Avaliação de Custos Operacionais:** incorporar métricas de custo-benefício para calibrar limiares de decisão conforme a capacidade operacional da rede hospitalar.

---

## 📈 Arquitetura da Solução

```text
Kaggle (SIH/SUS - Hospital Admissions Municipalities 2024)
       │
       ▼
baixar_dados.py
       │
       ▼
Dados Brutos (CSV)
       │
       ▼
Pré-processamento
(ColumnTransformer)
       │
       ▼
Validação Cruzada
(Decision Tree, XGBoost, LightGBM)
       │
       ▼
Tuning Automático
(RandomizedSearchCV)
       │
       ▼
Registro no MLflow
       │
       ▼
Modelo em Produção
       │
       ▼
FastAPI + Swagger
```

---

## 📜 Autor

**André Filipe Aloise**

Projeto desenvolvido como solução para o Desafio 01 de Data Science - AX Academy
