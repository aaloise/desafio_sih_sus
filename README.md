\# Radar de Custo Catastrófico Hospitalar (SIH-SUS)



Este projeto implementa uma esteira completa de engenharia de machine learning (MLOps) para a identificação precoce de internações com risco de \*\*Custo Catastrófico Hospitalar\*\* no âmbito do Sistema Único de Saúde (SUS), utilizando os dados de AIH (Autorização de Internação Hospitalar) de Dezembro de 2024.



O objetivo do negócio é atuar como um radar preventivo na recepção do hospital. Para evitar \*\*Data Leakage (vazamento de dados)\*\*, o modelo foi blindado para utilizar exclusivamente características cadastrais disponíveis \*\*no exato momento da admissão do paciente\*\*.



\---



\## 📅 Fonte de Dados e Licença



\* \*\*Origem:\*\* Ministério da Saúde - Departamento de Informática do SUS (DataSUS).

\* \*\*Dataset:\*\* Sistema de Informações Hospitalares (SIH/SUS) - Arquivos de AIH Reduzida.

\* \*\*Filtro Utilizado:\*\* Competência de Dezembro de 2024 (Arquivo `RDAM2412.csv` referente ao estado do Amazonas).

\* \*\*Endereço de Disseminação:\*\* Mecanismo de extração automatizado via FTP do DataSUS (`ftp.datasus.gov.br/dissemin/publicos/SIHSUS/200801\_/Dados/`).

\* \*\*Licença:\*\* Domínio Público / Dados Abertos Governamentais (Lei de Acesso à Informação nº 12.527/2011).



\---



\## 📊 Resumo dos Resultados Técnicos



\* \*\*Definição do Alvo (Target):\*\* Binarização baseada no corte rigoroso do percentil 90 ($P\_{90}$) dos custos nacionais da base do SIH.

\* \*\*Proporção de Classes:\*\* Classe 0 (Custo Controlado) = 90% | Classe 1 (Custo Catastrófico) = 10%.

\* \*\*Métrica Central de Avaliação:\*\* \*\*F1-Macro\*\*, escolhida devido ao desbalanceamento severo das classes para mitigar o viés em favor da classe majoritária.

\* \*\*Abordagem Competitiva (Fase de Seleção):\*\* Avaliação estatística via Validação Cruzada Estratificada em 5 dobras (5-Fold Stratified CV) comparando três algoritmos:

&#x20;   \* \*Decision Tree\* -> F1-Macro CV: \*\*0.6147 +- 0.0023\*\* (Vencedora)

&#x20;   \* \*XGBoost\* -> F1-Macro CV: 0.6052 +- 0.0013

&#x20;   \* \*LightGBM\* -> F1-Macro CV: 0.5915 +- 0.0055

\* \*\*Otimização (Tuning):\*\* RandomizedSearchCV refinou a árvore final para `max\_depth=20` e `min\_samples\_split=5`.

\* \*\*Performance no Teste de Produção (Holdout Invisível):\*\* \*\*F1-Macro de 0.5771\*\* sobre mais de 218 mil linhas inéditas, alcançando um \*\*Recall de 60,18%\*\* na captura dos custos catastróficos.



\---



\## 🧠 Decisões de Engenharia e Pré-Processamento



Para garantir a reprodutibilidade integral sem vazamento de dados, todas as transformações foram encapsuladas em um `ColumnTransformer` do Scikit-Learn:

1\. \*\*Variáveis Numéricas (`IDADE`):\*\* Tratamento de dados ausentes via `SimpleImputer(strategy='median')` seguido de padronização por `StandardScaler()`.

2\. \*\*Engenharia de Alta Cardinalidade (`DIAG\_PRINC`):\*\* O código CID-10 original registrou mais de 7.000 valores únicos. Foi aplicado um `FunctionTransformer` customizado para extrair apenas a letra inicial (Capítulo do CID-10), reduzindo drasticamente o espaço dimensional antes da aplicação do `OneHotEncoder(handle\_unknown='ignore')`.

3\. \*\*Variáveis Categóricas:\*\* Preenchimento por frequência via `SimpleImputer(strategy='most\_frequent')` e codificação via `OneHotEncoder(handle\_unknown='ignore')`.



\---



\## 📁 Estrutura do Repositório



```text

desafio\_sih\_sus/

├── data/

│   ├── processed/          # Base unificada e tratada (sih\_processed.csv)

│   └── raw/                # Dado bruto original do DataSUS (RDAM2412.csv)

├── mlruns/                 # Repositório local de experimentos e logs do MLflow

├── notebooks/

│   ├── 01\_eda.html         # Relatório exploratório exportado em HTML

│   ├── 01\_eda.ipynb        # Notebook Jupyter com a análise exploratória base

│   └── 01\_eda.pdf          # Documento final da EDA exportado em formato PDF

├── src/

│   ├── app.py              # Motor de inferência em tempo real (API FastAPI)

│   ├── baixar\_dados.py     # Script automatizado de ingestão de dados via FTP

│   ├── gerar\_eda\_notebook.py # Gerador programático das células do notebook de EDA

│   └── train.py            # Esteira MLOps (Validação Cruzada, Tuning e Registro Automático)

├── Dockerfile              # Receita de build da imagem isolada (Python 3.12-slim)

├── docker-compose.yml      # Orquestrador do microsserviço de produção

├── requirements.txt        # Dependências do projeto com versões estritas (pinned)

└── .gitignore              # Filtro de proteção do Git (ignora dados massivos e venv)

```



\---



\## 🛠️ Como Executar o Projeto Nativamente (Ambiente Local)



\### 1. Preparar o Ambiente Virtual (Python 3.12)

```powershell

\# Clonar o repositório (Substitua pela sua URL)

git clone \[https://github.com/seu-usuario/desafio\_sih\_sus.git](https://github.com/seu-usuario/desafio\_sih\_sus.git)

cd desafio\_sih\_sus



\# Instalação das dependências

python -m venv .venv

.venv\\Scripts\\Activate.ps1

python -m pip install -r requirements.txt

```



\### 2. Pipeline de Dados e Treinamento

```powershell

\# Passo A: Baixar os dados brutos e estruturar os diretórios

python src/baixar\_dados.py



\# Passo B: Executar a esteira completa (Treino, Seleção de Modelos e Otimização)

\# Nota de Arquitetura: A promoção do modelo campeão para a tag @production é feira de forma 100% automatizada ao final deste script através do MLflow Client.

python src/train.py

```



\### 3. Interface de Governança (MLflow)

Para auditar a matriz de confusão e comparar as curvas dos modelos concorrentes:

```powershell

mlflow ui

```

Acesse: http://127.0.0.1:5000



\---



\## 🐳 Como Executar via Docker Compose (Ambiente de Produção)



A aplicação está totalmente containerizada, garantindo independência de infraestrutura. Para construir a imagem e subir a API do FastAPI em segundo plano:



```powershell

docker compose up --build -d

```

Acesse a documentação interativa e execute predições em tempo real pelo Swagger:

👉 \*\*http://127.0.0.1:8000/docs\*\*



\---



\## ⚠️ Limitações Conhecidas e Próximos Passos



\### Limitações do Modelo Atual:

1\. \*\*Janela Temporal Estática:\*\* O modelo foi treinado com dados de uma única competência mensal (Dezembro/2024), impossibilitando a captura de sazonalidades clínicas (ex: surtos de arboviroses ou doenças respiratórias de inverno).

2\. \*\*Volumetria de Falsos Positivos:\*\* O modelo exibe uma taxa elevada de falsos alarmes (64.181 casos no teste). Embora aceitável sob a ótica de uma auditoria preventiva de riscos financeiros, pode gerar fadiga de alertas em um cenário operacional real.



\### Próximos Passos Recomendados:

\* \*\*Janela Histórica Expandida:\*\* Reexecutar o pipeline consumindo uma série histórica de 24 meses via DVC para estabilizar os gradientes do XGBoost e LightGBM.

\* \*\*Tuning Massivo de Ensembles:\*\* Expandir o espaço de busca de hiperparâmetros (aumentando `n\_estimators` e profundidade) dos modelos de Gradient Boosting em instâncias com maior capacidade computacional.

\* \*\*Monitoramento de Data Drift:\*\* Implementar rotinas com a biblioteca Evidently AI na API para monitorar mudanças no perfil de entrada das internações do SUS ao longo do tempo.

