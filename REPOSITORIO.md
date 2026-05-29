# Link do Repositório Oficial – Desafio SIH-SUS

O código-fonte completo, histórico de commits, arquivos de configuração do Docker e artefatos de experimentação estão disponíveis no repositório público do GitHub:

**🔗 Repositório:**
https://github.com/aaloise/desafio_sih_sus

## Conteúdo do Repositório

* `src/app.py`
  API de produção desenvolvida com FastAPI para inferência em tempo real.

* `src/train.py`
  Esteira automatizada de treinamento, validação cruzada, seleção de modelos e otimização de hiperparâmetros.

* `src/baixar_dados.py`
  Script responsável pela ingestão automatizada dos dados do SIH-SUS.

* `Dockerfile` e `docker-compose.yml`
  Arquivos de infraestrutura para conteinerização e execução da aplicação em ambientes isolados.

* `notebooks/`
  Relatórios de Análise Exploratória de Dados (EDA) disponibilizados nos formatos:

  * `.ipynb`
  * `.html`
  * `.pdf`

* `mlruns/`
  Repositório local de experimentos e métricas gerenciados pelo MLflow.

## Tecnologias Utilizadas

* Python 3.12
* Scikit-Learn
* XGBoost
* LightGBM
* MLflow
* FastAPI
* Docker
* Pandas
* NumPy

## Reprodutibilidade

Todas as dependências necessárias para execução do projeto estão especificadas no arquivo `requirements.txt`, permitindo a reprodução integral dos experimentos e do ambiente de produção.
