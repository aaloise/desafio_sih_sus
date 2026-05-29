import os
# Silencia o aviso do Git Portable no ecossistema MLflow
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import confusion_matrix, f1_score

# Importação dos novos titãs de Gradient Boosting
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Função para reduzir a cardinalidade do CID (Engenharia de Recursos)
def extrair_capitulo_cid(X):
    return pd.DataFrame(X).iloc[:, 0].str[0].to_frame()

def executar_esteira_mlops_avancada():
    print("-> Carregando dados processados...")
    df = pd.read_csv("data/processed/sih_processed.csv")
    
    # Garante que o alvo seja lido estritamente como inteiro para o XGBoost
    df['TARGET_CUSTO_CATASTROFICO'] = df['TARGET_CUSTO_CATASTROFICO'].astype(int)
    
    X = df.drop(columns=['TARGET_CUSTO_CATASTROFICO', 'VAL_TOT'])
    y = df['TARGET_CUSTO_CATASTROFICO']
    
    # Split de teste definitivo (80/20) com estratificação
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"-> Base total dividida. Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")
    
    # Amostragem técnica de 100k linhas para CV e Tuning rápidos (Eficiência de Processamento)
    X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=100000, random_state=42, stratify=y_train)
    
    # Cálculo dinâmico do peso para o XGBoost mitigar o desbalanceamento 90/10
    # Formula: total de negativos / total de positivos
    peso_positivo_xgb = (y_train_sub == 0).sum() / (y_train_sub == 1).sum()
    
    # Pipelines de Pré-processamento estruturados via ColumnTransformer
    pipe_num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    pipe_cid = Pipeline([('extrator', FunctionTransformer(extrair_capitulo_cid)), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    pipe_cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer([
        ('numeric', pipe_num, ['IDADE']),
        ('cid', pipe_cid, ['DIAG_PRINC']),
        ('categorical', pipe_cat, ['SEXO', 'GESTAO', 'CAR_INT', 'MUNIC_RES', 'MUNIC_MOV', 'INSTRU'])
    ])
    
    # Dicionário atualizado incluindo os 3 modelos exigidos pelo edital
    modelos_comparacao = {
        "DecisionTree": DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(max_depth=6, scale_pos_weight=peso_positivo_xgb, eval_metric='logloss', random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    }
    
    mlflow.set_experiment("Radar_Custo_Catastrofico_SUS")
    
    melhor_nome = None
    melhor_score = -1
    
    print("\n=== FASE 1: COMPARAÇÃO DE MODELOS COM 5-FOLD CV (AMOSTRA 100K) ===")
    for nome, clf in modelos_comparacao.items():
        with mlflow.start_run(run_name=f"baseline_{nome}"):
            pipe_teste = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
            
            # Validação Cruzada de 5 dobras calculando o F1-Macro estatístico
            cv_scores = cross_validate(pipe_teste, X_train_sub, y_train_sub, cv=5, scoring='f1_macro', n_jobs=-1)
            mean_f1 = cv_scores['test_score'].mean()
            std_f1 = cv_scores['test_score'].std()
            
            print(f"{nome} -> F1-Macro CV: {mean_f1:.4f} ± {std_f1:.4f}")
            
            # Logs de governança no MLflow UI
            mlflow.log_param("algoritmo", nome)
            mlflow.log_metric("f1_macro_cv_mean", mean_f1)
            mlflow.log_metric("f1_macro_cv_std", std_f1)
            
            if mean_f1 > melhor_score:
                melhor_score = mean_f1
                melhor_nome = nome

    print(f"\n Modelo vencedor da triagem estatística: {melhor_nome} (F1: {melhor_score:.4f})")
    
    print("\n=== FASE 2: TUNING DE HIPERPARÂMETROS DO VENCEDOR (RANDOM SEARCH) ===")
    with mlflow.start_run(run_name=f"tuning_{melhor_nome}"):
        
        # Grades de busca otimizadas dependendo de quem ganhou a triagem
        if melhor_nome == "DecisionTree":
            param_dist = {'classifier__max_depth': [10, 15, 20], 'classifier__min_samples_split': [2, 5]}
        elif melhor_nome == "XGBoost":
            param_dist = {'classifier__max_depth': [5, 7], 'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.05, 0.1]}
        elif melhor_nome == "LightGBM":
            param_dist = {'classifier__max_depth': [6, 8], 'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.05, 0.1]}
            
        pipe_tuning = Pipeline([('preprocessor', preprocessor), ('classifier', modelos_comparacao[melhor_nome])])
        
        search = RandomizedSearchCV(pipe_tuning, param_dist, n_iter=3, cv=3, scoring='f1_macro', n_jobs=-1, random_state=42)
        search.fit(X_train_sub, y_train_sub)
        
        print(f" Melhores Parâmetros Encontrados: {search.best_params_}")
        mlflow.log_params(search.best_params_)
        
        modelo_final = search.best_estimator_
        
        print("\n=== FASE 3: TREINAMENTO FINAL NA BASE INTEGRAL DE 1 MILHÃO DE LINHAS ===")
        modelo_final.fit(X_train, y_train)
        
        # Teste cego definitivo contra o Holdout isolado
        y_pred = modelo_final.predict(X_test)
        f1_final = f1_score(y_test, y_pred, average='macro')
        print(f"F1-Macro Final no Teste de Produção: {f1_final:.4f}")
        mlflow.log_metric("f1_macro_teste_final", f1_final)
        
        # Geração do artefato gráfico da Matriz de Confusão
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matriz de Confusão Final - {melhor_nome}')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        os.remove("confusion_matrix.png")
        
        # Salvamento oficial e inclusão automática no Model Registry do MLflow
        mlflow.sklearn.log_model(
            sk_model=modelo_final,
            name="model",
            registered_model_name="modelo_sih_sus_catastrofico"
        )
        print("Modelo registrado com sucesso no Model Registry!")

    print("\n=== FASE 4: GOVERNANÇA E PROMOÇÃO PARA @PRODUCTION ===")
    client = MlflowClient()
    versoes = client.get_latest_versions("modelo_sih_sus_catastrofico")
    ultima_versao = versoes[-1].version if versoes else 1
    
    # Atualiza a tag estável para a versão mais recente gerada
    client.set_registered_model_alias(
        name="modelo_sih_sus_catastrofico",
        alias="production",
        version=str(ultima_versao)
    )
    print(f"Sucesso Absoluto! Versão {ultima_versao} promovida para o ambiente de @production!")

if __name__ == "__main__":
    executar_esteira_mlops_avancada()