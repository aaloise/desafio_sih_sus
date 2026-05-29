import os
import pandas as pd
import numpy as np

def processar_csv_nacional_blindado():
    caminho_raw = "data/raw/sih_2024_12.csv"
    caminho_processed = "data/processed/sih_processed.csv"
    
    if not os.path.exists(caminho_raw):
        print(f"❌ Erro: O arquivo '{caminho_raw}' não foi encontrado em data/raw/.")
        return

    print("-> Analisando a estrutura e o cabeçalho do CSV...")
    
    # Lista de colunas que precisamos extrair (em letras maiúsculas para o nosso padrão)
    colunas_desejadas = ['MUNIC_RES', 'MUNIC_MOV', 'IDADE', 'SEXO', 'DIAG_PRINC', 'CAR_INT', 'INSTRU', 'GESTAO', 'VAL_TOT']
    
    # Testa os dois separadores mais comuns de arquivos do SUS (';' e ',')
    for sep in [';', ',']:
        try:
            # Lê apenas as primeiras linhas para testar o mapeamento
            df_teste = pd.read_csv(caminho_raw, nrows=3, sep=sep)
            colunas_reais = [str(c).upper().strip() for c in df_teste.columns]
            
            # Se encontrar colunas essenciais, descobrimos o separador correto!
            if 'VAL_TOT' in colunas_reais or 'IDADE' in colunas_reais:
                print(f"👉 Separador detectado com sucesso: '{sep}'")
                
                # Cria um mapeamento de 'NOME_MAIUSCULO' -> 'NomeOriginalDoArquivo'
                mapa_cases = {str(c).upper().strip(): c for c in df_teste.columns}
                
                # Isola as colunas exatas que existem no arquivo para carregar
                colunas_para_carregar = [mapa_cases[col] for col in colunas_desejadas if col in mapa_cases]
                
                print(f"-> Carregando apenas as colunas necessárias do arquivo...")
                df = pd.read_csv(caminho_raw, sep=sep, usecols=colunas_para_carregar, dtype=str)
                
                # Força todas as colunas carregadas a ficarem em letras maiúsculas para o projeto
                df.columns = [str(c).upper().strip() for c in df.columns]
                print(f"-> Base carregada com sucesso! Shape: {df.shape}")
                
                print("-> Tratando strings financeiras e gerando a variável-alvo (Target)...")
                # Remove pontos de milhar e substitui a vírgula decimal por ponto antes de converter
                valores_limpos = df['VAL_TOT'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                valores_numericos = pd.to_numeric(valores_limpos, errors='coerce')
                
                df = df.dropna(subset=['VAL_TOT'])
                
                # Define a linha de corte dos 10% dos casos mais caros (Percentil 90)
                limite_p90 = valores_numericos.quantile(0.90)
                print(f"-> Linha de corte do Custo Catastrófico (P90): R$ {limite_p90:.2f}")
                
                # Binarização do Target (1 = Catastrófico, 0 = Controlado)
                df['TARGET_CUSTO_CATASTROFICO'] = np.where(valores_numericos >= limite_p90, 1, 0)
                
                # Salva o arquivo pronto para consumo do notebook de EDA
                os.makedirs("data/processed", exist_ok=True)
                df.to_csv(caminho_processed, index=False)
                print(f"✅ Sucesso! Base nacional preparada e salva em: {caminho_processed}")
                return
                
        except Exception as e:
            continue

    print("❌ Erro crítico: Não foi possível mapear as colunas usando ',' ou ';'. Verifique o arquivo manualmente.")

if __name__ == "__main__":
    processar_csv_nacional_blindado()