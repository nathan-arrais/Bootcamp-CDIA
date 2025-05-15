import os
import pandas as pd
import joblib
import json

from sklearn.preprocessing import LabelEncoder
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# Caminho base
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Carrega dados de teste
df_teste = pd.read_csv(os.path.join(base_path, 'data', 'bootcamp_test.csv'))

# Cria novas features, igual ao treino
df_teste['razao_area_perimetro'] = df_teste['area_pixels'] / (df_teste['perimetro_x'] + df_teste['perimetro_y'] + 1)
df_teste['razao_luminosidade'] = df_teste['maximo_da_luminosidade'] / (df_teste['minimo_da_luminosidade'] + 1)
df_teste['indice_log_produto'] = df_teste['log_indice_x'] * df_teste['log_indice_y']
df_teste['razao_variacao'] = df_teste['indice_de_variacao_x'] / (df_teste['indice_de_variacao_y'] + 1)
df_teste['bordas_y_alto'] = (df_teste['indice_de_bordas_y'] > 0.95).astype(int)

# ID de cada amostra (para submissao)
ids = df_teste['id']
X_teste = df_teste.copy()

# Para funcionar, o mesmo LabelEncoder usado no treino precisa ser recriado:
colunas_falha = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']
classes = colunas_falha
label_encoder = LabelEncoder()
label_encoder.fit(classes)

# Reutilizar o modelo treinado 
from scripts.training.train_model import modelo

y_pred_label = modelo.predict(X_teste)
y_pred_classe = label_encoder.inverse_transform(y_pred_label)

# Cria DataFrame de submissao
df_resultado = pd.DataFrame(0, index=range(len(y_pred_classe)), columns=colunas_falha)
for i, classe in enumerate(y_pred_classe):
    df_resultado.loc[i, classe] = 1

# Adiciona coluna ID
df_resultado.insert(0, 'id', ids)

# Exporta o CSV
saida_path = os.path.join(base_path, 'outputs', 'submissao_final.csv')
df_resultado.to_csv(saida_path, index=False)
print(f"Arquivo de submissão gerado com sucesso: {saida_path}")
