import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from scipy.stats import uniform, randint
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.training.preprocessing import remover_outliers_por_classe

# Caminho base do projeto
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Carrega os dados
df = pd.read_csv(os.path.join(base_path, 'outputs', 'chapa-treino.csv'))

colunas_falha = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']
df['classe_defeito'] = df[colunas_falha].idxmax(axis=1)

for col in colunas_falha:
    df[col] = df[col].astype(int)

# Features adicionais
df['razao_area_perimetro'] = df['area_pixels'] / (df['perimetro_x'] + df['perimetro_y'] + 1)
df['razao_luminosidade'] = df['maximo_da_luminosidade'] / (df['minimo_da_luminosidade'] + 1)
df['indice_log_produto'] = df['log_indice_x'] * df['log_indice_y']
df['razao_variacao'] = df['indice_de_variacao_x'] / (df['indice_de_variacao_y'] + 1)
df['bordas_y_alto'] = (df['indice_de_bordas_y'] > 0.95).astype(int)

df_limpo = remover_outliers_por_classe(
    df,
    colunas=['perimetro_x', 'log_das_areas', 'indice_de_variacao_x', 'log_indice_y', 'log_indice_x'],
    classe_col='classe_defeito'
)

X = df_limpo.drop(columns=colunas_falha + ['classe_defeito'])
y = LabelEncoder().fit_transform(df_limpo['classe_defeito'])

numericas = X.select_dtypes(include='number').columns.tolist()
categoricas = X.select_dtypes(include='object').columns.tolist()

# Pipelines de pré-processamento
numerico = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

categorico = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessador = ColumnTransformer([
    ('num', numerico, numericas),
    ('cat', categorico, categoricas)
])

# Pipeline de modelo
pipeline = ImbPipeline(steps=[
    ('preprocessador', preprocessador),
    ('smote', SMOTE(random_state=42)),
    ('classificador', XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42))
])

# Espaço de busca de hiperparâmetros
param_dist = {
    'classificador__n_estimators': randint(150, 500),
    'classificador__max_depth': randint(4, 12),
    'classificador__learning_rate': uniform(0.01, 0.15),
    'classificador__subsample': uniform(0.7, 0.3),
    'classificador__colsample_bytree': uniform(0.6, 0.4),
    'classificador__gamma': uniform(0, 1),
    'classificador__reg_alpha': uniform(0, 1),
    'classificador__reg_lambda': uniform(0, 1)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

busca = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=75,
    scoring='f1_weighted',
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

busca.fit(X, y)

# Salva os melhores hiperparâmetros
parametros_convertidos = {
    k: float(v) if isinstance(v, np.floating) else int(v)
    for k, v in busca.best_params_.items()
}

with open(os.path.join(base_path, 'outputs', 'melhores_parametros.json'), 'w') as f:
    json.dump(parametros_convertidos, f, indent=2)

print("Melhores hiperparâmetros salvos com sucesso!")

