import os
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.training.preprocessing import remover_outliers_por_classe

# Caminho base do projeto
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Leitura e preparação dos dados
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

# Remoção de outliers
df_limpo = remover_outliers_por_classe(
    df,
    colunas=['perimetro_x', 'log_das_areas', 'indice_de_variacao_x', 'log_indice_y', 'log_indice_x'],
    classe_col='classe_defeito'
)

X = df_limpo.drop(columns=colunas_falha + ['classe_defeito'])
y = LabelEncoder().fit_transform(df_limpo['classe_defeito'])

numericas = X.select_dtypes(include='number').columns.tolist()
categoricas = X.select_dtypes(include='object').columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Pipeline de preprocessamento e modelo
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

# Carrega os melhores hiperparâmetros
with open(os.path.join(base_path, 'outputs', 'melhores_parametros.json'), 'r') as f:
    melhores_params = json.load(f)

xgb_params = {k.split('__')[-1]: v for k, v in melhores_params.items()}

# Pipeline final
modelo = ImbPipeline(steps=[
    ('preprocessador', preprocessador),
    ('smote', SMOTE(random_state=42)),
    ('classificador', XGBClassifier(**xgb_params))
])

modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("Classificação com XGBoost:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(xticks_rotation=45)
import matplotlib.pyplot as plt
plt.tight_layout()
plt.show()

print("\nValidação cruzada (F1 Macro, 5 folds):")
scores = cross_val_score(modelo, X, y, scoring='f1_macro', cv=5)
print(f"F1 Macro Médio: {scores.mean():.4f} ± {scores.std():.4f}")
