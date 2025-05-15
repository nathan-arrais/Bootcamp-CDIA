from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Caminho absoluto até o .pkl na pasta ../model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(BASE_DIR, '..', 'model', 'modelo_xgboost.pkl')

# Carrega o modelo
modelo = joblib.load(modelo_path)

# Inicializa API
app = FastAPI(title="API de Classificação de Defeitos em Chapas de Aço")

class EntradaModelo(BaseModel):
    id: float
    x_minimo: float
    x_maximo: float
    y_minimo: float
    y_maximo: float
    peso_da_placa: float
    area_pixels: float
    perimetro_x: float
    perimetro_y: float
    soma_da_luminosidade: float
    maximo_da_luminosidade: float
    comprimento_do_transportador: float
    tipo_do_aço_A300: int
    tipo_do_aço_A400: int
    espessura_da_chapa_de_aço: float
    temperatura: float
    index_de_bordas: float
    index_vazio: float
    index_quadrado: float
    index_externo_x: float
    indice_de_bordas_x: float
    indice_de_bordas_y: float
    indice_de_variacao_x: float
    indice_de_variacao_y: float
    indice_global_externo: float
    log_das_areas: float
    log_indice_x: float
    log_indice_y: float
    indice_de_orientaçao: float
    indice_de_luminosidade: float
    sigmoide_das_areas: float
    minimo_da_luminosidade: float


@app.post("/prever")
def prever_falha(entrada: EntradaModelo):
    try:
        df = pd.DataFrame([entrada.dict()])

        #Calcular features derivadas
        df['razao_area_perimetro'] = df['area_pixels'] / (df['perimetro_x'] + df['perimetro_y'] + 1)
        df['razao_luminosidade'] = df['maximo_da_luminosidade'] / (df['minimo_da_luminosidade'] + 1)
        df['indice_log_produto'] = df['log_indice_x'] * df['log_indice_y']
        df['razao_variacao'] = df['indice_de_variacao_x'] / (df['indice_de_variacao_y'] + 1)
        df['bordas_y_alto'] = (df['indice_de_bordas_y'] > 0.95).astype(float)

        pred = modelo.predict(df)
        return {"classe_prevista": int(pred[0])}

    except Exception as e:
        print("Erro:", e)
        return {"erro": str(e)}
