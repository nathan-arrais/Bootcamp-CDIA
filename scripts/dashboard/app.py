
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(page_title="Dashboard - Detecção de Defeitos", layout="wide")

# Carrega modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'modelo_xgboost.pkl')
modelo = joblib.load(MODEL_PATH)

image_path = os.path.join(BASE_DIR, 'image.webp')
image = Image.open(image_path)
st.image(image, use_container_width=True)

st.title("Dashboard - Detecção de Defeitos em Chapas de Aço")

# Upload
st.sidebar.header("Upload do Arquivo CSV")
arquivo = st.sidebar.file_uploader("Selecione o arquivo .csv com os dados", type=["csv"])

if arquivo:
    df = pd.read_csv(arquivo)
    st.success("Arquivo carregado com sucesso!")

    # Calcular features derivadas
    df['razao_area_perimetro'] = df['area_pixels'] / (df['perimetro_x'] + df['perimetro_y'] + 1)
    df['razao_luminosidade'] = df['maximo_da_luminosidade'] / (df['minimo_da_luminosidade'] + 1)
    df['indice_log_produto'] = df['log_indice_x'] * df['log_indice_y']
    df['razao_variacao'] = df['indice_de_variacao_x'] / (df['indice_de_variacao_y'] + 1)
    df['bordas_y_alto'] = (df['indice_de_bordas_y'] > 0.95).astype(float)

    # Previsão
    previsoes = modelo.predict(df)
    df['classe_prevista'] = previsoes

    st.subheader("Tabela com Previsões")

    # Filtro de classe prevista
    classes_disponiveis = sorted(df['classe_prevista'].unique())
    if 'classe_prevista' in df.columns:
        classes_disponiveis = sorted(df['classe_prevista'].unique())
        classe_filtro = st.multiselect("Filtrar por classe prevista:", classes_disponiveis, default=classes_disponiveis)
        df_filtrado = df[df['classe_prevista'].isin(classe_filtro)]
        st.dataframe(df_filtrado[['id', 'classe_prevista']])
    else:
        st.warning("Nenhuma classe prevista encontrada para aplicar filtro.")

    # Aba de análise
    st.subheader("Análise Exploratória")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Distribuição das Classes Previstas")
        fig1, ax1 = plt.subplots()
        df['classe_prevista'].value_counts().sort_index().plot(kind='bar', ax=ax1)
        ax1.set_xlabel("Classe")
        ax1.set_ylabel("Contagem")
        st.pyplot(fig1)

    with col2:
        st.markdown("### Correlação entre Variáveis")
        corr = df.select_dtypes(include='number').corr()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

    # Download
    csv_output = df.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar CSV com Previsões", csv_output, "previsoes.csv", "text/csv")

else:
    st.info("Faça upload de um arquivo CSV no menu lateral para começar.")