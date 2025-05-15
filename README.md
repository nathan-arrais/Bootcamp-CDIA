# Projeto IA — Detecção de Defeitos em Chapas de Aço Inoxidável

Este projeto aplica técnicas de aprendizado de máquina para classificar defeitos em chapas de aço inox. O pipeline abrange:

- Pré-processamento de dados
- Otimização de hiperparâmetros
- Treinamento do modelo com XGBoost
- Geração de predições para submissão
- Disponibilização do modelo via API (FastAPI)
- Criação de Dashboard com Streamlit

## Contexto do Projeto

Este projeto foi desenvolvido como parte do desafio final do Bootcamp de Ciência de Dados e IA.
O objetivo era construir um sistema inteligente de controle de qualidade para chapas de aço inoxidável, capaz de detectar automaticamente defeitos com base em dados extraídos de imagens.  
O conjunto de dados fornecido pela empresa contém inúmeras variáveis com aspectos geométricos, estatísticos e visuais de cada defeito.

## Apresentação do Projeto

Este projeto foi conduzido com foco na clareza do processo, desde a definição do problema até os resultados finais. A seguir estão os principais aspectos documentados:

### Metodologia Utilizada

- Desenvolvimento orientado a scripts com modularização clara (preprocessamento, modelagem, previsão, API e dashboard).
- Uso de pipelines com `scikit-learn`, balanceamento com `SMOTE` e otimização com `RandomizedSearchCV`.
- O modelo foi originalmente desenvolvido em notebook, mas está devidamente modularizado em scripts, organizados nas pastas training e prediction, facilitando reuso, manutenção e integração com outros sistemas.

### Gestão de Atividades

- Organização do projeto por etapas:
  - EDA (exploração dos dados)
  - Criação de features
  - Modelagem e tuning
  - Geração de submissão
  - Deploy com FastAPI + Streamlit
- Utilização de Git/GitHub para versionamento do projeto.

### Desenvolvimento

- Análise exploratória com gráficos, contagem de classes e correlações.
- Criação de features como razões, produtos logarítmicos e marcação de bordas.
- Modelagem com XGBoost, avaliação por F1 Macro e validação cruzada.
- Exportação do modelo e encoder via `.pkl` para reuso.
- Integração com API REST e criação de dashboard interativo.

### Resultados Obtidos

- **F1 Macro Médio**: aproximadamente 0.72
- Melhora nas classes desbalanceadas com uso de SMOTE.
- Predições salvas em `submissao_final.csv` no formato exigido.
- Modelo disponível via API + dashboard com filtro por classe.

## Acesso ao Dashboard Interativo

A visualização interativa dos resultados pode ser acessada pelo seguinte link:

[Acesse o Dashboard Streamlit](https://bootcamp-cdia-8mmda4attdvthx8ybjtpkt.streamlit.app/)

*Permite upload de arquivos CSV, visualização das predições e análise exploratória com filtros por classe.*

### Aprendizados

- Boas práticas de organização em projetos de ciência de dados.
- Integração entre pré-processamento, modelagem, API e visualização.
- Importância de documentar o raciocínio técnico para diferentes públicos.

## Estrutura do Projeto

```plaintext
PROJETO_IA/
├── data/                         # Dados originais (train/test)
│   ├── bootcamp_test.csv
│   └── bootcamp_train.csv
│
├── notebooks/                    # Notebooks utilizados (EDA e testes de modelos)
│   ├── EDA_ResIA.ipynb
│   ├── Model_RandomForest.ipynb
│   └── Model_xgboost.ipynb
│
├── outputs/                      # Arquivos gerados no processo
│   ├── chapa-treino.csv
│   ├── melhores_parametros.json
│   └── submissao_final.csv
│
├── scripts/
│   ├── api/                      # API FastAPI para servir o modelo
│   │   └── main.py
│   │
│   ├── dashboard/                # Dashboard interativo com Streamlit
│   │   ├── app.py
│   │   └── image.webp
│   │
│   ├── model/                    # Modelos salvos (pipeline e encoder)
│   │   ├── modelo_xgboost.pkl
│   │   └── label_encoder.pkl
│   │
│   ├── prediction/               # Geração do CSV de submissão
│   │   └── predict_test.py
│   │
│   └── training/                 # Treinamento e pré-processamento
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── hyperparam_tuning.py
│       └── train_model.py
│
├── requirements.txt             # Dependências do projeto
└── README.md                    # Instruções do projeto
```

## Como executar a API

1. Crie um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate   # ou .venv\Scripts\activate no Windows
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute a API com:

```bash
uvicorn scripts.api.main:app --reload
```

4. Acesse a documentação interativa:

[http://localhost:8000/docs](http://localhost:8000/docs)

## Exemplo de entrada

```json
{
  "id": 0,
  "x_minimo": 154,
  "x_maximo": 169,
  "y_minimo": 260124,
  "y_maximo": 260136,
  "peso_da_placa": 100,
  "area_pixels": 75,
  "perimetro_x": 27,
  "perimetro_y": 17,
  "soma_da_luminosidade": 9948.0,
  "maximo_da_luminosidade": 143.0,
  "comprimento_do_transportador": 1364,
  "tipo_do_aço_A300": 0,
  "tipo_do_aço_A400": 1,
  "espessura_da_chapa_de_aço": 40.0,
  "temperatura": 80.8,
  "index_de_bordas": 0.22,
  "index_vazio": 0.58,
  "index_quadrado": 0.8,
  "index_externo_x": 0.01,
  "indice_de_bordas_x": 0.55,
  "indice_de_bordas_y": 0.70,
  "indice_de_variacao_x": 0.14,
  "indice_de_variacao_y": 0.14,
  "indice_global_externo": 0.0,
  "log_das_areas": 1.87,
  "log_indice_x": 1.17,
  "log_indice_y": 1.07,
  "indice_de_orientaçao": -0.2,
  "indice_de_luminosidade": 0.03,
  "sigmoide_das_areas": 0.30,
  "minimo_da_luminosidade": 125
}
```

## Retorno esperado

```json
{
  "classe_prevista": 6
}
```

## Observações

- O modelo foi treinado com XGBoost e inclui pré-processamento com features derivadas.
- A API calcula internamente essas features antes da predição.

## Autor

Nathan Arrais
