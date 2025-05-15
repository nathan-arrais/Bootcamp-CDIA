import pandas as pd

def remover_outliers_iqr(df, colunas):
    
    # Remove outliers com base no IQR das colunas fornecidas.

    df_filtrado = df.copy()
    for col in colunas:
        q1 = df_filtrado[col].quantile(0.25)
        q3 = df_filtrado[col].quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr
        df_filtrado = df_filtrado[
            (df_filtrado[col] >= limite_inf) & (df_filtrado[col] <= limite_sup)
        ]
    return df_filtrado

def remover_outliers_por_classe(df, colunas, classe_col):

    # Aplica a remoção de outliers separadamente para cada classe.

    df_filtrado = pd.DataFrame()
    for classe in df[classe_col].unique():
        df_classe = df[df[classe_col] == classe]
        df_classe_filtrado = remover_outliers_iqr(df_classe, colunas)
        df_filtrado = pd.concat([df_filtrado, df_classe_filtrado])
    return df_filtrado
