# -*- coding: utf-8 -*-

"""
Created on Sun Jun 16 22:36:39 2019
@author: Gustavo Suto
"""
# %%
from numpy import dtype
import numpy as np
import pandas as pd


def breve_descricao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função breve_descricao
    Objetivo: Exclui atributos que estejam com colunas com todos os valores
    'NaN'. Imprime na tela a quantidade de atributos/campos e a quantidade
    de registros.

    Args:
        df ([pandas.DataFrame]): [Dataframe que queremos analisar.]
    """
    df_mod = df.copy()

    df_mod.dropna(axis=1, how="all", inplace=True)

    print(f"""O data set possui: \n- {df_mod.shape[1]} atributos/campos; e \n- {df_mod.shape[0]} registros.\n""")
    serie_nulos(df)


def serie_nulos(df, corte: float = 0.5):
    """
    Função serie_nulos
    Responsável: Suto
    Data: 04/05/19
    Objetivo: essa função retorna uma tupla com:
        (1) contendo uma pd.series com os atributos com maior proporção de nulos; e
        (2) uma string indicando quantos atributos estão com uma proporção de nulos acima do corte dado.

    Args:
        df ([pandas.DataFrame]): [Dataframe que queremos analisar.]
        corte (int, optional): [Limite mínimo de nulos presentes em um atributo para destacar]. Default = 50.
    Returns:
        [pandas.DataFrame]: [DataFrame contendo os atributos que possuem uma proporção de nulos acima ]
    """
    serie = (df.isnull().sum().sort_values(ascending=False) / len(df))
    serie_cortada = serie[serie > corte]
    print(f"{len(serie_cortada)} atributos/features/campos possuem mais de {corte} de valores nulos.")
    return serie_cortada


def cardinalidade(df):
    """
    responsável: suto
    data: 27/10/19
    objetivo:   essa função retorna um dataframe com os atributos não
    numéricos e sua respectiva cardinalidade em ordem crescente.
    Argumentos: somente 01 (um) argumento, o DataFrame que se deseja
    analisar.
    """
    df_temporario = df.copy()
    dct_cardialidade = {}

    for coluna in df_temporario.columns:

        if dtype(df_temporario[coluna]) not in [float, 'float32', 'float64']:
            df_temporario.loc[df_temporario[coluna].isna(), coluna] = 'NaN'
            proporcao_nulos = len(df_temporario.loc[df_temporario[coluna] == 'NaN']) / len(df_temporario)
            dct_cardialidade[coluna] = {
                "Atributo": coluna,
                "DType": dtype(df_temporario[coluna]),
                "Cardinalidade": len(df_temporario[coluna].unique()),
                "Valores": sorted(df_temporario[coluna].unique()),
                "Proporção Nulos": proporcao_nulos
            }

        else:
            df_temporario.loc[df_temporario[coluna].isna(), coluna] = np.nan
            proporcao_nulos = len(df_temporario.loc[df_temporario[coluna].isna()]) / len(df_temporario)

            if (df_temporario[coluna].std() == 0):
                valores = sorted(df_temporario[coluna].unique())
            else:
                valores = [df_temporario[coluna].min(), df_temporario[coluna].max()]

            dct_cardialidade[coluna] = {
                "Atributo": coluna,
                "DType": dtype(df_temporario[coluna]),
                "Cardinalidade": len(sorted(df_temporario[coluna].unique())),  # cardinalidade_n,
                "Valores": valores,
                "Proporção Nulos": proporcao_nulos
            }

    df_cardialidade = pd.DataFrame.from_dict(dct_cardialidade, orient='index')

    return df_cardialidade


def check_for_equal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """This function looks for equal columns within a pd.DataFrame.

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_equal_columns = pd.DataFrame(columns=df.columns.tolist(), index=df.columns.tolist())

    for column in df.columns:
        for line in df.columns:
            if df[column].equals(df[line]):
                df_equal_columns.loc[line, column] = 1
            else:
                df_equal_columns.loc[line, column] = 0

    df_equal_columns = df_equal_columns.loc[
        df_equal_columns.sum(axis=0) > 1,
        df_equal_columns.sum(axis=1) > 1
    ]

    return df_equal_columns


# TODO: remover essa função e colocar isolada numa outra classe.
def r2_ajustado(x, y, y_pred):
    """
    responsável: Suto
    data: 23/11/19
    r2_ajustado retorna o R² Ajustado e recebe como argumento as séries com
    o valor alvo teste e o predito.
    """
    from sklearn.metrics import r2_score

    n = x.shape[0]
    k = x.shape[1]
    return (1 - ((n - 1) / (n - (k + 1))) * (1 - r2_score(y, y_pred)))


if __name__ == '__main__':
    df = pd.DataFrame({
        'a': [1, 2, 3], 'b': ['a', 'b', 'c'], 'c': [1.23, 0.987, 123.5],
        'd': [0.001, 0.001, 0.001]
        })

    breve_descricao(df)

    display(cardinalidade(df))
    
# %%
