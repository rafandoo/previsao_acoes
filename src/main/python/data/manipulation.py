import pandas as pd
import numpy as np
from typing import Tuple, List

class DataProcessor:

    def __init__(self):
        pass

    @staticmethod
    def get_dataframe_by_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Filtra o DataFrame para incluir apenas as linhas com o ticker especificado.

        Args:
            df (pd.DataFrame): DataFrame a ser filtrado.
            ticker (str): Ticker a ser filtrado.

        Returns:
            pd.DataFrame: DataFrame filtrado.
        """
        return df[df['ticker'] == ticker]

    @staticmethod
    def create_window(data: pd.DataFrame, size: int, ahead: int, features: List[str], target: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Cria janelas temporais do dataset para o treinamento de modelos de séries temporais.

        Args:
            data (pd.DataFrame): DataFrame contendo as colunas de features e target.
            size (int): Tamanho da janela (quantidade de observações) para usar como entrada.
            ahead (int): Número de observações à frente para previsão (quantidade de passos à frente).
            features (List[str]): Colunas de entrada para a janela temporal.
            target (str): Coluna de saída (variável que se quer prever).

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Listas de janelas de entrada (X) e saídas (y).
        """
        X, y = [], []
        
        for i in range(len(data) - size - ahead + 1):
            X_window = data[features].iloc[i:i + size].values
            y_window = data[target].iloc[i + size:i + size + ahead].values
            
            X.append(X_window)
            y.append(y_window if ahead > 1 else y_window[0])
        
        return X, y

    def normalize(self, data: list, return_scaler: bool = False) -> pd.DataFrame:
        """
        Normaliza os dados para que tenham média 0 e desvio padrão 1.

        Args:
            data (pd.DataFrame): DataFrame de entrada.
            return_scaler (bool, optional): Indica se o escalador será retornado. Default é False.

        Returns:
            pd.DataFrame: DataFrame normalizado.
        """
        means = data.mean()
        stds = data.std()
        data_scaled = (data - means) / stds
        
        if (return_scaler):
            return data_scaled, means, stds
        return data_scaled

    def denormalize(data: np.ndarray, means: np.ndarray, stds: np.ndarray) -> pd.DataFrame:
        """
        Desnormaliza os dados.

        Args:
            data (pd.DataFrame): DataFrame de entrada.

        Returns:
            pd.DataFrame: DataFrame desnormalizado.
        """
        data_denorm = pd.DataFrame(data)
        # for col in data_denorm.columns:
        #     data_denorm[col] = data_denorm[col] * data_denorm[col].std() + data_denorm[col].mean()
        data_denorm = data_denorm * stds + means
        return data_denorm

    @staticmethod
    def split_dataset(items: List[np.ndarray], ratio: float = 0.8) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Divide o conjunto de dados em treinamento e teste com base na razão fornecida.

        Args:
            items (List[np.ndarray]): Lista de itens a serem separados.
            ratio (float, optional): Proporção de dados de treinamento. Default é 0.8.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Itens de treinamento e teste.
        """
        split_index = int(len(items) * ratio)
        return items[:split_index], items[split_index:]

    def prepare_dataset(self, df: pd.DataFrame, ticker: str, size: int, ahead: int, features: List[str], target: str, train_ratio: float = 0.8) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Prepara o dataset para o treinamento do modelo, aplicando filtro por ticker, criação de janela temporal,
        normalização e divisão em treino e teste.

        Args:
            df (pd.DataFrame): DataFrame bruto de entrada.
            ticker (str): Ticker a ser filtrado.
            size (int): Tamanho da janela temporal.
            ahead (int): Número de passos à frente para previsão.
            features (List[str]): Colunas de entrada.
            target (str): Coluna de saída.
            train_ratio (float): Proporção do conjunto de treinamento. Default é 0.8.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Conjuntos de treino e teste (X_train, y_train), (X_test, y_test).
        """
        # Filtra pelo ticker desejado
        df_filtered = self.get_dataframe_by_ticker(df, ticker)
        
        # Normaliza os dados
        df_normalized, means, stds = self.normalize(df_filtered[features + [target]], return_scaler=True)

        # Cria janelas temporais
        X, y = self.create_window(df_normalized, size, ahead, features, target)
        
        # X = self.normalize(X)
        # y, y_means, y_stds = self.normalize(y, return_scaler=True)


        # Divide em conjuntos de treino e teste
        X_train, X_test = self.split_dataset(X, train_ratio)
        y_train, y_test = self.split_dataset(y, train_ratio)

        # Converte para numpy arrays para facilitar o treinamento
        return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test)), (means[target], stds[target])
