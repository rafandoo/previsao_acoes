import tensorflow as tf
import pandas as pd
from models.lstm import mLSTM
from models.gru import mGRU
from models.cnn import mCNN
from util.logger import Logger
from util.menu import Menu
import warnings
from data.manipulation import DataProcessor
from typing import Tuple, List
import numpy as np
from models.model_enum import ModelEnum
import itertools


class Main:

    def __init__(self) -> None:
        warnings.filterwarnings('ignore')

        Logger.init(application='previsao-acoes', level='DEBUG')
        Logger.info('Iniciando a aplicação...')

    def _prepare_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        self._input = ['open', 'low', 'high']
        self._output = 'close'

        self._window_size = 180
        self._predict_ahead = 1
        
        self._ticker = 'ITUB4'

        Logger.info('Carregando o dataset...')
        self._df = pd.read_csv(
            '/workspaces/previsao-acoes/src/main/python/data/b3_stocks_1994_2020.csv', low_memory=False)
        Logger.info('Dataset carregado!')
        
        dataManipulation = DataProcessor()
        train, test, scaler = dataManipulation.prepare_dataset(
            df=self._df,
            ticker=self._ticker,
            size=self._window_size,
            ahead=self._predict_ahead,
            features=self._input,
            target=self._output
        )
        return train, test, scaler

    def _create_activation_functions(self, size: int) -> List[List[str]]:
        """
        Cria uma lista de combinações de funções de ativação, com o tamanho especificado.
        """
        base_activations = ModelEnum.ACTIVATION_FUNCTIONS.value
        activation_combinations = []

        for combination in itertools.product(base_activations, repeat=size):
            activation_combinations.append(list(combination))
        
        return activation_combinations

    def lstm(self):
        train, test, scaler = self._prepare_dataset()
        
        activations_combinations = self._create_activation_functions(ModelEnum.LSTM_ACTIVATION_FUNCTIONS_SIZE.value)
        
        for i, activation_functions in enumerate(activations_combinations):
            Logger.info('Treinando com funções de ativação: {}', activation_functions)

            lstm = mLSTM(
                id=f'ml-{i}',
                ticker=self._ticker,
                x_norm_train=train[0],
                x_norm_test=test[0],
                y_norm_train=train[1],
                y_norm_test=test[1],
                input_size=len(self._input),
                scaler=scaler,
                activation_functions=activation_functions
            )
            lstm.run()
        
    def gru(self):
        train, test, scaler = self._prepare_dataset()
        
        activations_combinations = self._create_activation_functions(ModelEnum.GRU_ACTIVATION_FUNCTIONS_SIZE.value)
        
        for i, activation_functions in enumerate(activations_combinations):
            Logger.info('Treinando com funções de ativação: {}', activation_functions)
        
            gru = mGRU(
                id=f'ml-{i}',
                ticker=self._ticker,
                x_norm_train=train[0],
                x_norm_test=test[0],
                y_norm_train=train[1],
                y_norm_test=test[1],
                input_size=len(self._input),
                scaler=scaler,
                activation_functions=activation_functions
            )
            gru.run()

    def cnn(self):
        train, test, scaler = self._prepare_dataset()
        
        activations_combinations = self._create_activation_functions(ModelEnum.CNN_ACTIVATION_FUNCTIONS_SIZE.value)
        
        for i, activation_functions in enumerate(activations_combinations):
            Logger.info('Treinando com funções de ativação: {}', activation_functions)

            cnn = mCNN(
                id=f'ml-{i}',
                ticker=self._ticker,
                x_norm_train=train[0],
                x_norm_test=test[0],
                y_norm_train=train[1],
                y_norm_test=test[1],
                input_size=len(self._input),
                scaler=scaler,
                activation_functions=activation_functions
            )
            cnn.run()


if __name__ == '__main__':
    Menu([
        ("teste", [
            ("aaa", 1)
        ]),
        ("LSTM", Main().lstm),
        ("GRU", Main().gru),
        ("CNN", Main().cnn)
    ])
