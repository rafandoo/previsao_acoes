from models.interface import InterfaceModels
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.layers import Dense, GRU, Dropout
from typing import override
from util.logger import Logger
import numpy as np
from typing import Optional, List, Tuple
from data.manipulation import DataProcessor
from models.model_enum import ModelEnum
import time


class mGRU(InterfaceModels):
    """
    Classe que implementa o modelo GRU.

    Attributes:
        _id (str): Identificador do modelo.
        _ticker (str): Ticker do ativo.
        _time (int): Tempo de execução do modelo.
        _model (Sequential): Objeto contendo a modelagem.
        _x_norm_train (np.array): Array contendo os dados de treino normalizados.
        _y_norm_train (np.array): Array contendo os dados de treino normalizados.
        _x_norm_test (np.array): Array contendo os dados de teste normalizados.
        _y_norm_test (np.array): Array contendo os dados de teste normalizados.
        _input_size (int): Tamanho dos dados de entrada.
        _activation_functions (List[str]): Funções de ativação.
        _batch_size (int, optional): Tamanho do lote.
        _epochs (int, optional): Quantidade de épocas.
        _stopped_epoch (int): Epoca em que o treinamento foi interrompido.
        _output (list): Array contendo as previsões realizadas.
    """

    def __init__(
        self,
        id: str,
        ticker: str,
        x_norm_train: np.array,
        x_norm_test: np.array,
        y_norm_train: np.array,
        y_norm_test: np.array,
        input_size: int,
        scaler: Tuple[int, int],
        activation_functions: List[str],
        batch_size: Optional[int] = ModelEnum.BATCH_SIZE.value,
        epochs: Optional[int] = ModelEnum.EPOCHS.value
    ) -> None:
        self._id = id
        self._ticker = ticker
        self._model = Sequential(name='GRU')
        self._x_norm_train = x_norm_train
        self._x_norm_test = x_norm_test
        self._y_norm_train = y_norm_train
        self._y_norm_test = y_norm_test
        self._input_size = input_size
        self._scaler = scaler

        if (len(activation_functions) != ModelEnum.GRU_ACTIVATION_FUNCTIONS_SIZE.value):
            raise Exception('Tamanho da lista de funções de ativação inválido!')

        self._activation_functions = activation_functions
        self._batch_size = batch_size
        self._epochs = epochs
        self._stopped_epoch = 0
        self._output = None
        self._time = 0

    @override
    @property
    def model(self) -> Sequential:
        return self._model

    @override
    @property
    def id(self) -> str:
        return f'{self._id}-{self.name}'

    @override
    @property
    def name(self) -> str:
        return self.model.name
    
    @override
    @property
    def activation_functions(self) -> List[str]:
        return self._activation_functions

    @override
    @property
    def ticker(self) -> str:
        return self._ticker

    @override
    @property
    def stopped_epoch(self) -> int:
        return self._stopped_epoch

    @override
    @property
    def output(self) -> Tuple[np.array, np.array]:
        return self._output

    @override
    @property
    def time(self) -> int:
        return self._time

    @override
    def create_model(self) -> None:
        Logger.info('Iniciando a criação do modelo {}.', self.name)

        try:
            self.model.add(
                GRU(
                    units=150,
                    return_sequences=True,
                    input_shape=(self._x_norm_train.shape[1], self._input_size),
                    activation=self._activation_functions[0]
                )
            )
            self.model.add(Dropout(rate=0.3))

            self.model.add(
                GRU(
                    units=100,
                    activation=self._activation_functions[1]
                )
            )
            self.model.add(Dropout(0.2))

            self.model.add(
                Dense(
                    units=8,
                    activation=self._activation_functions[2]
                )
            )

            self.model.add(
                Dense(
                    units=1,
                    activation=self._activation_functions[3]
                )
            )

            self.model.compile(
                optimizer=ModelEnum.OPTIMIZER.value,
                loss=ModelEnum.LOSS.value
            )
            Logger.notice('Modelo {} criado com sucesso!', self.name)
        except Exception as e:
            Logger.fatal('Erro crítico ao criar o modelo {}.', self.name, e)

    @override
    def fit(self) -> None:
        start_time = time.time()
        Logger.info('Iniciando o treinamento do modelo {}.', self.name)
        Logger.debug('Batch size: {} - Epochs: {}', self._batch_size, self._epochs)

        checkpoint = super().checkpoint()
        early_stop = super().early_stop()

        try:
            history = self.model.fit(
                self._x_norm_train,
                self._y_norm_train,
                validation_data=(self._x_norm_test, self._y_norm_test),
                batch_size=self._batch_size,
                epochs=self._epochs,
                callbacks=[checkpoint, early_stop]
            )
            self._stopped_epoch = early_stop.stopped_epoch
            Logger.notice('Treinamento do modelo {} finalizado com sucesso!', self.name)
            Logger.debug('Resumo do treinamento: Última época interrompida em {}', self._stopped_epoch)
        except Exception as e:
            Logger.error('Erro durante o treinamento do modelo', self.name, e)
        finally:
            self._time = time.time() - start_time

    @override
    def predict(self) -> None:
        Logger.trace('Iniciando previsão com o modelo {}.', self.name)
        self.create_model()
        self.fit()

        try:
            predictions = self.model.predict(self._x_norm_test)
            self._output = (
                DataProcessor.denormalize(predictions, self._scaler[0], self._scaler[1]),
                DataProcessor.denormalize(self._y_norm_test, self._scaler[0], self._scaler[1])
            )
            Logger.notice('Previsão do modelo {} concluída.', self.name)

            super().export_model()
        except Exception as e:
            Logger.error('Erro ao realizar previsão com o modelo', self.name, e)
