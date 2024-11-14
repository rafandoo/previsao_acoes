import abc
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from models.model_enum import ModelEnum
from typing import List


class InterfaceModels(metaclass=abc.ABCMeta):
    """
    Classe abstrata para os modelos de Machine Learning.

    Attributes:
        model (Sequential): Objeto contendo a modelagem.
        id (str): Identificador único do objeto.
        name (str): Nome do modelo.
        activation_functions (List[str]): Lista de funções de ativação.
        ticker (str): Identificador único do ativo a ser previsto.
        stopped_epoch (int): Época em que o treinamento foi parado.
        output (np.array): Array de previsões.
        time (float): Tempo total do treinamento do modelo. 
    """

    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        """ 
        Abstract method for checking if a class is a subclass of InterfaceModels.

        This method is used by the abc module to determine if a class is a subclass of InterfaceModels.

        Args:
            subclass (class): The class to be checked.

        Returns:
            bool: True if the class is a subclass of InterfaceModels, False otherwise.
        """
        return (
            hasattr(subclass, 'model') and
            callable(subclass.model) and
            hasattr(subclass, 'id') and
            callable(subclass.id) and
            hasattr(subclass, 'name') and
            callable(subclass.name) and
            hasattr(subclass, 'activation_functions') and
            callable(subclass.activation_functions) and
            hasattr(subclass, 'ticker') and
            callable(subclass.ticker) and
            hasattr(subclass, 'stopped_epoch') and
            callable(subclass.stopped_epoch) and
            hasattr(subclass, 'output') and
            callable(subclass.output) and
            hasattr(subclass, 'time') and
            callable(subclass.time) and
            hasattr(subclass, 'create_model') and
            callable(subclass.create_model) and
            hasattr(subclass, 'sumary') and
            callable(subclass.sumary) and
            hasattr(subclass, 'fit') and
            callable(subclass.fit) and
            hasattr(subclass, 'predict') and
            callable(subclass.predict) or
            NotImplemented
        )

    @property
    @abc.abstractmethod
    def model(self) -> Sequential:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def activation_functions(self) -> List[str]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ticker(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def stopped_epoch(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output(self) -> np.array:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def time(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def create_model(self) -> None:
        """
        Realiza a criação do modelo com a adição dos neurônios.
        """
        raise NotImplementedError

    def export_model(self) -> None:
        """
        Realiza o processo de exportação completa do modelo, salvando a rede,
        os pesos, as métricas obtidas e o sumario.
        """
        from data.export import Export
        Export(self).export_all()

    def checkpoint(self) -> ModelCheckpoint:
        """
        Durante o treinamento, o callback ModelCheckpoint monitorará o desempenho do modelo na métrica
        de validação e salvará os pesos no arquivo especificado sempre que uma melhoria for observada. 
        Isso é útil para reter apenas os melhores pesos do modelo, economizando recursos e facilitando 
        a recuperação do melhor modelo treinado.

        Returns:
            ModelCheckpoint: Checkpoint dos pesos dos modelos.
        """
        from data.export import Export
        return ModelCheckpoint(
            filepath=Export.generate_filename_checkpoint(self.name, self.id),
            verbose=2,
            save_best_only=True
        )

    def early_stop(self) -> EarlyStopping:
        """
        Em suma, utilizar o callback Early Stopping significa que, no final de cada época, devemos 
        calcular e verificar uma métrica com base nos dados de validação. E quando esta métrica parar de
        "melhorar", terminamos o treinamento. Ou seja, caso o valor de loss, por exemplo, não obtiver 
        melhores resultados com o decorrer das épocas, o treinamento será finalizado antes do fim das épocas.

        Returns:
            EarlyStopping: Controle de parada de épocas.
        """
        return EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=1,
            patience=ModelEnum.EARLY_STOPPING_PATIENCE.value
        )

    @abc.abstractmethod
    def fit(self) -> None:
        """
        Realiza o treinamento do modelo.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self) -> None:
        """
        Realiza a previsão do modelo.
        """
        raise NotImplementedError

    def run(self) -> None:
        """
        Realiza o processo de execução completa do modelo.
        """
        self.predict()
