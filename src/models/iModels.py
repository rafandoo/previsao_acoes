from abc import ABC, abstractmethod


class InterfaceModels(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def createModel(self) -> None:
        pass
    
    @abstractmethod
    def sumary(self) -> None:
        pass
    
    @abstractmethod
    def fit(self) -> None:
        pass
    
    @abstractmethod
    def predict(self) -> None:
        pass
    
    @abstractmethod
    def run(self) -> None:
        pass