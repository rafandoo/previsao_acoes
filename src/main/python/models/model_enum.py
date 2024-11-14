from enum import Enum
from typing import List

class ModelEnum(Enum):
    """
    Enumerador contendo dados e configurações para os
    modelos de Machine Learning.
    """
    
    ACTIVATION_FUNCTIONS: List[str] = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'softmax', 'swish']
    LSTM_ACTIVATION_FUNCTIONS_SIZE: int = 4
    CNN_ACTIVATION_FUNCTIONS_SIZE: int = 4
    GRU_ACTIVATION_FUNCTIONS_SIZE: int = 4
    
    EPOCHS: int = 300
    BATCH_SIZE: int = 32
    OPTIMIZER: str = 'adam'
    LOSS: str = 'huber'
    
    EARLY_STOPPING_PATIENCE: int = 25
