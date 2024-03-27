from models.iModels import InterfaceModels
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Dropout
from data.utils import Utils
from util.logger import LoggerFactory as lf

class mLSTM(InterfaceModels):
    
    def __init__(self, xTest, xNormTrain, xNormTest ,yNormTrain, yNormTest, inputs, output, batchSize = 64, epochs = 10) -> None:
        self.lstm = keras.Sequential()
        self.__xTest = xTest
        self.__xNormTrain = xNormTrain
        self.__yNormTrain = yNormTrain
        self.__xNormTest = xNormTest
        self.__yNormTest = yNormTest
        self.__inputs = inputs
        self.__output = output
        self.__batchSize = batchSize
        self.__epochs = epochs
        self.outTest = None
        self.outPreds = None
    
    def createModel(self):
        lf().logInfo('Criando o modelo LSTM...')
        self.lstm.add(
            LSTM(
                units=50, 
                return_sequences=True, 
                input_shape=(self.__xNormTrain.shape[1], len(self.__inputs))
            )
        )
        self.lstm.add(Dropout(0.2))
        self.lstm.add(
            LSTM(
                units=50,
                return_sequences=True
            )
        )
        self.lstm.add(Dropout(0.2))
        self.lstm.add(
            LSTM(
                units=50
            )
        )
        self.lstm.add(Dropout(0.2))
        self.lstm.add(
            Dense(
                units=1
            )
        )
        
        self.lstm.compile(optimizer='adam', loss='mean_squared_error')
        lf().logInfo('Modelo LSTM criado!')
        
    def sumary(self):
        self.lstm.summary()
        
    def fit(self):
        lf().logInfo('Treinando o modelo LSTM...')
        self.lstm.fit(
            self.__xNormTrain,
            self.__yNormTrain,
            validation_data=(self.__xNormTest, self.__yNormTest),
            batch_size=self.__batchSize,
            epochs=self.__epochs
        )
        lf().logInfo('Modelo LSTM treinado!')
    
    def predict(self):
        self.createModel()
        self.sumary()
        self.fit()
        
        normPreds = self.lstm.predict(self.__xNormTest)
        self.outPreds = Utils().denormPreds(self.__xTest, normPreds, self.__output)
        
        self.outTest = Utils().denormPreds(self.__xTest, self.__yNormTest, self.__output)

    def run(self):
        self.predict()