import pandas as pd
from data.utils import Utils
from data.view import View
from models.lstm import mLSTM
from util.logger import LoggerFactory as lf
import warnings

class Main:

    def __init__(self) -> None:
        warnings.filterwarnings('ignore')
        pass
    
    def run(self):
        lf(level='DEBUG')
        lf().logInfo('Iniciando a aplicação...')
        
        lf().logInfo('Carregando o dataset...')
        df = pd.read_csv('src\\data\\b3_stocks_1994_2020.csv', low_memory=False)
        lf().logInfo('Dataset carregado!')
        
        dft = Utils().getDfTicker(df, 'BBDC4')
        lf().logDebug('Dataset do ativo {} carregado, com {} linhas'.format(dft['ticker'].values[0], dft.shape[0]))
        
        input = ['close', 'volume']
        output = ['close']
        
        windowSize = 20
        predictAhead = 1
        
        lf().logDebug('Criando a janela temporal com os parametros: windowSize={}, ahead={}'.format(windowSize, predictAhead))
    
        xWindows, yWindows, xNorm, yNorm = Utils().createDataset(df=dft, windowSize=windowSize, ahead=predictAhead, xCols=input, yCol=output)

        lf().logInfo('Janela temporal criada!')
        
        lf().logDebug('xNorm.shape={}, yNorm.shape={}'.format(xNorm.shape, yNorm.shape))
        
        xTrain, xTest, yTrain, yTest = Utils().splitTrainTest([xWindows, yWindows], percent=0.85)
        xNormTrain, xNormTest, yNormTrain, yNormTest = Utils().splitTrainTest([xNorm, yNorm], percent=0.85)
        
        #breakpoint()
        lstm = mLSTM(
            xTest=xTest,
            xNormTrain=xNormTrain, 
            xNormTest=xNormTest, 
            yNormTrain=yNormTrain, 
            yNormTest=yNormTest, 
            inputs=input, 
            output=output
        )
        
        lstm.run()
        
        View(lstm).showGraph("BBDC4")
        
        
if __name__ == '__main__':
    Main().run()