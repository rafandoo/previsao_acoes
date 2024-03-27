import matplotlib.pyplot as plt
from models.iModels import InterfaceModels

class View:
    
    def __init__(self, model: InterfaceModels) -> None:
        self.__model = model
    
    def showGraph(self, ticker):
        plt.figure(figsize=(20,10))
        plt.plot(self.__model.outTest, color='orange', label=f"Valor real - {ticker}")
        plt.plot(self.__model.outPreds[:,0], color='blue', label=f"Valor previsto - {ticker}")
        plt.xlabel("tempo")
        plt.ylabel(f"{ticker} preço")
        plt.legend()
        plt.show()