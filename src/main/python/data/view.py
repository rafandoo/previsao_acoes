import matplotlib.pyplot as plt
import numpy as np


class ChartView:
    """
    Classe para gestão e controle de gráficos dos resultados obtidos pelos modelos.
    
        Atributtes:
        y_test (np.array): Valores reais.
        y_pred (np.array): Valores previstos.
        id (str): Identificador do modelo.
    """
    
    def __init__(self, y_test: np.array, y_pred: np.array, id: str) -> None:
        self._y_test = y_test
        self._y_pred = y_pred
        self._id = id

    def generate_line_chart(self, filepath: str) -> None:
        """
        Realiza a criação do gráfico de linha Real x Previsto.

        Args:
            filepath (str): Local para exportar o gráfico.
        """
        plt.figure(figsize=(20,10))
        plt.plot(self._y_test, color='orange', label='Valor real')
        plt.plot(self._y_pred, color='blue', label='Valor previsto')
        plt.xlabel('Tempo')
        plt.ylabel('Valor de fechamento')
        plt.title(f'Previsão de fechamento - {self._id}')
        plt.legend()
        plt.savefig(filepath)
        plt.close()

    def generate_scatter_chart(self, filepath: str) -> None:
        """
        Realiza a criação do gráfico dispersão dos dados. 

        Args:
            filepath (str): Local para exportar o gráfico.
        """
        plt.figure(figsize=(20,10))
        plt.scatter(self._y_test, self._y_pred, s=50, alpha=0.5)
        plt.xlabel('Valor real')
        plt.ylabel('Valor previsto')
        plt.title(f'Previsão de fechamento - {self._id}')
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.savefig(filepath)
        plt.close()

    def generate_residual_chart(self, filepath: str) -> None:
        """
        Realiza a criação do gráfico de residuos dos dados.

        Args:
            filepath (str): Local para exportar o gráfico.
        """
        plt.figure(figsize=(20,10))
        plt.scatter(self._y_pred, self._y_pred - self._y_test, s=50, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.ylabel('Residuo')
        plt.title(f'Residuais - {self._id}')
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.savefig(filepath)
        plt.close()
