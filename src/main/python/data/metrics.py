import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.utils import check_consistent_length


class AssessmentMetrics:
    """
    Classe para avaliação de modelos de regressão.

    Atributtes:
        y_test (np.array): Valores reais.
        y_pred (np.array): Valores previstos.
        n_predictors (int): Número de preditores.
    """

    def __init__(self, y_test: np.array, y_pred: np.array, n_predictors: int = 1) -> None:
        self._y_test = y_test
        self._y_pred = y_pred
        self._n_predictors = n_predictors

    def _r_squared(self) -> float:
        """
        O coeficiente de determinação, também conhecido como R², é uma medida estatística 
        que indica a proporção da variância na variável dependente que é previsível a partir 
        da(s) variável(eis) independente(s). O R² varia de 0 a 1, onde 1 indica um ajuste 
        perfeito do modelo aos dados.

        Returns:
            float: R².
        """
        return r2_score(y_true=self._y_test, y_pred=self._y_pred)

    def _adjusted_r_squared(self) -> float:
        """
        O R² ajustado leva em consideração o número de preditores no modelo. Ele penaliza a 
        inclusão de variáveis irrelevantes que não contribuem significativamente para a 
        explicação da variabilidade da variável dependente.

        Returns:
            float: R² ajustado.
        """
        n = len(self._y_test)
        return 1 - (1 - self._r_squared()) * (n - 1) / (n - self._n_predictors - 1)

    def _mean_squared_error(self) -> float:
        """
        O Erro Quadrático Médio é a média dos quadrados dos erros entre os valores previstos 
        e os valores reais. Quanto menor o MSE, melhor o modelo está ajustado aos dados.

        Returns:
            float: Erro quadratico medio.
        """
        check_consistent_length(self._y_test, self._y_pred, None)
        output_errors = np.average(
            (self._y_test - self._y_pred) ** 2,
            axis=0, 
            weights=None
        )

        return np.average(output_errors, weights=None)

    def _root_mean_squared_error(self) -> float:
        """
        O RMSE é a raiz quadrada do MSE, e representa a média dos erros absolutos. Ele fornece 
        uma interpretação mais intuitiva dos erros do modelo na mesma unidade da variável de resposta.

        Returns:
            float: Raiz quadrada do erro medio.
        """
        return np.sqrt(self._mean_squared_error())

    def _mean_absolute_percentage_error(self) -> float:
        """
        O Erro Médio Percentual Absoluto é uma métrica expressa como uma porcentagem que mede 
        a precisão do modelo em relação às previsões. O MAPE calcula a média das porcentagens 
        absolutas de erro.

        Returns:
            float: Erro medio percentual absoluto.
        """
        return np.mean(np.abs((self._y_test - self._y_pred) / self._y_test)) * 100

    def _mean_absolute_error(self) -> float:
        """
        O Erro Absoluto Médio é a média dos valores absolutos das diferenças entre os valores 
        previstos e os valores reais. Ele fornece uma medida da magnitude média dos erros.

        Returns:
            float: Erro absoluto medio.
        """
        return mean_absolute_error(y_true=self._y_test, y_pred=self._y_pred)

    def generate_metrics(self) -> dict:
        """
        Realiza a exportação dos resultados do modelo para o formato de dicionário.

        Raises:
            Exception: Se os valores de y_test e y_pred não forem preenchidos.

        Returns:
            dict: Dicionário contendo os resultados do modelo.
        """
        if (self._y_test is None) or (self._y_pred is None):
            raise Exception("y_test e y_pred devem ser preenchidos.")
        return {
            'r2': round(self._r_squared(), 4),
            'adjusted_r2': round(self._adjusted_r_squared(), 4),
            'rmse': round(self._root_mean_squared_error(), 4),
            'mape': round(self._mean_absolute_percentage_error(), 4),
            'mae': round(self._mean_absolute_error(), 4)
        }
