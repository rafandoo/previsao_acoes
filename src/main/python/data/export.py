import os
import pandas as pd
from data.metrics import AssessmentMetrics
from models.interface import InterfaceModels
from data.view import ChartView
import json


class Export:
    """
    Classe responsável por realizar os processos de exportação dos dados de um
    determinado modelo de Machine Learning.

    Atributtes:
        impl_model (InterfaceModels): Modelo de ML já treinado e testado.
        assessment_metrics (AssessmentMetrics): Classe de gestão de métricas.
        chart_view (ChartView): Classe de gestão de gráficos.
        dir (str): Diretório geral.
    """

    _PATH: str = '/workspaces/previsao-acoes/src/test/python/models'

    def __init__(self, impl_model: InterfaceModels) -> None:
        self._impl_model = impl_model

        y_pred, y_test = impl_model.output
        self._assessment_metrics = AssessmentMetrics(
            y_pred=y_pred,
            y_test=y_test
        )

        self._chart_view = ChartView(
            y_test=y_test,
            y_pred=y_pred,
            id=impl_model.id
        )

        self._dir = '{}/{}'.format(self._PATH, impl_model.name)

    @staticmethod
    def generate_filename_checkpoint(name: str, id: str) -> str:
        """
        Realiza a criação do path onde será salvo o arquivo de checkpoint
        utilizado pelo callback dos modelos.

        Args:
            name (str): Nome do modelo.
            id (str): Identificador do modelo.

        Returns:
            str: Path completo do arquivo.
        """
        return f'{Export._PATH}/{name}/{id}_weights_best.keras'

    def _generate_filepath(self, operation: str, extension: str = 'keras') -> str:
        """
        Realiza a criação padronizada dos nomes dos arquivos gerados pelos modelos.

        Args:
            operation (str): Operação executada.
            extension (str, optional): Extensão do arquivo. Defaults to 'keras'.

        Returns:
            str: Path completo do arquivo
        """
        return '{0}/{1}_{2}.{3}'.format(self._dir, self._impl_model.id, operation, extension)

    def _create_dir(self) -> None:
        """
        Realiza a criação do diretório base, caso o mesmo não exista.
        """
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)

    def _summary(self) -> None:
        """
        Realiza a exportação do 'summary' do modelo.
        """
        with open(self._generate_filepath('summary', 'txt'), 'w') as f:
            self._impl_model.model.summary(
                print_fn=lambda x: f.write(x + '\n'))

    def _ml_model(self) -> None:
        """
        Salva o modelo treinado com todas as suas caracteristícas.
        """
        self._impl_model.model.save(
            filepath=self._generate_filepath('model'),
            overwrite=True,
            include_optimizer=True
        )

    def _metrics(self) -> None:
        """
        Exporta todas as métricas obtidas pelo modelo durante o processo de teste.
        """
        metrics = self._assessment_metrics.generate_metrics()
        with open(self._generate_filepath('metrics', 'json'), 'w') as f:
            f.write(json.dumps(metrics))

    def _charts(self) -> None:
        """
        Realiza a criação de gráficos com base nas previsões e valores reais.
        """
        self._chart_view.generate_line_chart(
            self._generate_filepath('line_chart', 'png')
        )
        self._chart_view.generate_scatter_chart(
            self._generate_filepath('scatter_chart', 'png')
        )
        self._chart_view.generate_residual_chart(
            self._generate_filepath('residual_chart', 'png')
        )

    def _general_metrics(self) -> None:
        """
        Realiza a inclusão geral dos dados obtidos pelo modelo em um arquivo csv.
        """
        path = f'{Export._PATH}/{self._impl_model.name}_general_metrics.csv'

        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(
                    'id,ticker,time,stopped_epoch,activation_functions,r2,adjusted_r2,rmse,mape,mae\n')

        data = pd.read_csv(path, low_memory=False)

        id = [self._impl_model.id]
        ticker = [self._impl_model.ticker]
        time = [self._impl_model.time]
        stopped_epoch = [self._impl_model.stopped_epoch]
        activation_functions = [self._impl_model.activation_functions]
        metrics = self._assessment_metrics.generate_metrics()

        data = pd.concat(
            [
                data,
                pd.DataFrame({
                    'id': id,
                    'ticker': ticker,
                    'time': time,
                    'stopped_epoch': stopped_epoch,
                    'activation_functions': activation_functions,
                    **metrics
                })
            ],
            ignore_index=True
        )
        data.to_csv(path, index=False)

    def export_all(self) -> None:
        """
        Realiza o processo de exportação completo dos dados de um modelo.
        """
        self._create_dir()
        self._summary()
        # self._ml_model()
        self._metrics()
        self._charts()
        self._general_metrics()
