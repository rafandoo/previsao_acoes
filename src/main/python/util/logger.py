import logging
import locale
import os
import re
import unicodedata
from datetime import datetime
import threading
import traceback

class LoggerLevel:
    """
    Classe personalizada para os niveis de log.
    """
    
    FATAL = 50
    ERROR = 40
    WARN = 30
    NOTICE = 25
    INFO = 20
    DEBUG = 10
    TRACE = 5
    
    level_names = {
        FATAL: "FATAL",
        ERROR: "ERROR",
        WARN: "WARN",
        NOTICE: "NOTICE",
        INFO: "INFO",
        DEBUG: "DEBUG",
        TRACE: "TRACE"
    }
    
    logging.addLevelName(FATAL, "FATAL")
    logging.addLevelName(ERROR, "ERROR")
    logging.addLevelName(WARN, "WARN")
    logging.addLevelName(NOTICE, "NOTICE")
    logging.addLevelName(INFO, "INFO")
    logging.addLevelName(DEBUG, "DEBUG")
    logging.addLevelName(TRACE, "TRACE")
    
class CustomFormatter(logging.Formatter):
    """
    Custom formatter para formatar o log de acordo com o padrão.

    Args:
        logging (logging): Logging a ser formatado.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Formata o log de acordo com o padrão.

        Args:
            record (logging.LogRecord): Record a ser formatado.

        Returns:
            str: Log formatado.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        level_name = f"{record.levelname:<6}"

        thread_name = threading.current_thread().name

        message = record.getMessage()

        log_message = f"{timestamp} {level_name} ({thread_name}) {message}"
        return log_message

class Logger:
    """
    Classe para gerenciamento de log.

    Attributes:
        application (str): Nome da aplicação.
        logger_manager (logging): Gerenciador de log.

    """
    
    application = None
    DEFAULT_LEVEL = "ALL"
    logger_manager = None

    @staticmethod
    def init_locale() -> None:
        """
        Inicializa o locale.
        """
        locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

    @staticmethod
    def close_logger() -> None:
        """
        Realiza o fechamento do gerenciador de log.
        """
        Logger.logger_manager = None

    @staticmethod
    def init(application: str, level: str = "ALL", to_file: bool = False, log_path: str = None) -> None:
        """
        Realiza o processo de inicialização do gerenciador de log.

        Args:
            application (str): Nome da aplicação.
            level (str, optional): Nivel do log. Defaults to "ALL".
            to_file (bool, optional): Flag para registrar o log em arquivo. Defaults to False.
            log_path (str, optional): Caminho do arquivo de log. Defaults to None.
        """
        try:
            if Logger.logger_manager is None:
                Logger.set_application(application)
                Logger.logger_manager = logging.getLogger(application)
                Logger._log_file(to_file, log_path)
                Logger.set_level(level)
                Logger._set_custom_formatter()
                
                if not any(isinstance(handler, logging.StreamHandler) for handler in Logger.get_logger_manager().handlers):
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(CustomFormatter())
                    Logger.get_logger_manager().addHandler(console_handler)
                    
                Logger.get_logger_manager.propagate = False
        except Exception as e:
            logging.getLogger(Logger.__name__).fatal("Falha na inicialização do gerenciador de log.", exc_info=e)

    @staticmethod
    def get_logger_manager() -> logging:
        """
        Retorna o gerenciador de log.

        Returns:    
            logging: Gerenciador de log.
        """
        if Logger.logger_manager is None:
            Logger.init("br.dev.rplus.log")
        return Logger.logger_manager

    @staticmethod
    def set_application(application: str) -> None:
        Logger.application = application

    @staticmethod
    def get_application() -> str:
        return Logger.application

    @staticmethod
    def get_level() -> int:
        return Logger.get_logger_manager().level

    @staticmethod
    def set_level(level: str) -> None:
        parsed_level = Logger._level_parser(level)
        Logger.get_logger_manager().setLevel(parsed_level)
        try:
            logging.getLogger().handlers[0].setLevel(parsed_level)
        except IndexError:
            for handler in Logger.get_logger_manager().handlers:
                handler.setLevel(parsed_level)

    @staticmethod
    def _level_parser(level: str) -> int:
        """
        Realiza o parse do nível de log.

        Args:
            level (str): Nível de log a ser "parseado".

        Returns:
            int: Nível de log.
        """
        level_map = {
            "TRACE": LoggerLevel.TRACE,
            "DEBUG": LoggerLevel.DEBUG,
            "INFO": LoggerLevel.INFO,
            "NOTICE": LoggerLevel.NOTICE,
            "WARN": LoggerLevel.WARN,
            "ERROR": LoggerLevel.ERROR,
            "FATAL": LoggerLevel.FATAL,
            "ALL": logging.NOTSET
        }
        return level_map.get(level.upper(), logging.NOTSET)

    @staticmethod
    def _set_custom_formatter() -> None:
        """
        Configura o formater personalizado.
        """
        formatter = CustomFormatter()
        for handler in Logger.get_logger_manager().handlers:
            handler.setFormatter(formatter)

    @staticmethod
    def _get_file(log_path: str) -> str:
        """
        Retorna o caminho completo do arquivo de log.

        Args:   
            log_path (str): Caminho do arquivo de log.

        Returns:
            str: Caminho completo do arquivo de log.
        """
        filename = f"{datetime.now().strftime('%Y-%m-%d')}_{Logger.get_application()}.log"
        return os.path.join(log_path, filename)

    @staticmethod
    def _log_file(to_file: bool, log_path: str) -> None:
        """
        Realiza o registro do log em arquivo.

        Args:
            to_file (bool): Flag para registrar o log em arquivo.
            log_path (str): Caminho do arquivo de log.
        """
        if to_file:
            try:
                file_handler = logging.FileHandler(Logger._get_file(log_path), mode='a', encoding='utf-8')
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                Logger.get_logger_manager().addHandler(file_handler)
            except Exception as e:
                logging.getLogger(Logger.__name__).fatal("Falha na inicialização do gerenciador de log em arquivo.", exc_info=e)

    @staticmethod
    def _process_message(message: str, *params: tuple) -> str:
        """
        Realiza o processamento da mensagem a ser registrada, com base no contexto parametrizado.

        Args:
            message (str): Mensagem a ser tratada.
            params (tuple): Parâmetros a serem formatados.

        Returns:
            str: Mensagem tratada.
        """
        if message:
            exceptions = tuple(x for x in params if isinstance(x, Exception))
            params_list = tuple(x for x in params if not isinstance(x, Exception))
            formatted_message = message.replace("<br>", os.linesep)
            if params_list:
                formatted_message = formatted_message.format(*params_list)
            if exceptions:
                sb = [" Exceptions:\n"]
                for ex in exceptions:
                    sb.append(''.join(traceback.format_exception(type(ex), ex, ex.__traceback__)))
                formatted_message += ''.join(sb)
            return Logger._remove_accents_and_special_characters(formatted_message)
        return ""

    @staticmethod
    def _remove_accents_and_special_characters(text: str) -> str:
        """
        Remove acentos e caracteres especiais.

        Args:
            text (str): Texto a ser tratado.

        Returns:
            str: Texto tratado.
        """
        if text:
            normalized = unicodedata.normalize('NFD', text)
            return re.sub(r'[^\x00-\x7F]+', '', normalized)
        return text

    @staticmethod
    def _log(level: LoggerLevel, message: str) -> None:
        """
        Registra um log.

        Args:
            level (LoggerLevel): Nível do log.
            message (str): Mensagem a ser registrada.
        """
        try:
            caller = logging.currentframe().f_back.f_back
            Logger.get_logger_manager().log(level, message, extra={'caller': caller})
        except Exception as e:
            logging.getLogger(Logger.__name__).fatal("Falha ao registrar o log.", exc_info=e)

    @staticmethod
    def trace(message: str, *params: tuple) -> None:
        Logger._log(LoggerLevel.TRACE, Logger._process_message(message, *params))

    @staticmethod
    def debug(message: str, *params: tuple) -> None:
        Logger._log(LoggerLevel.DEBUG, Logger._process_message(message, *params))

    @staticmethod
    def info(message: str, *params: tuple) -> None:
        Logger._log(LoggerLevel.INFO, Logger._process_message(message, *params))

    @staticmethod
    def notice(message: str, *params: tuple) -> None:
        Logger._log(LoggerLevel.NOTICE, Logger._process_message(message, *params))

    @staticmethod
    def warn(message: str, *params: tuple) -> None:
        Logger._log(LoggerLevel.WARN, Logger._process_message(message, *params))

    @staticmethod
    def error(message: str, *params: tuple) -> None:
        Logger._log(LoggerLevel.ERROR, Logger._process_message(message, *params))

    @staticmethod
    def fatal(message: str, *params: tuple) -> None:
        Logger._log(LoggerLevel.FATAL, Logger._process_message(message, *params))
