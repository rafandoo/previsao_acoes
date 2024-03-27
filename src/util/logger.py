import logging
import inspect

class LoggerFactory:

    _instance = None

    def __new__(cls, level: str = 'INFO', toFile: bool = False):
        if cls._instance is None:
            cls._instance = super(LoggerFactory, cls).__new__(cls)
            cls._instance.__configure(level, toFile)
        return cls._instance

    def __configure(self, level: str, toFile: bool) -> None:
        logging.basicConfig(
            level=logging.getLevelName(level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=('pync.log' if toFile else None),
            encoding='utf-8'            
        )
    
    @staticmethod
    def logInfo(message: str, *args) -> None:
        LoggerFactory._instance.__log('INFO', LoggerFactory._instance.__formatMessage(message, *args))
        
    @staticmethod
    def logError(message: str, *args) -> None:
        LoggerFactory._instance.__log('ERROR', LoggerFactory._instance.__formatMessage(message, *args))
        
    @staticmethod
    def logWarning(message: str, *args) -> None:
        LoggerFactory._instance.__log('WARNING', LoggerFactory._instance.__formatMessage(message, *args))
        
    @staticmethod
    def logDebug(message: str, *args) -> None:
        LoggerFactory._instance.__log('DEBUG', LoggerFactory._instance.__formatMessage(message, *args))

    @staticmethod
    def logCritical(message: str, *args) -> None: 
        LoggerFactory._instance.__log('CRITICAL', LoggerFactory._instance.__formatMessage(message, *args))
    
    @staticmethod
    def __formatMessage(message: str, *args) -> str:
        try:
            return message.format(*args) if args is not None else message
        except:
            return message
        
    def __log(self, level: str, message: str) -> None:
        caller = inspect.stack()[2]
        className = caller[0].f_locals.get('self').__class__.__name__
        methodName = caller[3]
        msg = f'{className}.{methodName}: {message}'
        logging.log(logging.getLevelName(level), msg)
