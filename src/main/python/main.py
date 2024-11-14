from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
import subprocess
import time
import os


class ReloadOnChange(FileSystemEventHandler):
    """
    Observador de arquivos que reinicia o script Python quando um arquivo Python for modificado.
    """
    
    def __init__(self, script):
        self.script = script
        self.process = None
        self.start_process()

    def start_process(self):
        """
        Inicia o processo do script Python.
        """
        if self.process:
            self.process.terminate()
        self.process = subprocess.Popen(['python', self.script])

    def on_modified(self, event: FileModifiedEvent):
        """
        Executa o script Python quando um arquivo Python for modificado.

        Args:
            event (FileModifiedEvent): Evento de modificação de arquivo.
        """
        if event.src_path.endswith('.py'):
            print(f'{event.src_path} foi modificado. Reiniciando o script...')
            self.start_process()


def start_watcher(script: str):
    """
    Inicia o observador de arquivos.

    Args:
        script (str): Caminho do script Python.
    """
    event_handler = ReloadOnChange(script)
    observer = Observer()
    observer.schedule(
        event_handler, 
        path=os.path.dirname(os.path.abspath(script)), 
        recursive=True    
    )

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# Execute a função com o script Python desejado
start_watcher('src/main/python/menu.py')
