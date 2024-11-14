from typing import List, Tuple


class Menu:

    def __init__(self, options: List[Tuple[str, (callable|List)]]) -> None:
        self.display_menu(options)
        
    def display_menu(self, options: List[Tuple[str, (callable|List)]], title: str = "Menu") -> None:
        while True:
            self._print_menu_header(title)
            for index, option in enumerate(options, start=1):
                self._print_menu_option(index, option)

            choice = self._get_valid_choice(options)
            self._execute_choice(options, choice)

    def _print_menu_header(self, title: str) -> None:
        line = '=' * 10
        print(f"\n{line} {title} {line}")

    def _print_menu_option(self, index: int, option: Tuple[str, callable]) -> None:
        print(f"{index}. {option[0]}")

    def _get_valid_choice(self, options: List[Tuple[str, (callable|List)]]) -> None:
        while True:
            choice = input("Digite uma opção: ")
            if self._is_valid_choice(choice, options):
                return int(choice) - 1
            print("Opção inválida! Tente novamente.")

    def _is_valid_choice(self, choice: int, options: List[Tuple[str, (callable|List)]]) -> bool:
        return choice.isdigit() and 1 <= int(choice) <= len(options)

    def _execute_choice(self, options: List[Tuple[str, (callable|List)]], choice: int) -> None:
        function = options[choice][1]
        if callable(function):
            function()
        else:
            self.display_menu(function)

    def close():
        exit()
