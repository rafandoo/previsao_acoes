
class Menu:
    
    def __init__(self, options = []) -> None:
        self.display_menu(options)
        
    def display_menu(self, options, title="Menu") -> None:
        while True:
            self.print_menu_header(title)
            for index, option in enumerate(options, start=1):
                self.print_menu_option(index, option)

            choice = self.get_valid_choice(options)
            self.execute_choice(options, choice)

    def print_menu_header(self, title) -> None:
        line = '=' * 10
        print(f"\n{line} {title} {line}")

    def print_menu_option(self, index, option) -> None:
        print(f"{index}. {option[0]}")

    def get_valid_choice(self, options):
        while True:
            choice = input("Enter your choice: ")
            if self.is_valid_choice(choice, options):
                return int(choice) - 1
            print("Invalid choice. Please try again.")

    def is_valid_choice(self, choice, options) -> bool:
        return choice.isdigit() and 1 <= int(choice) <= len(options)

    def execute_choice(self, options, choice) -> None:
        function = options[choice][1]
        if callable(function):
            function()
        else:
            self.display_menu(function)
                
    def close():
        exit()
        