"""
    This python file is responsible for printing messages for the user in colors
"""


from colorama import init, Fore, Style
    

def printError(message):
    """
    prints a message in red
    
    param message: a meesage for printing
    """
    init(convert = True)
    print(Fore.RED + message) 
    Style.RESET_ALL
        
def printProcess(message):
    """
    prints a message in blue
    
    param message: a meesage for printing
    """
    init(convert = True)
    print(Fore.BLUE + message) 
    Style.RESET_ALL





