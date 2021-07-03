"""
this python file is responsible for checking the paths
"""


import os
import string

import printmessage


def is_new_and_valid(messageHeb, messageExists, messageValid, path, hebNotValid = False):
    """
    checks if the path received is a valid path to create a new folder, that does not exist yet. 
    Prints appropriate messages.
    
    param messageHeb:  a message to print in case path contains Hebrew letters
    param messageExists: a message to print if the path already exists
    param messageValid: a message to print if the path is not valid
    param path: a directory  
    param hebNotValid: True if Hebrew letters in the path are considered an error, False else & by default
        
    return: True if the path is new and valid, false else    
        
    """
    if not __is_new_and_valid(path, messageHeb, messageExists, hebNotValid):
        return False
    
    try:
        #tries to make a folder in the current path. 
        #If it succeeded: the path is valid, return true.
        #if it If it failed: the path is not valid, return false.
        os.mkdir(path) #create
        os.rmdir(path) #remove
        return True
     
    except: ##??
        if(path != ""):
            printmessage.printError(messageValid)
            return False
    
   
def __is_new_and_valid(path, messageHeb, messageExists, hebNotValid = False):
    """
    preforms the actual checking of the path- is it new and valid.
    Prints appropriate messages.
    
    param path: a directory  
    param messageHeb: a message to print in case path contains Hebrew letters
    param messageExists: a message to print if the path already exists
    param hebNotValid: True if Hebrew letters in the path are considered an error, False else & by default
        
    return: Ture if the path is new and valid, False else.
        
    """
    if(hebNotValid):
        if not check_language(messageHeb, path):
            printmessage.printError(messageHeb)
            return False
   
    if(os.path.exists(path)):
        #check if the path is already exists. If so, returns false.
        printmessage.printError(messageExists)
        return False
    
    return True


def is_exists_and_valid(messageHeb, messageNotExists, path, hebNotValid = False):
    """
    a static and private method that checks whether the path it is receiving exists and whether the flag it is receiving is True. 
    If so, it also checks the characters that make up the address.
    The method will return True if the address is valid. Otherwise, will return False.
    
    param messageHeb: a message to print in case path contains Hebrew letters
    param messageNotExists: a message to print if the path does not exist  
    param path: a directory  
    param hebNotValid: True if Hebrew letters in the path are considered an error, False else & by default      
        
    """
    if(hebNotValid):
        if not check_language(messageHeb, path):
            return 
    
    if not os.path.exists(path):
        printmessage.printError(messageNotExists)


def check_language(messageHeb, path):
    """
    a static and private method that checks if there are hebrew letters in a path
    
    param messageHeb: a message to print in case path contains Hebrew letters
    param path: a directory     
        
    return: True if path does not contain Hebrew letters, False else      
    """
    for character in path:
        if not (character in string.printable): 
            printmessage.printError(messageHeb)
            return False
    
    return True

