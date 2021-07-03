"""
this python file is responsible for extracting the zip files
"""


from zipfile import ZipFile

import checkdirectorys
import printmessage

    
def extract_Zip_File(path, messageNotZip, messageSucceed, messageNotSucceed):
    """
    preforms the actual extracting or prints a message if the extracting faild (because the paths were not valid)
    
    param path: a path to a zip file for extracting
    param messageNotZip: a message for printing in case the path is not a directory to a zip file
    param messageSucceed: a message for printing in case the extracting process succeed 
    param messageNotSucceed: a message for printing in case the extracting process did not succeed 
    """
    path_and_extension = path.split(".")
    if path_and_extension[len(path_and_extension)-1] != "zip":
        printmessage.printError(messageNotZip)
    else:
        extracted_dir_path = path_and_extension[0]
        with ZipFile(path, 'r') as zipObj:
            #extract all the contents of the zip file in a different directory with the same name
            is_succeed = checkdirectorys.is_new_and_valid("", "", extracted_dir_path, True) 
            if is_succeed:
                zipObj.extractall(extracted_dir_path)
                printmessage.printProcess(messageSucceed)
            if not is_succeed:
                printmessage.printError(messageNotSucceed)  
                

    