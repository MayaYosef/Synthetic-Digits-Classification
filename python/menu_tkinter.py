"""
this python file is the main file that runs the project
"""


import tkinter as tk
from PIL import ImageTk, Image

import train_test_model
import checkdirectorys
import printmessage
import classifyimage
import extractzipfiles
import config


def main_window():
    """
    The main method that runs the whole project. creates the screen and buttons on it that the user selects. 
    defines and activates the four buttons. responsible for performing the function of each button (each function is called from it button).
    """
    #The dimensions of the buttons
    WIDTH = 200
    HEIGHT = 100
    
    main_window = tk.Tk()
    main_window.title("Maya's project menu")
    main_window.geometry("300x470") #The dimensions of the window


    main_lbl = tk.Label(main_window,
                        text="Welcome to my project menu! Please choose an option:",
                        foreground="black",
                        background="azure")
    main_lbl.pack()
    
    #Button 1- Extract Zip Files
    zip_photo = Image.open(config.Zip_Button_Image_File_Name)
    zip_photo = zip_photo.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    zip_photo = ImageTk.PhotoImage(zip_photo)
    btn_zip = tk.Button(main_window, image=zip_photo, command=extract_zip_files)
    btn_zip.pack()

    #Button 2- Train the model
    train_photo = Image.open(config.Train_Button_Image_File_Name)
    train_photo = train_photo.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    train_photo = ImageTk.PhotoImage(train_photo)
    btn_train = tk.Button(main_window, image=train_photo, command=train_model)
    btn_train.pack()

    #Button 3- Test the model
    test_photo = Image.open(config.Test_Button_Image_File_Name)
    test_photo = test_photo.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    test_photo = ImageTk.PhotoImage(test_photo)
    btn_test = tk.Button(main_window, image=test_photo, command=test_model)
    btn_test.pack()

    #Button 4- Predict an Image
    predict_photo = Image.open(config.Predict_Button_Image_File_Name)
    predict_photo = predict_photo.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    predict_photo = ImageTk.PhotoImage(predict_photo)
    btn_predict = tk.Button(main_window, image=predict_photo, command=predict_image)
    btn_predict.pack()


    main_window.mainloop()


def extract_zip_files():
    """
    responsible for extracting the zip files of the images
    """
    
    dataset_zip_path = config.Data_Set_Zip_File_Name
    data_predictions_path = config.Data_Prediction_Zip_File_Name
    extractzipfiles.extract_Zip_File(dataset_zip_path, 
                       "the dataset path is not a zip file", 
                       "Finished to extract dataset dirs", "Could not finish to extract dataset dirs")
    extractzipfiles.extract_Zip_File(data_predictions_path, "the predictions path is not a zip file",
                       "Finished to extract predictions dirs", 
                       "Could not finish to extract predictions dirs")
    
    
def train_model():
    """
    responsible for training the model
    """
    
    dataset_path = config.Data_Set_Dir_Name
    checkdirectorys.is_exists_and_valid("Error! the images path contains hebrew letters", "Error! the images path does not exist", dataset_path, True)
    
    model_path = config.Trained_Model_Dir_Name
    checkdirectorys.is_new_and_valid("", "Error! the model path is already exists", model_path, False)
    
    labels_path = config.Traiend_Model_Labels_File_Name
    checkdirectorys.is_new_and_valid("", "Error! the labels path is already exists", labels_path, False)
    
    graphs_path = config.Trained_Model_Graphs_Dir_Name
    checkdirectorys.is_new_and_valid("", "Error! the graphs path is already exists", graphs_path, False)
    
    if model_path == labels_path:
        printmessage.printError("Error! model path defined is the same as the labels path defined. file will be override")
        return
    
    if model_path == graphs_path:
        printmessage.printError("Error! model path defined is the same as the graphs path defined. file will be override")
        return
    
    if graphs_path == labels_path:
        printmessage.printError("Error! graphs path defined is the same as the labels path defined. file will be override")
        return
    
    training = train_test_model.TrainAndTest(dataset_path, model_path, labels_path, graphs_path)
    training.handle_train() 
      

def test_model():
    """
    responsible for testing the model
    """
    dataset_path = config.Data_Set_Dir_Name
    checkdirectorys.is_exists_and_valid("Error! the images path contains hebrew letters", "Error! the images path does not exist", dataset_path, True)
    
    model_path =  config.Trained_Model_Dir_Name
    checkdirectorys.is_exists_and_valid("", "Error! the model path does not exist", model_path, False)
    
    testing = train_test_model.TrainAndTest(dataset_path, model_path, "", "")
    testing.handle_test()

    
    
def predict_image():
    """
    responsible for predicting an image
    """
    model_path =  config.Trained_Model_Dir_Name
    checkdirectorys.is_exists_and_valid("", "Error! the model path does not exist", model_path, False)
    
    labels_path = config.Traiend_Model_Labels_File_Name 
    checkdirectorys.is_exists_and_valid("", "Error! the labels path does not exist", labels_path, False)
    
    image_path = config.Data_Prediction_Dir_Name+"/"+config.Current_Image_To_Predict
    checkdirectorys.is_exists_and_valid("Error! the image path contains hebrew letters", "Error! the image path does not exist", image_path, True)
    
    predicting = classifyimage.ImagePrediction(model_path, labels_path)  
    predicting.handle_classify(image_path)



if __name__ == '__main__':
    main_window()

