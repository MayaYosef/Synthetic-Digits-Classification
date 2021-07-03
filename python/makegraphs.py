"""
this python file is responsible for making and saving graphs of the training process
"""


import matplotlib
matplotlib.use( 'tkagg' )
import matplotlib.pyplot as plt
import numpy as np
import os

import printmessage
    

def graphs(H, plot_path, epoches):
    """
    responsible for making, saving and presenting the graphs of the training process
    & printing a message with the location of the plots in the computer.
    
    param plot_path: the directory in which the graphs will be saved
    param epoches: the number of epochs- The number of times the images ran on the model for learning purposes
    """
    os.mkdir(plot_path)
    plotpath1 = plot_path + r"\plot1.png"
    plotpath2 = plot_path + r"\plot2.png"
    plotpath3 = plot_path + r"\plot3.png"
    graph1(H, plotpath1, epoches)
    graph2(H, plotpath2)
    graph3(H, plotpath3)
    printmessage.printProcess("Graphs of the training process are in " + plot_path)


def graph1(H, plot_path1, epoches):
    """ 
    Creates a png image file in which it draws the learning graph of the model and showing it at the end of the training.
    (traning & validation accuracy & loss)
    
    param H: the history of the model training
    param plot_path1: the directory in which graphs 1 will be saved
    param epochs: the number of times the images ran on the model for learning purpose
    """
    plt.style.use("ggplot")
    plt.figure()
    N = epoches
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(plot_path1) #save plot to file
    plt.show()


def graph2(H, plot_path2):
    """
    Creates a png image file in which it draws the learning graph of the model and showing it after the user closed the first plot.
    (traning & validation cross entropy loss)
    
    param H: the history of the model training
    param plot_path2: the directory in which graphs 2 will be saved
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.title('Cross Entropy Loss')
    plt.plot(H.history['loss'], color='blue', label='train')
    plt.plot(H.history['val_loss'], color='orange', label='val')
    plt.legend(loc="upper left")
    plt.savefig(plot_path2)
    plt.show()
    
    
def graph3(H, plot_path3):
    """
    Creates a png image file in which it draws the learning graph of the model and showing it after the user closed the second plot.
    (traning & validation classification accuracy)
    
    param H: the history of the model training
    param plot_path3: the directory in which graphs 3 will be saved
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.title('Classification Accuracy')
    plt.plot(H.history['accuracy'], color='blue', label='train')
    plt.plot(H.history['val_accuracy'], color='orange', label='val')
    plt.legend(loc="upper left")
    plt.savefig(plot_path3)
    plt.show()    

    
    

