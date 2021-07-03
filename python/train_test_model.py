"""
this python file is responsible for the training and for the test
"""


import matplotlib
matplotlib.use("Agg")

#import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os

from keras.models import load_model

import themodel 
import printmessage
import makegraphs


class TrainAndTest():
    
    def __init__(self, dataset_path, model_path, labels_path, graphs_path):
        """
        param dataset_path: the data set directory
        param model_path :  the directory the user chose to save the trained model OR
        param labels_path: the directory the user chose to save the images labels OR
        param graphs_path: the directory that the user chose to save the graph images in
        
        Constructs a new 'TrainAndTest' object
        """
 
        self.__dataset_path = dataset_path 
        self.__model_path = model_path 
        self.__labels_path = labels_path 
        self.__graphs_path = graphs_path 
        
        self.__EPOCHS = 50  #number of epochs
        self.__LEARNING_RATE = 1e-3  #learning rate
        self.__BATCH_SIZE = 32 #batch size
        self.__IMAGE_DIMS = (50, 50, 1) #image dimensions
        
        self.__data = [] #list of all the images as arrays
        self.__labels = [] #list labels of all the images
        
    
    def handle_test(self):
        """
        a public method that manages the test process
        """
        
        #preparing the data for testing
        self.__prepare_Images()
        
        #binarize the labels (a tool for classification)
        lb = LabelBinarizer()
    
        #Linear transformation
        self.__labels = lb.fit_transform(self.__labels)

        #spliting the date for train, test and validation
        (trainX, testX, valX, trainY, testY, valY) = self.__split_data()
        
        #loading the model
        model = load_model(self.__model_path)
        
        self.__evaluate_Model(testX, testY, model, batch_size=32)
        
    
    def handle_train(self):
    
        """
        a public method that manages the train section: responsible for:
        the training process, updating the model & label paths,
        creating graphs that describe the learning of the model, and saving them in the plot folder.
        """
        
        #preparing the data for training
        self.__prepare_Images()
      
        #binarize the labels (a tool for classification)
        lb = LabelBinarizer()
   
        #Linear transformation
        self.__labels = lb.fit_transform(self.__labels)

        #spliting the date for train, test and validation
        (trainX, testX, valX, trainY, testY, valY) = self.__split_data()
        
        #train the model and get the history of the training
        training_history = self.__train(trainX, testX, valX, trainY, testY, valY, lb)
                
        #making, saving & showing the graphs
        makegraphs.graphs(training_history, self.__graphs_path, self.__EPOCHS)

    
    def __prepare_Images(self):
        """""
        a private method that prepare the images for running on the model: 
        Resizes the images to a uniform size I set (50, 50), converts the images from RGB to GrayScale, then converts the images to an array,
        reduces its pixel range from [0,255] to [0,1].
        """
        
        printmessage.printProcess("INFO: Loading images...")
        
        #making a list of the images paths arranged randomly 
        images_paths = sorted(list(paths.list_images(self.__dataset_path)))
        random.seed(42)
        random.shuffle(images_paths)

        #loop over the data set images in order to prepare the data for running on the model
        for image_path in images_paths:
            #this loop: 
            #1. loads the image as gray scale 
            #2. converts it to numpy array 
            #3. stores it in the data list
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (self.__IMAGE_DIMS[1], self.__IMAGE_DIMS[0]))
            image = img_to_array(image)
            self.__data.append(image)
            
            #extracting the class label from the image path and update the labels list
            label = image_path.split(os.path.sep)[-2]
            self.__labels.append(label)
            
        self.__data = np.array(self.__data, dtype = "float") / 255.0
        self.__labels = np.array(self.__labels)
        printmessage.printProcess("INFO data matrix: {:.2f}MB".format(self.__data.nbytes / (1024 * 1000.0)))    
        
    
    def __split_data(self):
        """        
        a private method that splits the data for 3 categories: train set (70%), test set (20%), and validation set (10%)
        
        return: the train, validation and test data after the spliting.
        """
        
        #spliting the data to: train set (80%) and test set (20%)
        (trainX, testX, trainY, testY) = train_test_split(self.__data, self.__labels, test_size=0.2, random_state=42)
        
        #spliting the train data to: train set (87.5% of 80% = 70%) and validation set (12.5% of 80% = 10%)
        (trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=0.125, random_state=42)
        
        return(trainX, testX, valX, trainY, testY, valY)
    
    
    def __evaluate_Model(self, testX, testY, model, batch_size = 32):
        """
        a method that evaluates the model on the test data using `evaluate`- Performs the testing  
        
        param testX: a list of the test images
        param testY: a list of the test images' labels
        param model: the trained model
        param batch_size: the batch size- The number of images that the model will work in parallel while learning.
        """
        
        printmessage.printProcess('\n# Evaluate on test data')
        results = model.evaluate(testX, testY, batch_size=32)
        print('test loss ' + str(results[0])  + ' , test acc ' + str(results[1]))
    
    
    def __train(self, trainX, testX, valX, trainY, testY, valY, lb):
        """
        a private method that performs the actual model training:
        This method saves the list of labels in a binary file that will be used in the model training as well as
        in predicting categories of images later.
        This method also creates a transformation from the binary file that will be used to model the model.
        This method divides the data using a method from another module.
        In addition, this method saves the weights/ model file and the binary label file so that the 
        learning can be used in other runs of the program and so we would not have to run the model again for each prediction.
        
        param trainX: a list of the train images 
        param testX: a list of the test images
        param valX: a list of the validation images
        param trainY: a list of the train images' labels
        param testY: a list of the test images' labels
        param valY: a list of the validation images' labels
        param lb: Label Binarizer
        
        return: the history of model learning.
        """
        
        #constructing the image generator for data augmentation which will occur while running
        data_augmentation = ImageDataGenerator(rotation_range=25, 
                                 width_shift_range=0.1, 
                                 height_shift_range=0.1, 
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode="nearest")
   
        #initializing the model
        printmessage.printProcess("INFO: Compiling model...")
        
        model = themodel.MyModel.buildModel(width=self.__IMAGE_DIMS[1], height=self.__IMAGE_DIMS[0],
        depth= self.__IMAGE_DIMS[2], classes=len(lb.classes_))
        
        #setting the Adam Optimization Algorithm which will be used for optimizing the model
        optimization_algorithm = Adam(lr=self.__LEARNING_RATE, decay=self.__LEARNING_RATE / self.__EPOCHS)
        
        #configing the model before training: compile the model and define the loss function, the optimizer and the metrics
        model.compile(loss="categorical_crossentropy", optimizer = optimization_algorithm, metrics=["accuracy"])
        
        #printing a model summary table 
        model.summary()
        
        #training the network
        printmessage.printProcess("INFO: Training neural network...")
        
        history = model.fit(
        data_augmentation.flow(trainX, trainY, batch_size=self.__BATCH_SIZE),
        validation_data=(valX, valY),
        steps_per_epoch = len(trainX) // self.__BATCH_SIZE,
        epochs=self.__EPOCHS, verbose=1)
              
        #saving the model
        printmessage.printProcess("INFO: Serializing neural network...")
        model.save(self.__model_path)
    
        #saving the label binarizer
        printmessage.printProcess("INFO: Serializing label binarizer...")
        file = open(self.__labels_path, "wb")
        file.write(pickle.dumps(lb))
        file.close()
        
        return history

