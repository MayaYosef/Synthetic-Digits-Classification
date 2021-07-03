"""
this python file is responsible for predicting an image
"""


from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import os
import cv2

import printmessage


class ImagePrediction():
    
    def __init__(self, model_path, labels_path):
        """
        constructs a new 'ImagePrediction' object
        
        param model_path: the directory of the trained model
        param labels_path: the directory of the labels file
        """
        
        self.__model_path = model_path
        self.__labels_path = labels_path
        self.__image_path = ""
        self.__model = None
        self.__label_binarizer = None
        self.__IMAGE_SIZE = (50,50)
        
        
    def handle_classify(self, image_path):
        """
        a public method that maneges the classifition. It is responsible for
        calling the rest of the methods in the correct order for the final image_copy.

        param image_path: the directory for the image that the user chooses to return.
        The method initializes its attribute, image_path in this parameter.
        """
        
        self.__load_Model_And_label_binarizer()
        self.__image_path = image_path
        
        image_array, image_copy = self.__prepare_Image()
        
        self.__predict_Image(image_array, image_copy)


    def __load_Model_And_label_binarizer(self):
        """
        a private method that loads the saved model i.e. the trained convolutional neural network
        and the prediction_label binarizer.
        The method initializes the class attributes
        """
        
        printmessage.printProcess("INFO: Loading network...")
        
        self.__model = load_model(self.__model_path)
        self.__label_binarizer = pickle.loads(open(self.__labels_path, "rb").read())
    
    
    def __prepare_Image(self):
        """
       a private method that adjusts the saved image in the image's paths to run on the model, just like before the training.
       Resizes the image to a uniform size I set (50, 50), converts the image from RGB to GrayScale, then converts the image to an array,
       reduces its pixel range from [0,255] to [0,1]. Also returns a copy of the image as received as input.
       
       return:
       1. the image numpy array (after fitting the image colors, dims, pixels scale... ) 
       2. copy of the image as gray scale
        """
        
        printmessage.printProcess("INFO: Loading image...")

        #loading the image
        image = cv2.imread(self.__image_path)
        image_copy = image.copy()
        
        #pre-process the image for classification
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, self.__IMAGE_SIZE)
        image = image.astype("float") / 255.0 ##
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image, image_copy


    def __predict_Image(self, image_arr, image_copy):
        """
        a private method that is responsible for performing the prediction itself. The method performs predict for the image on the loaded model and prints:
        1. Its prediction
        2. The probabilitiesbility that a model correctly identified the image according to its calculations
        3. Is the identification correct or not, by comparing the prediction_label of the image that is in the file name with 
        the prediction_label identified by the model (the prediction_label indicates the category to which the image belongs).
        In addition, the method creates and displays an image of the predictive image on which 1, 2, 3 appear.
        
        param image_arr: the image for prediction as array
        param image_copy: a copy of the image
        """
        
        #classify the input image
        printmessage.printProcess("INFO: Classifying image...")
        
        probabilities = self.__model.predict(image_arr)[0]
        index_most_likely = np.argmax(probabilities)
        prediction_label = self.__label_binarizer.classes_[index_most_likely]
        
        #print "correct" if the input image prediction_label is fit to the prediction prediction_label
        #else print "incorrect"
        filename = os.path.basename(self.__image_path)
        is_correct = filename.startswith(str(prediction_label))
        is_correct_txt = "correct" if is_correct else "incorrect"
        
        #bulid the prediction_label and draw the prediction_label on the image
        prediction_label_for_user = "{}: {:.2f}% ({})".format(prediction_label, probabilities[index_most_likely] * 100, is_correct_txt)        
        image_copy = imutils.resize(image_copy, width=400)
        text_color = (0, 255, 0) if is_correct else (0,0,255)
        cv2.putText(image_copy, prediction_label_for_user, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,	0.7, text_color, 2)
        
        #show the image_copy image
        printmessage.printProcess("INFO: {}".format(prediction_label_for_user))
        cv2.imshow("image_copy", image_copy)
        cv2.waitKey(0)    

