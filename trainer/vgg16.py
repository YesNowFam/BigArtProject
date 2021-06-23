import cv2
import numpy as np
import math as m
import sys
# for gamma function, called 
from scipy.special import gamma as tgamma
import os

from libsvm import svmutil as a
from libsvm import svm as b
from os import system
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model

model = VGG16(weights="imagenet", include_top=True)
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def compute_features(path):
    image = load_img(path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return model.predict(image)[0]

def train_model():
    
    n = 180 #int(input("n: "))
    with open("scores.txt", "r") as s:
        scores = s.readlines()
        
    with open("train.txt", "w+") as t:
        with open("scores.txt", "r") as s:
            for i in range(n):
                features = compute_features(f"images/{i}.bmp")
                
           
                score = int(scores[i].rstrip()) 
                t.write("%.6f" % score)
                print(f"images/{i}.bmp {score}")
                
                for k in range(len(features)):
                    t.write(f" {k+1}:" + ("%.6f" % features[k]))
                    
                t.write("\n")

    system("svm-scale.exe -l -1 -u 1 -s allrange train.txt > train_scale")
    system("svm-train.exe -s 3 -c 1024 -b 1 -q train_scale allmodel")



def test_measure(path):
    dis = cv2.imread(path, 1)
    if(dis is None):
        print("Wrong image path given")
        print("Exiting...")
        sys.exit(0)
    dis = cv2.cvtColor(dis, cv2.COLOR_BGR2GRAY)


    features = compute_features(path)
    y = features
    #y = [0]
    
    # pre loaded lists from C++ Module to rescale brisquefeatures vector to [-1, 1]
    #min_= [0.336999 ,0.019667 ,0.230000 ,-0.125959 ,0.000167 ,0.000616 ,0.231000 ,-0.125873 ,0.000165 ,0.000600 ,0.241000 ,-0.128814 ,0.000179 ,0.000386 ,0.243000 ,-0.133080 ,0.000182 ,0.000421 ,0.436998 ,0.016929 ,0.247000 ,-0.200231 ,0.000104 ,0.000834 ,0.257000 ,-0.200017 ,0.000112 ,0.000876 ,0.257000 ,-0.155072 ,0.000112 ,0.000356 ,0.258000 ,-0.154374 ,0.000117 ,0.000351]
    
    #max_= [9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672, 1.644021, 0.169548, 0.713132, 0.467896, 1.553016, 0.101368, 0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133, 3.639918, 0.800955, 1.096995, 0.175286, 0.755547, 0.399270, 1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209, 0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484]


    with open("allrange", "r") as allrange:
        for i, line in enumerate(allrange):
            if i > 1:
                (n1, n2, n3) = line.split(" ")
                min__ = float(n2)
                max__ = float(n3.rstrip())
                #print(min__, max__)

                if abs(max__ - min__) > 0:
                    y.append(-1 + (2.0/(max__ - min__) * (features[i] - min__)))
                else:
                    y.append(0)
                    
                if int(n1) == len(features):
                    #print(n1)
                    break


    
    """  
    # append the rescaled vector to x 
    for i in range(0, len(features)):
        min__ = min_[i]
        max__ = max_[i]
        if abs(max__ - min__) > 0:
            y.append(-1 + (2.0/(max__ - min__) * (features[i] - min__)))
        else:
            y.append(0)
    """

    # load model 
    model = a.svm_load_model("allmodel")

    # create svm node array from python list
    x, idx = b.gen_svm_nodearray(y[1:], isKernel=(model.param.kernel_type == 4))
    #print([f"{u.index} {u.value}" for u in x])
    #x[len(features)].index = -1.0 # set last index to -1 to indicate the end.
        
    # get important parameters from model
    svm_type = model.get_svm_type()
    is_prob_model = model.is_probability_model()
    nr_class = model.get_nr_class()
    
    if svm_type in (b.ONE_CLASS, b.EPSILON_SVR, b.NU_SVC):
        # here svm_type is EPSILON_SVR as it's regression problem
        nr_classifier = 1
    dec_values = (b.c_double * nr_classifier)()
    
    # calculate the quality score of the image using the model and svm_node_array
    qualityscore = b.libsvm.svm_predict_probability(model, x, dec_values)

    return qualityscore

#sys.argv[1]
#train_model()
for i in range(180):
    print (i,"Score of the given image: {}".format(test_measure(f"images\{i}.bmp"))) 
