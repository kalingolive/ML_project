#abenezer
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
import pickle
data = pd.read_csv("list.csv")
y = np.array(data)
imsizex=100
imsizey=100
n_trainimg=y.shape[0]

lists = np.empty([n_trainimg,imsizex,imsizey])
classes=np.empty([n_trainimg],dtype=str)


def adjustim(x):
    img = cv.imread(x, 0)
    img = cv.medianBlur(img, 11)
    img = cv.resize(img, (100, 100), interpolation=cv.INTER_NEAREST)
    print(img.mean())
    re, image = cv.threshold(img,80, 1, cv.THRESH_BINARY)
    image = cv.adaptiveThreshold(img, 1, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,3,11)
    #print(image)
    #image = image.take(range(10, 90), axis=0).take(range(10, 90), axis=1)
    kernel = np.array([[0,0,0],[1,1,1],[0,0,0]], np.uint8)
    
    kernel2 = np.ones((3,3), np.uint8)
    image = cv.erode(image, kernel2, iterations=1)
    image = cv.dilate(image, kernel, iterations=1)

    return image


def trainme():
    for i in range(0,y.shape[0]):
        x="upload\/"+y[i][0]
        print(x)
        image= adjustim(x)

        lists[i] = image
        classes[i] = y[i][1]
        #print(image[:,:,1])

    n_samples = len(lists)
    #axes2[0][1].imshow(lists[2,:,:])
    data = lists.reshape(n_trainimg,-1)
    classifier = svm.SVC(gamma=0.001,kernel="linear", C=3)
    print(classes.shape)
    # Split data into train and test subsets
    #X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.2, shuffle=False)
    classifier.fit(data, classes)

    with open("readfile2.pickle", "wb") as f:
        pickle.dump(classifier, f)

