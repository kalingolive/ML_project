import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
import pickle
import basic
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
data = pd.read_csv("list.csv")
y = np.array(data)
_, axes2 = plt.subplots(1,1)


#imsizex=80
#imsizey=80
n_trainimg=y.shape[0]
#lists = np.empty([n_trainimg,imsizex,imsizey])
classes=np.empty([n_trainimg],dtype=str)


tests = np.array([basic.adjustim("upload/one.jpg"),basic.adjustim("upload/two.jpg"),basic.adjustim("upload/three.jpg"),basic.adjustim("upload/four.jpg"),basic.adjustim("upload/two.jpg")])
tests2 = tests.reshape(tests.shape[0],-1)
actual= np.array(["1","2","3","4","1"])

axes2.set_axis_off()
image2 = basic.adjustim("upload/test.jpg")
axes2.imshow(image2)

basic.trainme()
pickle_in = open("readfile2.pickle","rb")
classifier = pickle.load(pickle_in)
#axes2[0][0].hist((1.2*np.log(tests[1])))


predicted = classifier.predict(tests2)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(actual, predicted)))
disp = metrics.confusion_matrix(actual, predicted )
print("__________________________________")
for i in range(len(predicted)):
    print("__________________________________")
    print("\t\tpred \t\t", predicted[i])

    print("\t\tActual \t\t",actual[i])
    print("__________________________________")
plt.show()
