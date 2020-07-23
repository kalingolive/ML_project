import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import basic

from sklearn import linear_model,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
data = pd.read_csv("list.csv")
y = np.array(data)
_, axes2 = plt.subplots(1,1)
x="upload/"+y[0][0]
#image = cv.imread(x)
#axes2[1][1].imshow(image)
#plt.imshow(image)
imsizex=100
imsizey=100
n_trainimg=y.shape[0]
lists = np.empty([n_trainimg,imsizex,imsizey])
classes = np.empty([n_trainimg],dtype=str)


tests = np.array([basic.adjustim("upload/one.jpg"),basic.adjustim("upload/two.jpg"),basic.adjustim("upload/three.jpg"),basic.adjustim("upload/four.jpg"),basic.adjustim("upload/four.jpg")])
tests2 = tests.reshape(tests.shape[0],-1)
actual= np.array(["1","2","3","4","2"])

for i in range(0,y.shape[0]):
    x="upload/"+y[i][0]
    print("Learning from image: ",i)
    image= basic.adjustim(x)
    
    lists[i] = image
    classes[i] = y[i][1]
    #print(image[:,:,1])

n_samples = len(lists)
#axes2[0][1].imshow(lists[2,:,:])
data = lists.reshape(n_samples,-1)
#classifier = svm.SVC(gamma=0.001)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(data,classes)


axes2.set_axis_off()
image2 = basic.adjustim("upload/four.jpg")
axes2.imshow(image2)


# Split data into train and test subsets
#X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.2, shuffle=False)
#classifier.fit(data, classes)




#axes2[0][0].hist((1.2*np.log(tests[1])))


accuracy = model.score(tests2,actual)
print(accuracy)
#predicted = classifier.predict(tests2)
predicted = model.predict(tests2)
print("Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(actual, predicted)))
disp = metrics.confusion_matrix(actual, predicted )
print("__________________________________")
for i in range(len(predicted)):
    print("__________________________________")
    print("\t\tpred \t\t", predicted[i])

    print("\t\tActual \t\t",actual[i])
    print("__________________________________")
"""
for x in range(len(predicted)):
    print("predicted: ", predicted[x],  "Actual: ", actual[x],"\n")
    n = model.kneighbors([tests2[x]], 2, True)
    print("N: ", n)
"""
plt.show()
