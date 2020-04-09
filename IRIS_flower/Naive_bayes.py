#-------------------------------------------------IMPORTING NECESSARY PACKAGES------------------------------------------------------------#
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from  sklearn.preprocessing import StandardScaler
#---------------------------------------------LOADING DATASET AND DOING DATA PREPROCESSING------------------------------------------------#
dataset=pd.read_csv("IRIS.csv")

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
labelencoder=LabelEncoder()
Y=labelencoder.fit_transform(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=GaussianNB()
model.fit(X_train,Y_train)
Y_predict=model.predict(X_test)

print("Accuracy of our trained model on test set is : {}".format(metrics.accuracy_score(Y_test,Y_predict)))
print("The Corresponding confusion matrix is : \n {}".format(metrics.confusion_matrix(Y_test,Y_predict)))

#-----------------------------LINES OF CODE BELOW THIS POINT ARE USED FOR CONSTRUCTING A COLOR MAP OF THE CLASSIFIER-----------------------------------#
#-----------------------------BUT IS USED TO PLOT ONLY 2 FEATURES, SO PICK ANY TWO FEATURES AND THEN PLOT BY SLICING X_TRAIN USED BELOW----------------#
'''
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2,model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],c = ListedColormap(('red', 'green','blue'))(i), label = j)

plt.title('Naive bayes(Train set)')
plt.xlabel('Sepal_length')
plt.ylabel('Sepal_width')
plt.legend()
plt.show()

X_set, Y_set = X_test, Y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2,model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],c = ListedColormap(('red', 'green','blue'))(i), label = j)

plt.title('Naive bayes(Test set)')
plt.xlabel('Sepal_length')
plt.ylabel('Sepal_width')
plt.legend()
plt.show()
'''
