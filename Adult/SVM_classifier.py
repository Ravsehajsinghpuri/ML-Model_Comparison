#-----------------------------------IMPORTING NECESSARY LIBRARIES AND PACKAGES------------------------------------#
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
#---------------------------------LOADING DATASET AND DATA PREPROCESSING-------------------------------------------#
dataset=pd.read_csv('adult.csv')

dataset=dataset[dataset["workclass"]!='?']
dataset=dataset[dataset["occupation"]!='?']
dataset=dataset[dataset["native.country"]!='?']
dataset.loc[dataset["workclass"]=="Without-pay","workclass"]="unemployed"

dataset.loc[dataset["workclass"]=="Self-emp-inc","workclass"]="self-employed"
dataset.loc[dataset["workclass"]=="Self-emp-not-inc","workclass"]="self-employed"
dataset.loc[dataset["workclass"]=="Local-gov","workclass"]="SL-gov"
dataset.loc[dataset["workclass"]=="State-gov","workclass"]="SL-gov"
dataset.loc[dataset["marital.status"]=="Married-civ-spouse","marital.status"]="Married"
dataset.loc[dataset["marital.status"]=="Married-AF-spouse","marital.status"]="Married"
dataset.loc[dataset["marital.status"]=="Married-spouse-absent","marital.status"]="Married"
dataset.loc[dataset["marital.status"]=="Divorced","marital.status"]="Not-Married"
dataset.loc[dataset["marital.status"]=="Separated","marital.status"]="Not-Married"
dataset.loc[dataset["marital.status"]=="Widowed","marital.status"]="Not-Married"
North_America=["United-States","Mexico","Canada","Dominican-Republic","El-Salvador","Guatemala","Haiti","Honduras","Jamaica","Puerto-Rico","Trinadad&Tobago","Outlying-US(Guam-USVI-etc)","Cuba","Nicaragua"]
Asia=["Cambodia","China","Hong","India","Iran","Japan","Laos","Philippines","Taiwan","Thailand","Vietnam"]
South_America=["Columbia","Ecuador","Peru"]
Europe=["England", "France", "Germany", "Greece", "Holand-Netherlands","Hungary", "Ireland", "Italy", "Poland", "Portugal", "Scotland","Yugoslavia"]
Other=["south"]
dataset.loc[dataset["native.country"].isin(North_America),"native.country"]="North America"
dataset.loc[dataset["native.country"].isin(Asia),"native.country"]="Asia"
dataset.loc[dataset["native.country"].isin(South_America),"native.country"]="South America"
dataset.loc[dataset["native.country"].isin(Europe),"native.country"]="Europe"
dataset.loc[dataset["native.country"].isin(Other),"native.country"]="Other"

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

dataset=MultiColumnLabelEncoder(columns=['workclass','education','marital.status','occupation','relationship','race','sex','native.country']).fit_transform(dataset)
dataset=pd.get_dummies(dataset,columns=['workclass','education','marital.status','occupation','relationship','race','sex','native.country'],drop_first=True)
columns=['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss','hours.per.week', 'workclass_1', 'workclass_2', 'workclass_3','workclass_4', 'education_1', 'education_2', 'education_3','education_4', 'education_5', 'education_6', 'education_7','education_8', 'education_9', 'education_10', 'education_11','education_12', 'education_13', 'education_14', 'education_15','marital.status_1', 'marital.status_2', 'occupation_1', 'occupation_2',
       'occupation_3', 'occupation_4', 'occupation_5', 'occupation_6','occupation_7', 'occupation_8', 'occupation_9', 'occupation_10','occupation_11', 'occupation_12', 'occupation_13', 'relationship_1','relationship_2', 'relationship_3', 'relationship_4', 'relationship_5','race_1', 'race_2', 'race_3', 'race_4', 'sex_1', 'native.country_1','native.country_2', 'native.country_3', 'native.country_4','income']
dataset=dataset[columns]

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

sc=StandardScaler()
X=sc.fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
clf=SVC(kernel="rbf",random_state=0)
clf.fit(X_train,Y_train)
Y_predict=clf.predict(X_test)

print("Accuracy of our trained model on test set is : {}".format(metrics.accuracy_score(Y_test,Y_predict)))
print("The Corresponding confusion matrix is : \n {}".format(metrics.confusion_matrix(Y_test,Y_predict)))
