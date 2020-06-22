#!/usr/bin/env python
# coding: utf-8

# In[76]:


from sklearn.decomposition import PCA
import numpy as np
from pandas import DataFrame 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import manifold 
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split


# In[77]:


Y= np.loadtxt('Y.txt')


# In[78]:


# Here is the data of wavelength of mangroves data, but I don't know why I can't upload the to this file, but I can upload it on my
# own jutper notebook, so I woulad also sent the data to you, but if you want to run the code, I thik you should change the code below.


# In[79]:


Wavelength= pd.read_csv('mangroves color.csv')


# In[80]:


# in the first plot, can find noisy data in range (1800,2000), here are two thoughts to solve this, one is use LLE to reduce dimension 
# to the other is to remove data here.


# In[81]:


Mean=np.mean(Wavelength,axis=0)
Sigma=np.std(Wavelength,axis=0)
T=Sigma/Mean


# In[82]:


plt.plot(Y[0:2000],Mean[0:2000])


# In[83]:


plt.plot(Y[0:1350],T[0:1350])


# In[84]:


# As required, I try teo wavelength range here,(400,900), (400,2400)


# In[85]:


Y[50],Y[550],Y[2050]


# In[86]:


x=Wavelength.T
wavelength2=x[50:2050].T
wavelength2.shape


# In[87]:


x=Wavelength.T
wavelength1=x[50:550].T
wavelength1.shape


# In[88]:


# I try different methods to reduce the dimension here.
# There are some unlabeled data, and I want to mark them at last. 
# In my idea, I should reduce the dimension and then find the good classifier using the labeled data, and fit this best method to 
# unlabeled.
# So I should reduce the dimension for all the samples,not just the labeled data.
# I try some dimensions I should use and I showed their plots in some meetings. In this code I removed them, this is the next step 
# in my work.
# but these dimensions are ok and I have roughly compared them.


# In[89]:


# these two are linear method.


# In[90]:


pca=PCA(n_components=10)
X_PCA1=pca.fit_transform(wavelength1)
X_PCA2=pca.fit_transform(wavelength2)


# In[91]:


ICA = FastICA(n_components=15,random_state=100) 
X_ICA1=ICA.fit_transform(wavelength1)
X_ICA2=ICA.fit_transform(wavelength2)


# In[92]:


# This is nonlinear method


# In[93]:


lle=LocallyLinearEmbedding(n_components=30,n_neighbors=50)
X_lle1=lle.fit_transform(wavelength1)
X_lle2=lle.fit_transform(wavelength2)


# In[94]:


# new data, choose the labeled ones.


# In[95]:


X_Pca1_labeled=X_PCA1[0:841]
X_Ica1_labeled=X_ICA1[0:841]
X_Lle1_labeled=X_lle1[0:841]
x_n1=wavelength1[0:841]


# In[96]:


X_Pca2_labeled=X_PCA2[0:841]
X_Ica2_labeled=X_ICA2[0:841]
X_Lle2_labeled=X_lle2[0:841]
x_n2=wavelength2[0:841]


# In[97]:


# target for four class(cloor).


# In[98]:


from numpy import array


# In[99]:


Target=array([ "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M",
               "M","M","M","M","M","M","M","M","M","M","M" ])


# In[100]:


# target for three class, the mud class have many noisy data, but after comparing, this data won't affect the classification effect
# since the characteristic of these data is out standing, so I give up this idea.


# In[101]:


target=array([ "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W","W",
               "W","W","W","W","W","W","W","W","W","W",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",
               "R","R","R","R","R","R","R","R","R","R",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B"])


# In[102]:


# split train data,and test data at random


# In[103]:


xpca1_train, xpca1_test, ypca1_train, ypca1_test = train_test_split(X_Pca1_labeled,Target, test_size=0.3)
xIca1_train, xIca1_test, yIca1_train, yIca1_test = train_test_split(X_Ica1_labeled,Target, test_size=0.3)
xlle1_train, xlle1_test, ylle1_train, ylle1_test = train_test_split(X_Lle1_labeled,Target, test_size=0.3)
x1_train, x1_test, y1_train, y1_test = train_test_split(x_n1,Target, test_size=0.3)

xpca2_train, xpca2_test, ypca2_train, ypca2_test = train_test_split(X_Pca2_labeled,Target, test_size=0.3)
xIca2_train, xIca2_test, yIca2_train, yIca2_test = train_test_split(X_Ica2_labeled,Target, test_size=0.3)
xlle2_train, xlle2_test, ylle2_train, ylle2_test = train_test_split(X_Lle2_labeled,Target, test_size=0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(x_n2,Target, test_size=0.3)


# In[104]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV


# In[105]:


from sklearn.ensemble import BaggingClassifier


# In[106]:


# SVM model for two range (400,900) , (400,2400)


# In[107]:


SVM= SVC(kernel='rbf')


# In[108]:


# when choose kernel, I find linear and sigmod is bad, so I choose rbf, which is often used.
# for gussian kernel (rbf), there are C and gamma to choose
# I first limit their range, then find the good combination roughly.


# In[109]:


distributions = dict(C=np.logspace(-1,1,40),gamma=np.logspace(-1,1,40))


# In[110]:


clf = RandomizedSearchCV(SVM, distributions, random_state=42)


# In[111]:


search1 = clf.fit(xlle1_train, ylle1_train)


# In[112]:


search1.best_params_ 


# In[113]:


# Then is the effects of SVM ,this model is good and I want to explain the method and how to use this.


# In[114]:


clf1= SVC(C=7,kernel='rbf',gamma=1)


# In[115]:


clf1.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, clf1.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, clf1.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, clf1.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, clf1.predict(xpca1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca1_train, clf1.predict(xpca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca1_test, clf1.predict(xpca1_test)))


# In[116]:


Clf1= SVC(C=7,kernel='rbf',gamma=1)


# In[117]:


Clf1.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, Clf1.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, Clf1.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, Clf1.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, Clf1.predict(xpca2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca2_train, Clf1.predict(xpca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca2_test, Clf1.predict(xpca2_test)))


# In[118]:


clf3= SVC(C=6,kernel='rbf',gamma=1)
Clf3= SVC(C=6,kernel='rbf',gamma=1)


# In[119]:


clf3.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, clf3.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, clf3.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, clf3.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, clf3.predict(xIca1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca1_train, clf3.predict(xIca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca1_test, clf3.predict(xIca1_test)))


# In[120]:


Clf3.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, Clf3.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, Clf3.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, Clf3.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, Clf3.predict(xIca2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca2_train, Clf3.predict(xIca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca2_test, Clf3.predict(xIca2_test)))


# In[121]:


clf4= SVC(C=3.5,kernel='rbf',gamma=9)
Clf4= SVC(C=3.5,kernel='rbf',gamma=9)


# In[122]:


clf4.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, clf4.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, clf4.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, clf4.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, clf4.predict(xlle1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle1_train, clf4.predict(xlle1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle1_test, clf4.predict(xlle1_test)))


# In[123]:


Clf4.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, Clf4.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, Clf4.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, Clf4.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, Clf4.predict(xlle2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, Clf4.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, Clf4.predict(xlle2_test)))


# In[124]:


clf= SVC(C=1,kernel='rbf',gamma=1)
Clf= SVC(C=1,kernel='rbf',gamma=0.2)


# In[125]:


bagging = BaggingClassifier(Clf,max_samples=0.6, max_features=0.6)
bagging.fit(x2_train,y2_train)


# In[126]:


clf.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, clf.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, clf.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, clf.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, clf.predict(x1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y1_train, clf.predict(x1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y1_test, clf.predict(x1_test)))


# In[127]:


bagging.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, bagging.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, bagging.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, bagging.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, bagging.predict(x2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y2_train, bagging.predict(x2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y2_test, bagging.predict(x2_test)))


# In[128]:


Clf.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, Clf.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, Clf.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, Clf.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, Clf.predict(x2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y2_train, Clf.predict(x2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y2_test, Clf.predict(x2_test)))


# In[129]:


# Random Forest effect, too avoid overfitting, I choose set the max depth.


# In[130]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate


# In[131]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[132]:


randomforest = RandomForestClassifier(max_depth=8)


# In[133]:


randomforest.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, randomforest.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, randomforest.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, randomforest.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, randomforest.predict(xpca1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca1_train, randomforest.predict(xpca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca1_test, randomforest.predict(xpca1_test)))


# In[134]:


randomforest.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, randomforest.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, randomforest.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, randomforest.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, randomforest.predict(xpca2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca2_train, randomforest.predict(xpca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca2_test, randomforest.predict(xpca2_test)))


# In[135]:


randomforest.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, randomforest.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, randomforest.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, randomforest.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, randomforest.predict(xIca1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca1_train, randomforest.predict(xIca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca1_test, randomforest.predict(xIca1_test)))


# In[136]:


randomforest.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, randomforest.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, randomforest.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, randomforest.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, randomforest.predict(xIca2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca2_train, randomforest.predict(xIca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca2_test, randomforest.predict(xIca2_test)))


# In[137]:


randomforest.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, randomforest.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, randomforest.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, randomforest.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, randomforest.predict(xlle1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle1_train, randomforest.predict(xlle1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle1_test, randomforest.predict(xlle1_test)))


# In[138]:


randomforest.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, randomforest.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, randomforest.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, randomforest.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, randomforest.predict(xlle2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, randomforest.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, randomforest.predict(xlle2_test)))


# In[139]:


randomforest.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, randomforest.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, randomforest.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, randomforest.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, randomforest.predict(x1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y1_train, randomforest.predict(x1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y1_test, randomforest.predict(x1_test)))


# In[140]:


randomforest.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, randomforest.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, randomforest.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, randomforest.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, randomforest.predict(x2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y2_train, randomforest.predict(x2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y2_test, randomforest.predict(x2_test)))


# In[141]:


# this decision tree model, I am sure it is bad than Random forest and GBDT, so no need to consider much.


# In[142]:


from sklearn import tree
clf_tree = tree.DecisionTreeClassifier(max_depth=8)


# In[143]:


clf_tree.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, clf_tree.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, clf_tree.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, clf_tree.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, clf_tree.predict(x1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y1_train, clf_tree.predict(x1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y1_test, clf_tree.predict(x1_test)))


# In[144]:


clf_tree.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, clf_tree.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, clf_tree.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, clf_tree.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, clf_tree.predict(x2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y2_train, clf_tree.predict(x2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y2_test, clf_tree.predict(x2_test)))


# In[145]:


clf_tree.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, clf_tree.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, clf_tree.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, clf_tree.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, clf_tree.predict(xpca1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca1_train, clf_tree.predict(xpca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca1_test, clf_tree.predict(xpca1_test)))


# In[146]:


clf_tree.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, clf_tree.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, clf_tree.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, clf_tree.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, clf_tree.predict(xpca2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca2_train, clf_tree.predict(xpca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca2_test, clf_tree.predict(xpca2_test)))


# In[147]:


clf_tree.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, clf_tree.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, clf_tree.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, clf_tree.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, clf_tree.predict(xIca1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca1_train, clf_tree.predict(xIca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca1_test, clf_tree.predict(xIca1_test)))


# In[148]:


clf_tree.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, clf_tree.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, clf_tree.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, clf_tree.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, clf_tree.predict(xIca2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca2_train, clf_tree.predict(xIca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca2_test, clf_tree.predict(xIca2_test)))


# In[149]:


clf_tree.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, clf_tree.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, clf_tree.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, clf_tree.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, clf_tree.predict(xlle1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle1_train, clf_tree.predict(xlle1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle1_test, clf_tree.predict(xlle1_test)))


# In[150]:


clf_tree.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, clf_tree.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, clf_tree.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, clf_tree.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, clf_tree.predict(xlle2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, clf_tree.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, clf_tree.predict(xlle2_test)))


# In[151]:


# GBDT model, booting+decision, a method I want to explain, and compare it with Randomforest.
# I find roughly good parameter, limit max_depth to avoid overfitting (but for train data easy to reach accuracy is 1).
# I perfer to show how to fulfil GBDT since it is a little better than Random Forest(other articles and my work both find this)


# In[152]:


from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
GradientBoosting = GradientBoostingClassifier(max_depth=1)
GradientBoosting1 = GradientBoostingClassifier(max_depth=2)
GradientBoosting2 = GradientBoostingClassifier(random_state=10,max_depth=3,min_samples_split=20,min_samples_leaf=8)


# In[153]:


GradientBoosting1.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, GradientBoosting1.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, GradientBoosting1.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, GradientBoosting1.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, GradientBoosting1.predict(x1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y1_train, GradientBoosting1.predict(x1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y1_test, GradientBoosting1.predict(x1_test)))


# In[154]:


GradientBoosting.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, GradientBoosting.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, GradientBoosting.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, GradientBoosting.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, GradientBoosting.predict(x2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y2_train, GradientBoosting.predict(x2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y2_test, GradientBoosting.predict(x2_test)))


# In[155]:


GradientBoosting.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, GradientBoosting.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, GradientBoosting.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, GradientBoosting.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, GradientBoosting.predict(xpca1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca1_train, GradientBoosting.predict(xpca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca1_test, GradientBoosting.predict(xpca1_test)))


# In[156]:


GradientBoosting2.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, GradientBoosting2.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, GradientBoosting2.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, GradientBoosting2.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, GradientBoosting2.predict(xpca2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca2_train, GradientBoosting2.predict(xpca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca2_test, GradientBoosting2.predict(xpca2_test)))


# In[157]:


GradientBoosting.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, GradientBoosting.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, GradientBoosting.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, GradientBoosting.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, GradientBoosting.predict(xIca1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca1_train, GradientBoosting.predict(xIca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca1_test, GradientBoosting.predict(xIca1_test)))


# In[158]:


GradientBoosting2.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, GradientBoosting2.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, GradientBoosting2.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, GradientBoosting2.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, GradientBoosting2.predict(xIca2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca2_train, GradientBoosting2.predict(xIca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca2_test, GradientBoosting2.predict(xIca2_test)))


# In[159]:


GradientBoosting1.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, GradientBoosting1.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, GradientBoosting1.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, GradientBoosting1.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, GradientBoosting1.predict(xlle1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle1_train, GradientBoosting1.predict(xlle1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle1_test, GradientBoosting1.predict(xlle1_test)))


# In[160]:


GradientBoosting.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, GradientBoosting.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, GradientBoosting.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, GradientBoosting.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, GradientBoosting.predict(xlle2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, GradientBoosting.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, GradientBoosting.predict(xlle2_test)))


# In[161]:


# next step is to find the best parameters of GBDT.
# But I haven't do this


# In[162]:


from __future__ import print_function

import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D


# In[163]:


# Then try deep learning,DNN, the CNN is used for image and time series, not suitable for my work.


# In[164]:


# whrite=0, red=1 ,black=2, mud=3


# In[165]:


tar=array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,
          2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
          2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
          2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
          2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
          2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
          2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
          3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])


# In[166]:


import pandas as pd
T= pd.DataFrame(tar)
T.head()


# In[167]:


wavelength1[0:841].head()
Wavelength1=wavelength1[0:841]


# In[168]:


Wavelength1.dtypes
Wavelength1.iloc[:,1:4] = Wavelength1.iloc[:,1:4].astype(np.float32)
columns = Wavelength1.columns[0:500]
print(columns)
import tensorflow as tf


# In[169]:


Wavelength1.iloc[:,0:500].head()


# In[170]:


Xn_train, Xn_test, yn_train, yn_test = train_test_split(Wavelength1.iloc[:,0:500], T, test_size=0.33, random_state=42)


# In[171]:


Xn_train.shape, Xn_test.shape, yn_train.shape, yn_test.shape


# In[172]:


feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]


# In[173]:


def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}
    label = tf.constant(labels.values, shape = [labels.size,1])
    return feature_cols,label


# In[174]:


classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[20,40,20],n_classes = 4)


# In[175]:


classifier.fit(input_fn=lambda: input_fn(Xn_train,yn_train),steps = 2000)


# In[176]:


ev = classifier.evaluate(input_fn=lambda: input_fn(Xn_test,yn_test),steps=1)


# In[177]:


print(ev)


# In[178]:


wavelength2[0:841].head()
Wavelength2=wavelength2[0:841]
Wavelength2.head()


# In[179]:


Wavelength2.dtypes
Wavelength2.iloc[:,1:10] = Wavelength2.iloc[:,1:10].astype(np.float32)
columns2 = Wavelength2.columns[0:2000]
print(columns2)
import tensorflow as tf


# In[180]:


Xn2_train, Xn2_test, yn2_train, yn2_test = train_test_split(Wavelength2.iloc[:,0:2000], T, test_size=0.3, random_state=42)


# In[181]:


Xn2_train.shape, Xn2_test.shape, yn2_train.shape, yn2_test.shape


# In[182]:


feature_columns2 = [tf.contrib.layers.real_valued_column(k) for k in columns2]


# In[183]:


def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns2}
    label = tf.constant(labels.values, shape = [labels.size,1])
    return feature_cols,label


# In[184]:


classifier2 = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns2,hidden_units=[20,40,20],n_classes = 4)


# In[185]:


classifier2.fit(input_fn=lambda: input_fn(Xn2_train,yn2_train),steps = 3000)


# In[186]:


ev2 = classifier2.evaluate(input_fn=lambda: input_fn(Xn2_test,yn2_test),steps=1)


# In[187]:


print(ev2)


# In[188]:


# I try two wavelength range (400,900), (400,2400)
# give up this DNN for these reasons: 
# too difficult to find the best net, and this method for the data lack of meanings.
# no good than GBDT classifier model,SVM model and even Random Forest.


# In[189]:


# I can also use sklearn to duild Netural N etwork, but it can be difficult to tell the meaning of the net.
# also the best net is difficult to find, I can just try some and find a roughly suitable one.
# I also find for DNN, reduce dimension is a good way to improvE.


# In[190]:


from sklearn.neural_network import MLPClassifier


# In[209]:


MLP1=MLPClassifier(random_state=1,activation ="relu",solver="lbfgs",max_iter=800,beta_1=0.7,beta_2=0.7,hidden_layer_sizes=(40, ))
MLP2=MLPClassifier(random_state=1,activation ="relu",solver="lbfgs",max_iter=500,beta_1=0.7,beta_2=0.7,hidden_layer_sizes=(20, ))


# In[210]:


MLP1.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, MLP1.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, MLP1.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, MLP1.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, MLP1.predict(x1_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y1_train, MLP1.predict(x1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y1_test, MLP1.predict(x1_test)))


# In[211]:


print (MLP1.score(x1_test,y1_test))
print (MLP1.n_layers_)
print (MLP1.loss_)


# In[194]:


MLP2.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, MLP2.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, MLP2.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, MLP2.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, MLP2.predict(x2_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y2_train, MLP2.predict(x2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y2_test, MLP2.predict(x2_test)))


# In[195]:


print (MLP2.score(x2_test,y2_test))
print (MLP2.n_layers_)
print (MLP2.loss_)


# In[ ]:




