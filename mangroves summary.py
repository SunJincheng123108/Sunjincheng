#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# then we consider the summary of mangroves data.
# first read in the data.


# In[3]:


Y= np.loadtxt('Y.txt')
# Here is the data of wavelength of mangroves data, but I don't know why I can't upload the to this file, but I can upload it on my
# own jutper notebook, so I woulad also sent the data to you, but if you want to run the code, I thik you should change the code below.
Wavelength= pd.read_csv('mangroves color.csv')


# In[4]:


# in the first plot, can find noisy data in range (1800,2000), here are two thoughts to solve this, one is use LLE to reduce dimension 
# to the other is to remove data here.
Mean=np.mean(Wavelength,axis=0)
Sigma=np.std(Wavelength,axis=0)
T=Sigma/Mean
plt.plot(Y[0:2000],Sigma[0:2000])


# In[5]:


plt.plot(Y[0:2000],Mean[0:2000])


# In[6]:


# in fact, the data noisy is mainly for mud class


# In[7]:


Wavelength_w = Wavelength[0:220]
xw=np.std(Wavelength_w,axis=0)
xw_mean=np.mean(Wavelength_w,axis=0)

Wavelength_r = Wavelength[220:470]
xr=np.std(Wavelength_r,axis=0)
xr_mean=np.mean(Wavelength_r,axis=0)

Wavelength_b = Wavelength[470:800]
xb=np.std(Wavelength_b,axis=0)
xb_mean=np.mean(Wavelength_b,axis=0)

Wavelength_m = Wavelength[800:841]
xm=np.std(Wavelength_m,axis=0)
xm_mean=np.mean(Wavelength_m,axis=0)


# In[8]:


# the mud class is very noisy in the range of (1800nm,2000nm), let us see the variance and the mean.


# In[9]:


plt.plot(Y[0:2000],xm[0:2000])


# In[10]:


plt.plot(Y[0:2000],xm_mean[0:2000])


# In[11]:


# then also see the other class.


# In[12]:


plt.plot(Y[0:2000],xw[0:2000])
plt.plot(Y[0:2000],xr[0:2000])
plt.plot(Y[0:2000],xb[0:2000])


# In[13]:


plt.plot(Y[0:2000],xw_mean[0:2000])
plt.plot(Y[0:2000],xr_mean[0:2000])
plt.plot(Y[0:2000],xb_mean[0:2000])


# In[14]:


# just see from these plots, it seems difficult to get any useful imformation, but I can find the variance is not large.


# In[15]:


# As required, I try teo wavelength range here,(400,900), (400,2400)
Y[50],Y[550],Y[2050],Y[1450],Y[1650]


# In[16]:


# for the range (400,2400), we can remove the noisy data,then see the effect.
# in fact,there are two methods to remove the noise. 
# remove the range (1800nm,2000nm), keep four classes.
# remove the mud samples, the problem is this may loss one class.(if this method's effect is not very good, we should not use this,
# since there are four class but this can just classify three class)


# In[17]:


# remove the range (1800nm,2000nm), keep four classes.


# In[18]:


x=Wavelength.T
wavelength2=np.vstack((x[0:1400],x[1650:2050])).T
wavelength2=wavelength2[0:841]
wavelength2.shape


# In[19]:


# remove the mud samples


# In[20]:


x=Wavelength.T
wavelength3=x[50:2050].T
wavelength3=wavelength3[0:800]
wavelength3.shape


# In[21]:


# wave range (400nm,900nm)


# In[22]:


x=Wavelength.T
wavelength1=x[50:550].T
wavelength1=wavelength1[0:841]
wavelength1.shape


# In[23]:


# in my idea, if we can classifyf our classes well, we don't need to consider the data with remove 1 class.


# In[24]:


# target for four class(cloor).
from numpy import array
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
               "B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B",
               "M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M","M",
               "M","M","M","M","M","M","M","M","M","M","M" ])


# In[25]:


# the target for three classes, remove the mud class.


# In[26]:


Target=target[0:800]
Target.shape


# In[27]:


# I try different methods to reduce the dimension here. 
# In my idea, I should reduce the dimension and then find the good classifier 
# I should reduce the dimension for all the samples,not just the labeled data.
# I try some dimensions I should use and I showed their plots in some meetings. In this code I removed them, this is the next step 
# in my work.
# these dimensions reduction methods are ok and I have roughly compared them.


# In[28]:


# these two are linear method.
pca=PCA(n_components=0.99)
X_PCA1=pca.fit_transform(wavelength1)
X_PCA2=pca.fit_transform(wavelength2)
X_PCA3=pca.fit_transform(wavelength3)

ICA = FastICA(n_components=20,random_state=100) 
X_ICA1=ICA.fit_transform(wavelength1)
X_ICA2=ICA.fit_transform(wavelength2)
X_ICA3=ICA.fit_transform(wavelength3)


# In[29]:


# This is nonlinear method
lle1=LocallyLinearEmbedding(n_components=20,n_neighbors=50)
lle2=LocallyLinearEmbedding(n_components=20,n_neighbors=50)
lle3=LocallyLinearEmbedding(n_components=20,n_neighbors=50)

X_lle1=lle1.fit_transform(wavelength1)
X_lle2=lle2.fit_transform(wavelength2)
X_lle3=lle3.fit_transform(wavelength3)


# In[30]:


# split train data,and test data at random
xpca1_train, xpca1_test, ypca1_train, ypca1_test = train_test_split(X_PCA1,target, test_size=0.3)
xIca1_train, xIca1_test, yIca1_train, yIca1_test = train_test_split(X_ICA1,target, test_size=0.3)
xlle1_train, xlle1_test, ylle1_train, ylle1_test = train_test_split(X_lle1,target, test_size=0.3)
x1_train, x1_test, y1_train, y1_test = train_test_split(wavelength1,target, test_size=0.3)

xpca2_train, xpca2_test, ypca2_train, ypca2_test = train_test_split(X_PCA2,target, test_size=0.3)
xIca2_train, xIca2_test, yIca2_train, yIca2_test = train_test_split(X_ICA2,target, test_size=0.3)
xlle2_train, xlle2_test, ylle2_train, ylle2_test = train_test_split(X_lle2,target, test_size=0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(wavelength2,target, test_size=0.3)

xpca3_train, xpca3_test, ypca3_train, ypca3_test = train_test_split(X_PCA3,Target, test_size=0.3)
xIca3_train, xIca3_test, yIca3_train, yIca3_test = train_test_split(X_ICA3,Target, test_size=0.3)
xlle3_train, xlle3_test, ylle3_train, ylle3_test = train_test_split(X_lle3,Target, test_size=0.3)
x3_train, x3_test, y3_train, y3_test = train_test_split(wavelength3,Target, test_size=0.3)


# In[31]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


# In[32]:


# accuracy= predict right/ all samples.
# kappa and F1 score are used to estimate the accuracy in each class.


# In[33]:


# there are some kinds of machine learning models, like, netural network, tree model(can be improved by ensemble), svm model, naivebayes,
# knn(unsupervised machine learning, classify the unlabeled samples, not suitable for my data), logistic regression clasifier( good but
# this model should classifier the positive data),but after reduce the dimension, the data can be negative, so the logistic regression 
# classifier may not a good choice.
# other models are just the improvement based on these models, like adding boosting or bagging.
# so we just cinsider four main kinds of models: svm, decision tree, naive bayes,netural network.


# In[34]:


# first let us see the naive bayes model, from the coffee data, I find the GaussianNB is better, so I just consider this method.


# In[35]:


# for all data, three kinds


# In[36]:


gnb = GaussianNB()
gnb.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, gnb.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, gnb.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, gnb.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, gnb.predict(x1_test)))
print ('F1_train：', f1_score(y1_train, gnb.predict(x1_train),average='weighted'))
print ('F1_test：', f1_score(y1_test, gnb.predict(x1_test),average='weighted'))


# In[37]:


gnb = GaussianNB()
gnb.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, gnb.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, gnb.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, gnb.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, gnb.predict(x2_test)))
print ('F1_train：', f1_score(y2_train, gnb.predict(x2_train),average='weighted'))
print ('F1_test：', f1_score(y2_test, gnb.predict(x2_test),average='weighted'))


# In[38]:


gnb = GaussianNB()
gnb.fit(x3_train, y3_train)
print ('accuracy_train：', accuracy_score(y3_train, gnb.predict(x3_train)))
print ('accuracy_test：', accuracy_score(y3_test, gnb.predict(x3_test)))

print ('Kappa_train：', cohen_kappa_score(y3_train, gnb.predict(x3_train)))
print ('Kappa_test：', cohen_kappa_score(y3_test, gnb.predict(x3_test)))
print ('F1_train：', f1_score(y3_train, gnb.predict(x3_train),average='weighted'))
print ('F1_test：', f1_score(y3_test, gnb.predict(x3_test),average='weighted'))


# In[39]:


# consider PCA, find it not good.


# In[40]:


gnb = GaussianNB()
gnb.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, gnb.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, gnb.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, gnb.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, gnb.predict(xpca1_test)))
print ('F1_train：', f1_score(ypca1_train, gnb.predict(xpca1_train),average='weighted'))
print ('F1_test：', f1_score(ypca1_test, gnb.predict(xpca1_test),average='weighted'))


# In[41]:


gnb = GaussianNB()
gnb.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, gnb.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, gnb.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, gnb.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, gnb.predict(xpca2_test)))
print ('F1_train：', f1_score(ypca2_train, gnb.predict(xpca2_train),average='weighted'))
print ('F1_test：', f1_score(ypca2_test, gnb.predict(xpca2_test),average='weighted'))


# In[42]:


gnb = GaussianNB()
gnb.fit(xpca3_train, ypca3_train)
print ('accuracy_train：', accuracy_score(ypca3_train, gnb.predict(xpca3_train)))
print ('accuracy_test：', accuracy_score(ypca3_test, gnb.predict(xpca3_test)))

print ('Kappa_train：', cohen_kappa_score(ypca3_train, gnb.predict(xpca3_train)))
print ('Kappa_test：', cohen_kappa_score(ypca3_test, gnb.predict(xpca3_test)))
print ('F1_train：', f1_score(ypca3_train, gnb.predict(xpca3_train),average='weighted'))
print ('F1_test：', f1_score(ypca3_test, gnb.predict(xpca3_test),average='weighted'))


# In[43]:


# then using ICA method, its effect is better.


# In[44]:


gnb = GaussianNB()
gnb.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, gnb.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, gnb.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, gnb.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, gnb.predict(xIca1_test)))
print ('F1_train：', f1_score(yIca1_train, gnb.predict(xIca1_train),average='weighted'))
print ('F1_test：', f1_score(yIca1_test, gnb.predict(xIca1_test),average='weighted'))


# In[47]:


gnb = GaussianNB()
gnb.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, gnb.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, gnb.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, gnb.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, gnb.predict(xIca2_test)))
print ('F1_train：', f1_score(yIca2_train, gnb.predict(xIca2_train),average='weighted'))
print ('F1_test：', f1_score(yIca2_test, gnb.predict(xIca2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca2_train, gnb.predict(xIca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca2_test, gnb.predict(xIca2_test)))


# In[46]:


scores = cross_val_score(gnb,X_ICA2,target, cv=20)
sum(scores)/20


# In[49]:


gnb = GaussianNB()
gnb.fit(xIca3_train, yIca3_train)
print ('accuracy_train：', accuracy_score(yIca3_train, gnb.predict(xIca3_train)))
print ('accuracy_test：', accuracy_score(yIca3_test, gnb.predict(xIca3_test)))

print ('Kappa_train：', cohen_kappa_score(yIca3_train, gnb.predict(xIca3_train)))
print ('Kappa_test：', cohen_kappa_score(yIca3_test, gnb.predict(xIca3_test)))
print ('F1_train：', f1_score(yIca3_train, gnb.predict(xIca3_train),average='weighted'))
print ('F1_test：', f1_score(yIca3_test, gnb.predict(xIca3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca3_train, gnb.predict(xIca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca3_test, gnb.predict(xIca3_test)))


# In[51]:


scores = cross_val_score(gnb,X_ICA3,Target, cv=20)
sum(scores)/20


# In[52]:


# here I find for naive bayes model, the range (400,900) is not good, (400,2400) can be good.
# ICA is a good method to reduce dimension.
# then I also use the cross valid to see the effect.


# In[53]:


gnb = GaussianNB()
gnb.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, gnb.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, gnb.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, gnb.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, gnb.predict(xlle1_test)))
print ('F1_train：', f1_score(ylle1_train, gnb.predict(xlle1_train),average='weighted'))
print ('F1_test：', f1_score(ylle1_test, gnb.predict(xlle1_test),average='weighted'))


# In[55]:


gnb = GaussianNB()
gnb.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, gnb.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, gnb.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, gnb.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, gnb.predict(xlle2_test)))
print ('F1_train：', f1_score(ylle2_train, gnb.predict(xlle2_train),average='weighted'))
print ('F1_test：', f1_score(ylle2_test, gnb.predict(xlle2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, gnb.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, gnb.predict(xlle2_test)))


# In[56]:


scores = cross_val_score(gnb,X_lle2,target, cv=20)
sum(scores)/20


# In[57]:


gnb = GaussianNB()
gnb.fit(xlle3_train, ylle3_train)
print ('accuracy_train：', accuracy_score(ylle3_train, gnb.predict(xlle3_train)))
print ('accuracy_test：', accuracy_score(ylle3_test, gnb.predict(xlle3_test)))

print ('Kappa_train：', cohen_kappa_score(ylle3_train, gnb.predict(xlle3_train)))
print ('Kappa_test：', cohen_kappa_score(ylle3_test, gnb.predict(xlle3_test)))
print ('F1_train：', f1_score(ylle3_train, gnb.predict(xlle3_train),average='weighted'))
print ('F1_test：', f1_score(ylle3_test, gnb.predict(xlle3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle3_train, gnb.predict(xlle3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle3_test, gnb.predict(xlle3_test)))


# In[58]:


scores = cross_val_score(gnb,X_lle3,Target, cv=20)
sum(scores)/20


# In[59]:


# here I find ICA is a quite good method, then consider a way to improve the naive bayes model.
# we can also find this model seems not good nough, the bagging is a way to improvement overfitting, but we can find the naive bayes 
# model is not overfitting, so I think we can considr boosting, which is a method to improve the model acuracy.


# In[60]:


bdt = AdaBoostClassifier(GaussianNB(),algorithm="SAMME",n_estimators=500, learning_rate=0.6)


# In[61]:


bdt.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, bdt.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, bdt.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, bdt.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, bdt.predict(x1_test)))
print ('F1_train：', f1_score(y1_train, bdt.predict(x1_train),average='weighted'))
print ('F1_test：', f1_score(y1_test, bdt.predict(x1_test),average='weighted'))


# In[64]:


bdt.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, bdt.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, bdt.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, bdt.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, bdt.predict(x2_test)))
print ('F1_train：', f1_score(y2_train, bdt.predict(x2_train),average='weighted'))
print ('F1_test：', f1_score(y2_test, bdt.predict(x2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y2_train, bdt.predict(x2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y2_test, bdt.predict(x2_test)))


# In[66]:


# this is good but runs slow.


# In[65]:


bdt.fit(x3_train, y3_train)
print ('accuracy_train：', accuracy_score(y3_train, bdt.predict(x3_train)))
print ('accuracy_test：', accuracy_score(y3_test, bdt.predict(x3_test)))

print ('Kappa_train：', cohen_kappa_score(y3_train, bdt.predict(x3_train)))
print ('Kappa_test：', cohen_kappa_score(y3_test, bdt.predict(x3_test)))
print ('F1_train：', f1_score(y3_train, bdt.predict(x3_train),average='weighted'))
print ('F1_test：', f1_score(y3_test, bdt.predict(x3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y3_train, bdt.predict(x3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y3_test, bdt.predict(x3_test)))


# In[67]:


# here we find if we don't remove the dimension, remove the mud class is the best.


# In[68]:


bdt.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, bdt.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, bdt.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, bdt.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, bdt.predict(xpca1_test)))
print ('F1_train：', f1_score(ypca1_train, bdt.predict(xpca1_train),average='weighted'))
print ('F1_test：', f1_score(ypca1_test, bdt.predict(xpca1_test),average='weighted'))


# In[69]:


bdt.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, bdt.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, bdt.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, bdt.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, bdt.predict(xpca2_test)))
print ('F1_train：', f1_score(ypca2_train, bdt.predict(xpca2_train),average='weighted'))
print ('F1_test：', f1_score(ypca2_test, bdt.predict(xpca2_test),average='weighted'))


# In[70]:


bdt.fit(xpca3_train, ypca3_train)
print ('accuracy_train：', accuracy_score(ypca3_train, bdt.predict(xpca3_train)))
print ('accuracy_test：', accuracy_score(ypca3_test, bdt.predict(xpca3_test)))

print ('Kappa_train：', cohen_kappa_score(ypca3_train, bdt.predict(xpca3_train)))
print ('Kappa_test：', cohen_kappa_score(ypca3_test, bdt.predict(xpca3_test)))
print ('F1_train：', f1_score(ypca3_train, bdt.predict(xpca3_train),average='weighted'))
print ('F1_test：', f1_score(ypca3_test, bdt.predict(xpca3_test),average='weighted'))


# In[71]:


# if use ICA to reduce dimension, all the three conditions ar good.


# In[75]:


bdt.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, bdt.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, bdt.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, bdt.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, bdt.predict(xIca1_test)))
print ('F1_train：', f1_score(yIca1_train, bdt.predict(xIca1_train),average='weighted'))
print ('F1_test：', f1_score(yIca1_test, bdt.predict(xIca1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca1_train, bdt.predict(xIca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca1_test, bdt.predict(xIca1_test)))


# In[76]:


scores = cross_val_score(bdt,X_ICA1,target, cv=20)
sum(scores)/20


# In[85]:


Bdt = AdaBoostClassifier(GaussianNB(),algorithm="SAMME",n_estimators=40, learning_rate=0.1)


# In[91]:


Bdt.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, Bdt.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, Bdt.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, Bdt.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, Bdt.predict(xIca2_test)))
print ('F1_train：', f1_score(yIca2_train, Bdt.predict(xIca2_train),average='weighted'))
print ('F1_test：', f1_score(yIca2_test, Bdt.predict(xIca2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca2_train, Bdt.predict(xIca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca2_test, Bdt.predict(xIca2_test)))


# In[92]:


scores = cross_val_score(Bdt,X_ICA2,target, cv=20)
sum(scores)/20


# In[93]:


Bdt1 = AdaBoostClassifier(GaussianNB(),algorithm="SAMME",n_estimators=40, learning_rate=0.05)


# In[94]:


Bdt1.fit(xIca3_train, yIca3_train)
print ('accuracy_train：', accuracy_score(yIca3_train, Bdt1.predict(xIca3_train)))
print ('accuracy_test：', accuracy_score(yIca3_test, Bdt1.predict(xIca3_test)))

print ('Kappa_train：', cohen_kappa_score(yIca3_train, Bdt1.predict(xIca3_train)))
print ('Kappa_test：', cohen_kappa_score(yIca3_test, Bdt1.predict(xIca3_test)))
print ('F1_train：', f1_score(yIca3_train, Bdt1.predict(xIca3_train),average='weighted'))
print ('F1_test：', f1_score(yIca3_test, Bdt1.predict(xIca3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca3_train, Bdt1.predict(xIca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca3_test, Bdt1.predict(xIca3_test)))


# In[95]:


scores = cross_val_score(Bdt1,X_ICA3,Target, cv=20)
sum(scores)/20


# In[96]:


# now we can find all the three conditions, the ICA + boosting + naive bayes is quite good.
# range (400,900), the accuracy around 0.93.
# range (400,2400), if reduce the range (1800,2000), the accuracy is not very good, around 0.89.
# but if reduce the mud class, the accuracy can be  very good 0.95, but ther are just three classes, so if other classifier model can
# classify the 4 classes well, I think I should use other models. 


# In[97]:


bdt.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, bdt.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, bdt.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, bdt.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, bdt.predict(xlle1_test)))
print ('F1_train：', f1_score(ylle1_train, bdt.predict(xlle1_train),average='weighted'))
print ('F1_test：', f1_score(ylle1_test, bdt.predict(xlle1_test),average='weighted'))


# In[99]:


bdt.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, bdt.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, bdt.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, bdt.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, bdt.predict(xlle2_test)))
print ('F1_train：', f1_score(ylle2_train, bdt.predict(xlle2_train),average='weighted'))
print ('F1_test：', f1_score(ylle2_test, bdt.predict(xlle2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, bdt.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, bdt.predict(xlle2_test)))


# In[100]:


scores = cross_val_score(bdt,X_lle2,target, cv=20)
sum(scores)/20


# In[102]:


bdt.fit(xlle3_train, ylle3_train)
print ('accuracy_train：', accuracy_score(ylle3_train, bdt.predict(xlle3_train)))
print ('accuracy_test：', accuracy_score(ylle3_test, bdt.predict(xlle3_test)))

print ('Kappa_train：', cohen_kappa_score(ylle3_train, bdt.predict(xlle3_train)))
print ('Kappa_test：', cohen_kappa_score(ylle3_test, bdt.predict(xlle3_test)))
print ('F1_train：', f1_score(ylle3_train, bdt.predict(xlle3_train),average='weighted'))
print ('F1_test：', f1_score(ylle3_test, bdt.predict(xlle3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca3_train, bdt.predict(xIca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca3_test, bdt.predict(xIca3_test)))


# In[103]:


scores = cross_val_score(bdt,X_lle3,Target, cv=20)
sum(scores)/20


# In[104]:


# now in summary, we can find ICA/lle + boost + naive bayes can get a good effect.
# range (400,900): accuracy around 0.92
# range (400,2400) four classes: accuracy around 0.89
# range (400,2400) three classes: accuracy around 0.95

# then we see othere models


# In[105]:


# split train data,and test data at random
xpca1_train, xpca1_test, ypca1_train, ypca1_test = train_test_split(X_PCA1,target, test_size=0.3)
xIca1_train, xIca1_test, yIca1_train, yIca1_test = train_test_split(X_ICA1,target, test_size=0.3)
xlle1_train, xlle1_test, ylle1_train, ylle1_test = train_test_split(X_lle1,target, test_size=0.3)
x1_train, x1_test, y1_train, y1_test = train_test_split(wavelength1,target, test_size=0.3)

xpca2_train, xpca2_test, ypca2_train, ypca2_test = train_test_split(X_PCA2,target, test_size=0.3)
xIca2_train, xIca2_test, yIca2_train, yIca2_test = train_test_split(X_ICA2,target, test_size=0.3)
xlle2_train, xlle2_test, ylle2_train, ylle2_test = train_test_split(X_lle2,target, test_size=0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(wavelength2,target, test_size=0.3)

xpca3_train, xpca3_test, ypca3_train, ypca3_test = train_test_split(X_PCA3,Target, test_size=0.3)
xIca3_train, xIca3_test, yIca3_train, yIca3_test = train_test_split(X_ICA3,Target, test_size=0.3)
xlle3_train, xlle3_test, ylle3_train, ylle3_test = train_test_split(X_lle3,Target, test_size=0.3)
x3_train, x3_test, y3_train, y3_test = train_test_split(wavelength3,Target, test_size=0.3)


# In[106]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
# SVM model for two range (400,900) , (400,2400)
# Then is SVM model.


# In[107]:


SVM= SVC(kernel='rbf')

# when choose kernel, I find linear and sigmod is bad, so I choose rbf, which is often used.
# for gussian kernel (rbf), there are C and gamma to choose
# I first limit their range, then find the good combination roughly.

# for svm model, the kernel can be any styles, but too difficult to find a good one, we often choose gusssian kernel(but it may not best)
# then the range of gamma is also difficult to deal with, gamma = 1/(2*sigma^2), so if gamma is large, this model will be meaningless,
# (can only explain the sample itself) what I want to say is just the gamma is difficult to set.
# so this model may difficult to design.
 
# ICA not a good choice here, since the data is very small after transfermation, so the sigma should be small, the the gamma 
# can be large, but it can be difficult to set a good range(if small, effect will be bad, if large, very easy to loss meaning)
# so here we just consider two kinds of methods to reduce dimensions.


# In[108]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-1,1,40),gamma=np.logspace(-1,1,40))
clf = RandomizedSearchCV(SVM, distributions, random_state=42)
search1 = clf.fit(xlle1_train, ylle1_train)
search1.best_params_ 


# In[109]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-1,1,40),gamma=np.logspace(-1,1,40))
clf = RandomizedSearchCV(SVM, distributions, random_state=42)
search1 = clf.fit(xlle2_train, ylle2_train)
search1.best_params_ 


# In[110]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-1,1,40),gamma=np.logspace(-1,1,40))
clf = RandomizedSearchCV(SVM, distributions, random_state=42)
search1 = clf.fit(xlle3_train, ylle3_train)
search1.best_params_ 


# In[111]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-1,1,40),gamma=np.logspace(-1,1,40))
clf = RandomizedSearchCV(SVM, distributions, random_state=42)
search1 = clf.fit(x1_train, y1_train)
search1.best_params_ 


# In[112]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-1,1,40),gamma=np.logspace(-1,1,40))
clf = RandomizedSearchCV(SVM, distributions, random_state=42)
search1 = clf.fit(x2_train, y2_train)
search1.best_params_ 


# In[113]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-1,1,40),gamma=np.logspace(-1,1,40))
clf = RandomizedSearchCV(SVM, distributions, random_state=42)
search1 = clf.fit(x3_train, y3_train)
search1.best_params_ 


# In[114]:


# first see PCA/LLE+ SVM model, we find the method is not good for the range (400,900), but good in range (400,2400)


# In[115]:


clf1= SVC(C=7,kernel='rbf',gamma=1)
clf1.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, clf1.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, clf1.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, clf1.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, clf1.predict(xpca1_test)))
print ('F1_train：', f1_score(ypca1_train, clf1.predict(xpca1_train),average='weighted'))
print ('F1_test：', f1_score(ypca1_test, clf1.predict(xpca1_test),average='weighted'))


# In[117]:


Clf1= SVC(C=6,kernel='rbf',gamma=1)
Clf1.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, Clf1.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, Clf1.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, Clf1.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, Clf1.predict(xpca2_test)))
print ('F1_train：', f1_score(ypca2_train, Clf1.predict(xpca2_train),average='weighted'))
print ('F1_test：', f1_score(ypca2_test, Clf1.predict(xpca2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca2_train, Clf1.predict(xpca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca2_test, Clf1.predict(xpca2_test)))


# In[118]:


Clf1.fit(xpca3_train, ypca3_train)
print ('accuracy_train：', accuracy_score(ypca3_train, Clf1.predict(xpca3_train)))
print ('accuracy_test：', accuracy_score(ypca3_test, Clf1.predict(xpca3_test)))

print ('Kappa_train：', cohen_kappa_score(ypca3_train, Clf1.predict(xpca3_train)))
print ('Kappa_test：', cohen_kappa_score(ypca3_test, Clf1.predict(xpca3_test)))
print ('F1_train：', f1_score(ypca3_train, Clf1.predict(xpca3_train),average='weighted'))
print ('F1_test：', f1_score(ypca3_test, Clf1.predict(xpca3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca3_train, Clf1.predict(xpca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca3_test, Clf1.predict(xpca3_test)))


# In[119]:


scores = cross_val_score(Clf1,X_PCA3,Target, cv=20)
sum(scores)/20


# In[120]:


clf3= SVC(C=1,kernel='rbf',gamma=1)
Clf3= SVC(C=1,kernel='rbf',gamma=1)


# In[121]:


clf3.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, clf3.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, clf3.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, clf3.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, clf3.predict(xlle1_test)))
print ('F1_train：', f1_score(ylle1_train, clf3.predict(xlle1_train),average='weighted'))
print ('F1_test：', f1_score(ylle1_test, clf3.predict(xlle1_test),average='weighted'))


# In[123]:


Clf3.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, Clf3.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, Clf3.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, Clf3.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, Clf3.predict(xlle2_test)))
print ('F1_train：', f1_score(ylle2_train, Clf3.predict(xlle2_train),average='weighted'))
print ('F1_test：', f1_score(ylle2_test, Clf3.predict(xlle2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, Clf3.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, Clf3.predict(xlle2_test)))


# In[124]:


scores = cross_val_score(Clf3,X_ICA2,target, cv=20)
sum(scores)/20


# In[125]:


Clf3.fit(xlle3_train, ylle3_train)
print ('accuracy_train：', accuracy_score(ylle3_train, Clf3.predict(xlle3_train)))
print ('accuracy_test：', accuracy_score(ylle3_test, Clf3.predict(xlle3_test)))

print ('Kappa_train：', cohen_kappa_score(ylle3_train, Clf3.predict(xlle3_train)))
print ('Kappa_test：', cohen_kappa_score(ylle3_test, Clf3.predict(xlle3_test)))
print ('F1_train：', f1_score(ylle3_train, Clf3.predict(xlle3_train),average='weighted'))
print ('F1_test：', f1_score(ylle3_test, Clf3.predict(xlle3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca3_train, Clf3.predict(xIca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca3_test, Clf3.predict(xIca3_test)))


# In[126]:


scores = cross_val_score(Clf3,X_lle3,Target, cv=20)
sum(scores)/20


# In[127]:


clf= SVC(C=1,kernel='rbf',gamma=1)
Clf= SVC(C=1,kernel='rbf',gamma=0.2)


# In[128]:


clf.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, clf.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, clf.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, clf.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, clf.predict(x1_test)))
print ('F1_train：', f1_score(y1_train, clf.predict(x1_train),average='weighted'))
print ('F1_test：', f1_score(y1_test, clf.predict(x1_test),average='weighted'))


# In[131]:


Clf.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, Clf.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, Clf.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, Clf.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, Clf.predict(x2_test)))
print ('F1_train：', f1_score(y2_train, Clf.predict(x2_train),average='weighted'))
print ('F1_test：', f1_score(y2_test, Clf.predict(x2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y2_train, Clf.predict(x2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y2_test, Clf.predict(x2_test)))


# In[132]:


scores = cross_val_score(Clf,wavelength2,target, cv=20)
sum(scores)/20


# In[133]:


Clf.fit(x3_train, y3_train)
print ('accuracy_train：', accuracy_score(y3_train, Clf.predict(x3_train)))
print ('accuracy_test：', accuracy_score(y3_test, Clf.predict(x3_test)))

print ('Kappa_train：', cohen_kappa_score(y3_train, Clf.predict(x3_train)))
print ('Kappa_test：', cohen_kappa_score(y3_test, Clf.predict(x3_test)))
print ('F1_train：', f1_score(y3_train, Clf.predict(x3_train),average='weighted'))
print ('F1_test：', f1_score(y3_test, Clf.predict(x3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y3_train, Clf.predict(x3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y3_test, Clf.predict(x3_test)))


# In[134]:


scores = cross_val_score(Clf,wavelength3,Target, cv=20)
sum(scores)/20


# In[135]:


# after comparing svm models, I find this model is not very good,accuracy around 0.85 especially for data in range (400,900).
# for the range (400,2400),it seems remove the noisy data range (1800,2000) is a little better than remove the mud samples, accuracy 
# around 0.94.
# reduce can just have a little improvement for SVM model, the most benefit is runs faster after reducing the dimension.
# the bigest problem is SVM model is difficult to set the range of gamma.
# compared with naive bayes model, SVM model seems a litle bad.


# In[136]:


# Then we consider the tree models, to avoid overfitting, roughly set a depth.
# consider different dimension reduction method, pca, Ica, lle.


# In[137]:


# this decision tree model, I am sure it is bad than Random forest and GBDT, so no need to consider much.
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier(max_depth=8)


# In[138]:


clf_tree.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, clf_tree.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, clf_tree.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, clf_tree.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, clf_tree.predict(x1_test)))
print ('F1_train：', f1_score(y1_train, clf_tree.predict(x1_train),average='weighted'))
print ('F1_test：', f1_score(y1_test, clf_tree.predict(x1_test),average='weighted'))


# In[139]:


clf_tree.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, clf_tree.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, clf_tree.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, clf_tree.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, clf_tree.predict(x2_test)))
print ('F1_train：', f1_score(y2_train, clf_tree.predict(x2_train),average='weighted'))
print ('F1_test：', f1_score(y2_test, clf_tree.predict(x2_test),average='weighted'))


# In[140]:


clf_tree.fit(x3_train, y3_train)
print ('accuracy_train：', accuracy_score(y3_train, clf_tree.predict(x3_train)))
print ('accuracy_test：', accuracy_score(y3_test, clf_tree.predict(x3_test)))

print ('Kappa_train：', cohen_kappa_score(y3_train, clf_tree.predict(x3_train)))
print ('Kappa_test：', cohen_kappa_score(y3_test, clf_tree.predict(x3_test)))
print ('F1_train：', f1_score(y3_train, clf_tree.predict(x3_train),average='weighted'))
print ('F1_test：', f1_score(y3_test, clf_tree.predict(x3_test),average='weighted'))


# In[141]:


clf_tree.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, clf_tree.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, clf_tree.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, clf_tree.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, clf_tree.predict(xpca1_test)))
print ('F1_train：', f1_score(ypca1_train, clf_tree.predict(xpca1_train),average='weighted'))
print ('F1_test：', f1_score(ypca1_test, clf_tree.predict(xpca1_test),average='weighted'))


# In[142]:


clf_tree.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, clf_tree.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, clf_tree.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, clf_tree.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, clf_tree.predict(xpca2_test)))
print ('F1_train：', f1_score(ypca2_train, clf_tree.predict(xpca2_train),average='weighted'))
print ('F1_test：', f1_score(ypca2_test, clf_tree.predict(xpca2_test),average='weighted'))


# In[143]:


clf_tree.fit(xpca3_train, ypca3_train)
print ('accuracy_train：', accuracy_score(ypca3_train, clf_tree.predict(xpca3_train)))
print ('accuracy_test：', accuracy_score(ypca3_test, clf_tree.predict(xpca3_test)))

print ('Kappa_train：', cohen_kappa_score(ypca3_train, clf_tree.predict(xpca3_train)))
print ('Kappa_test：', cohen_kappa_score(ypca3_test, clf_tree.predict(xpca3_test)))
print ('F1_train：', f1_score(ypca3_train, clf_tree.predict(xpca3_train),average='weighted'))
print ('F1_test：', f1_score(ypca3_test, clf_tree.predict(xpca3_test),average='weighted'))


# In[144]:


clf_tree.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, clf_tree.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, clf_tree.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, clf_tree.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, clf_tree.predict(xIca1_test)))
print ('F1_train：', f1_score(yIca1_train, clf_tree.predict(xIca1_train),average='weighted'))
print ('F1_test：', f1_score(yIca1_test, clf_tree.predict(xIca1_test),average='weighted'))


# In[145]:


clf_tree.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, clf_tree.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, clf_tree.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, clf_tree.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, clf_tree.predict(xIca2_test)))
print ('F1_train：', f1_score(yIca2_train, clf_tree.predict(xIca2_train),average='weighted'))
print ('F1_test：', f1_score(yIca2_test, clf_tree.predict(xIca2_test),average='weighted'))


# In[146]:


clf_tree.fit(xIca3_train, yIca3_train)
print ('accuracy_train：', accuracy_score(yIca3_train, clf_tree.predict(xIca3_train)))
print ('accuracy_test：', accuracy_score(yIca3_test, clf_tree.predict(xIca3_test)))

print ('Kappa_train：', cohen_kappa_score(yIca3_train, clf_tree.predict(xIca3_train)))
print ('Kappa_test：', cohen_kappa_score(yIca3_test, clf_tree.predict(xIca3_test)))
print ('F1_train：', f1_score(yIca3_train, clf_tree.predict(xIca3_train),average='weighted'))
print ('F1_test：', f1_score(yIca3_test, clf_tree.predict(xIca3_test),average='weighted'))


# In[147]:


scores = cross_val_score(clf_tree,X_ICA3,Target, cv=20)
sum(scores)/20


# In[148]:


clf_tree.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, clf_tree.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, clf_tree.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, clf_tree.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, clf_tree.predict(xlle1_test)))
print ('F1_train：', f1_score(ylle1_train, clf_tree.predict(xlle1_train),average='weighted'))
print ('F1_test：', f1_score(ylle1_test, clf_tree.predict(xlle1_test),average='weighted'))


# In[149]:


clf_tree.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, clf_tree.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, clf_tree.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, clf_tree.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, clf_tree.predict(xlle2_test)))
print ('F1_train：', f1_score(ylle2_train, clf_tree.predict(xlle2_train),average='weighted'))
print ('F1_test：', f1_score(ylle2_test, clf_tree.predict(xlle2_test),average='weighted'))


# In[150]:


clf_tree.fit(xlle3_train, ylle3_train)
print ('accuracy_train：', accuracy_score(ylle3_train, clf_tree.predict(xlle3_train)))
print ('accuracy_test：', accuracy_score(ylle3_test, clf_tree.predict(xlle3_test)))

print ('Kappa_train：', cohen_kappa_score(ylle3_train, clf_tree.predict(xlle3_train)))
print ('Kappa_test：', cohen_kappa_score(ylle3_test, clf_tree.predict(xlle3_test)))
print ('F1_train：', f1_score(ylle3_train, clf_tree.predict(xlle3_train),average='weighted'))
print ('F1_test：', f1_score(ylle3_test, clf_tree.predict(xlle3_test),average='weighted'))


# In[151]:


# for the model of decision tree, we can find it seems using the range of (400,900) is worse than the range of (400,2400).
# but the effect of the tree model is not good, so w consider to improve this.
# we can think adding boosting to the tree to improve the model, boosting is a good way to improve model.


# In[152]:


from sklearn.ensemble import AdaBoostClassifier


# In[153]:


bdt1 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=8),algorithm="SAMME",n_estimators=100, learning_rate=0.6)


# In[154]:


bdt1.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, bdt1.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, bdt1.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, bdt1.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, bdt1.predict(x1_test)))
print ('F1_train：', f1_score(y1_train, bdt1.predict(x1_train),average='weighted'))
print ('F1_test：', f1_score(y1_test, bdt1.predict(x1_test),average='weighted'))


# In[155]:


bdt1.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, bdt1.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, bdt1.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, bdt1.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, bdt1.predict(x2_test)))
print ('F1_train：', f1_score(y2_train, bdt1.predict(x2_train),average='weighted'))
print ('F1_test：', f1_score(y2_test, bdt1.predict(x2_test),average='weighted'))


# In[156]:


bdt1.fit(x3_train, y3_train)
print ('accuracy_train：', accuracy_score(y3_train, bdt1.predict(x3_train)))
print ('accuracy_test：', accuracy_score(y3_test, bdt1.predict(x3_test)))

print ('Kappa_train：', cohen_kappa_score(y3_train, bdt1.predict(x3_train)))
print ('Kappa_test：', cohen_kappa_score(y3_test, bdt1.predict(x3_test)))
print ('F1_train：', f1_score(y3_train, bdt1.predict(x3_train),average='weighted'))
print ('F1_test：', f1_score(y3_test, bdt1.predict(x3_test),average='weighted'))


# In[157]:


bdt1.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, bdt1.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, bdt1.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, bdt1.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, bdt1.predict(xpca1_test)))
print ('F1_train：', f1_score(ypca1_train, bdt1.predict(xpca1_train),average='weighted'))
print ('F1_test：', f1_score(ypca1_test, bdt1.predict(xpca1_test),average='weighted'))


# In[158]:


bdt1.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, bdt1.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, bdt1.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, bdt1.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, bdt1.predict(xpca2_test)))
print ('F1_train：', f1_score(ypca2_train, bdt1.predict(xpca2_train),average='weighted'))
print ('F1_test：', f1_score(ypca2_test, bdt1.predict(xpca2_test),average='weighted'))


# In[159]:


bdt1.fit(xpca3_train, ypca3_train)
print ('accuracy_train：', accuracy_score(ypca3_train, bdt1.predict(xpca3_train)))
print ('accuracy_test：', accuracy_score(ypca3_test, bdt1.predict(xpca3_test)))

print ('Kappa_train：', cohen_kappa_score(ypca3_train, bdt1.predict(xpca3_train)))
print ('Kappa_test：', cohen_kappa_score(ypca3_test, bdt1.predict(xpca3_test)))
print ('F1_train：', f1_score(ypca3_train, bdt1.predict(xpca3_train),average='weighted'))
print ('F1_test：', f1_score(ypca3_test, bdt1.predict(xpca3_test),average='weighted'))


# In[162]:


# the PCA seems not good, then consider ICA, it is good.


# In[163]:


bdt1.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, bdt1.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, bdt1.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, bdt1.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, bdt1.predict(xIca1_test)))
print ('F1_train：', f1_score(yIca1_train, bdt1.predict(xIca1_train),average='weighted'))
print ('F1_test：', f1_score(yIca1_test, bdt1.predict(xIca1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca1_train, bdt1.predict(xIca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca1_test, bdt1.predict(xIca1_test)))


# In[161]:


scores = cross_val_score(bdt1,X_ICA1,target, cv=20)
sum(scores)/20


# In[164]:


bdt1.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, bdt1.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, bdt1.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, bdt1.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, bdt1.predict(xIca2_test)))
print ('F1_train：', f1_score(yIca2_train, bdt1.predict(xIca2_train),average='weighted'))
print ('F1_test：', f1_score(yIca2_test, bdt1.predict(xIca2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca2_train, bdt1.predict(xIca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca2_test, bdt1.predict(xIca2_test)))


# In[165]:


scores = cross_val_score(bdt1,X_ICA2,target, cv=20)
sum(scores)/20


# In[166]:


bdt1.fit(xIca3_train, yIca3_train)
print ('accuracy_train：', accuracy_score(yIca3_train, bdt1.predict(xIca3_train)))
print ('accuracy_test：', accuracy_score(yIca3_test, bdt1.predict(xIca3_test)))

print ('Kappa_train：', cohen_kappa_score(yIca3_train, bdt1.predict(xIca3_train)))
print ('Kappa_test：', cohen_kappa_score(yIca3_test, bdt1.predict(xIca3_test)))
print ('F1_train：', f1_score(yIca3_train, bdt1.predict(xIca3_train),average='weighted'))
print ('F1_test：', f1_score(yIca3_test, bdt1.predict(xIca3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca3_train, bdt1.predict(xIca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca3_test, bdt1.predict(xIca3_test)))


# In[167]:


scores = cross_val_score(bdt1,X_ICA3,Target, cv=20)
sum(scores)/20


# In[169]:


bdt1.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, bdt1.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, bdt1.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, bdt1.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, bdt1.predict(xlle1_test)))
print ('F1_train：', f1_score(ylle1_train, bdt1.predict(xlle1_train),average='weighted'))
print ('F1_test：', f1_score(ylle1_test, bdt1.predict(xlle1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle1_train, bdt1.predict(xlle1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle1_test, bdt1.predict(xlle1_test)))


# In[171]:


scores = cross_val_score(bdt1,X_lle1,target, cv=20)
sum(scores)/20


# In[170]:


bdt1.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, bdt1.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, bdt1.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, bdt1.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, bdt1.predict(xlle2_test)))
print ('F1_train：', f1_score(ylle2_train, bdt1.predict(xlle2_train),average='weighted'))
print ('F1_test：', f1_score(ylle2_test, bdt1.predict(xlle2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, bdt1.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, bdt1.predict(xlle2_test)))


# In[172]:


scores = cross_val_score(bdt1,X_lle2,target, cv=20)
sum(scores)/20


# In[174]:


bdt1.fit(xlle3_train, ylle3_train)
print ('accuracy_train：', accuracy_score(ylle3_train, bdt1.predict(xlle3_train)))
print ('accuracy_test：', accuracy_score(ylle3_test, bdt1.predict(xlle3_test)))

print ('Kappa_train：', cohen_kappa_score(ylle3_train, bdt1.predict(xlle3_train)))
print ('Kappa_test：', cohen_kappa_score(ylle3_test, bdt1.predict(xlle3_test)))
print ('F1_train：', f1_score(ylle3_train, bdt1.predict(xlle3_train),average='weighted'))
print ('F1_test：', f1_score(ylle3_test, bdt1.predict(xlle3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle3_train, bdt1.predict(xlle3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle3_test, bdt1.predict(xlle3_test)))


# In[175]:


scores = cross_val_score(bdt1,X_lle3,Target, cv=20)
sum(scores)/20


# In[176]:


# now we can find for boost decision tree, the effect is much better, for range (400,900), the accuracy is around 0.94,
# for range (400,2400), the two methods almost the same effect, accuracy around 0.94.
# but it may overfit, so we consider the bagging method.


# In[177]:


# split train data,and test data at random
xpca1_train, xpca1_test, ypca1_train, ypca1_test = train_test_split(X_PCA1,target, test_size=0.3)
xIca1_train, xIca1_test, yIca1_train, yIca1_test = train_test_split(X_ICA1,target, test_size=0.3)
xlle1_train, xlle1_test, ylle1_train, ylle1_test = train_test_split(X_lle1,target, test_size=0.3)
x1_train, x1_test, y1_train, y1_test = train_test_split(wavelength1,target, test_size=0.3)

xpca2_train, xpca2_test, ypca2_train, ypca2_test = train_test_split(X_PCA2,target, test_size=0.3)
xIca2_train, xIca2_test, yIca2_train, yIca2_test = train_test_split(X_ICA2,target, test_size=0.3)
xlle2_train, xlle2_test, ylle2_train, ylle2_test = train_test_split(X_lle2,target, test_size=0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(wavelength2,target, test_size=0.3)

xpca3_train, xpca3_test, ypca3_train, ypca3_test = train_test_split(X_PCA3,Target, test_size=0.3)
xIca3_train, xIca3_test, yIca3_train, yIca3_test = train_test_split(X_ICA3,Target, test_size=0.3)
xlle3_train, xlle3_test, ylle3_train, ylle3_test = train_test_split(X_lle3,Target, test_size=0.3)
x3_train, x3_test, y3_train, y3_test = train_test_split(wavelength3,Target, test_size=0.3)


# In[178]:


# Random Forest is a improvement of Decision Tree , too avoid overfitting, I choose set the max depth.
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
randomforest = RandomForestClassifier(max_depth=10)


# In[179]:


randomforest.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, randomforest.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, randomforest.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, randomforest.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, randomforest.predict(xpca1_test)))
print ('F1_train：', f1_score(ypca1_train, randomforest.predict(xpca1_train),average='weighted'))
print ('F1_test：', f1_score(ypca1_test, randomforest.predict(xpca1_test),average='weighted'))


# In[180]:


randomforest.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, randomforest.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, randomforest.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, randomforest.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, randomforest.predict(xpca2_test)))
print ('F1_train：', f1_score(ypca2_train, randomforest.predict(xpca2_train),average='weighted'))
print ('F1_test：', f1_score(ypca2_test, randomforest.predict(xpca2_test),average='weighted'))


# In[181]:


randomforest.fit(xpca3_train, ypca3_train)
print ('accuracy_train：', accuracy_score(ypca3_train, randomforest.predict(xpca3_train)))
print ('accuracy_test：', accuracy_score(ypca3_test, randomforest.predict(xpca3_test)))

print ('Kappa_train：', cohen_kappa_score(ypca3_train, randomforest.predict(xpca3_train)))
print ('Kappa_test：', cohen_kappa_score(ypca3_test, randomforest.predict(xpca3_test)))
print ('F1_train：', f1_score(ypca3_train, randomforest.predict(xpca3_train),average='weighted'))
print ('F1_test：', f1_score(ypca3_test, randomforest.predict(xpca3_test),average='weighted'))


# In[187]:


randomforest.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, randomforest.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, randomforest.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, randomforest.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, randomforest.predict(xIca1_test)))
print ('F1_train：', f1_score(yIca1_train, randomforest.predict(xIca1_train),average='weighted'))
print ('F1_test：', f1_score(yIca1_test, randomforest.predict(xIca1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca1_train, randomforest.predict(xIca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca1_test, randomforest.predict(xIca1_test)))


# In[185]:


scores = cross_val_score(randomforest,X_ICA1,target, cv=20)
sum(scores)/20


# In[188]:


randomforest.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, randomforest.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, randomforest.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, randomforest.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, randomforest.predict(xIca2_test)))
print ('F1_train：', f1_score(yIca2_train, randomforest.predict(xIca2_train),average='weighted'))
print ('F1_test：', f1_score(yIca2_test, randomforest.predict(xIca2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca2_train, randomforest.predict(xIca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca2_test, randomforest.predict(xIca2_test)))


# In[151]:


scores = cross_val_score(randomforest,X_ICA2,target, cv=20)
sum(scores)/20


# In[189]:


randomforest.fit(xIca3_train, yIca3_train)
print ('accuracy_train：', accuracy_score(yIca3_train, randomforest.predict(xIca3_train)))
print ('accuracy_test：', accuracy_score(yIca3_test, randomforest.predict(xIca3_test)))

print ('Kappa_train：', cohen_kappa_score(yIca3_train, randomforest.predict(xIca3_train)))
print ('Kappa_test：', cohen_kappa_score(yIca3_test, randomforest.predict(xIca3_test)))
print ('F1_train：', f1_score(yIca3_train, randomforest.predict(xIca3_train),average='weighted'))
print ('F1_test：', f1_score(yIca3_test, randomforest.predict(xIca3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca3_train, randomforest.predict(xIca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca3_test, randomforest.predict(xIca3_test)))


# In[190]:


scores = cross_val_score(randomforest,X_ICA3,Target, cv=20)
sum(scores)/20


# In[194]:


randomforest.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, randomforest.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, randomforest.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, randomforest.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, randomforest.predict(xlle1_test)))
print ('F1_train：', f1_score(ylle1_train, randomforest.predict(xlle1_train),average='weighted'))
print ('F1_test：', f1_score(ylle1_test, randomforest.predict(xlle1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle1_train, randomforest.predict(xlle1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle1_test, randomforest.predict(xlle1_test)))


# In[192]:


scores = cross_val_score(randomforest,X_lle1,target, cv=20)
sum(scores)/20


# In[195]:


randomforest.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, randomforest.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, randomforest.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, randomforest.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, randomforest.predict(xlle2_test)))
print ('F1_train：', f1_score(ylle2_train, randomforest.predict(xlle2_train),average='weighted'))
print ('F1_test：', f1_score(ylle2_test, randomforest.predict(xlle2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, randomforest.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, randomforest.predict(xlle2_test)))


# In[196]:


scores = cross_val_score(randomforest,X_lle2,target, cv=20)
sum(scores)/20


# In[197]:


randomforest.fit(xlle3_train, ylle3_train)
print ('accuracy_train：', accuracy_score(ylle3_train, randomforest.predict(xlle3_train)))
print ('accuracy_test：', accuracy_score(ylle3_test, randomforest.predict(xlle3_test)))

print ('Kappa_train：', cohen_kappa_score(ylle3_train, randomforest.predict(xlle3_train)))
print ('Kappa_test：', cohen_kappa_score(ylle3_test, randomforest.predict(xlle3_test)))
print ('F1_train：', f1_score(ylle3_train, randomforest.predict(xlle3_train),average='weighted'))
print ('F1_test：', f1_score(ylle3_test, randomforest.predict(xlle3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle3_train, randomforest.predict(xlle3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle3_test, randomforest.predict(xlle3_test)))


# In[198]:


scores = cross_val_score(randomforest,X_lle3,Target, cv=20)
sum(scores)/20


# In[199]:


randomforest.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, randomforest.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, randomforest.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, randomforest.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, randomforest.predict(x1_test)))
print ('F1_train：', f1_score(y1_train, randomforest.predict(x1_train),average='weighted'))
print ('F1_test：', f1_score(y1_test, randomforest.predict(x1_test),average='weighted'))


# In[201]:


randomforest.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, randomforest.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, randomforest.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, randomforest.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, randomforest.predict(x2_test)))
print ('F1_train：', f1_score(y2_train, randomforest.predict(x2_train),average='weighted'))
print ('F1_test：', f1_score(y2_test, randomforest.predict(x2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y2_train, randomforest.predict(x2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y2_test, randomforest.predict(x2_test)))


# In[203]:


randomforest.fit(x3_train, y3_train)
print ('accuracy_train：', accuracy_score(y3_train, randomforest.predict(x3_train)))
print ('accuracy_test：', accuracy_score(y3_test, randomforest.predict(x3_test)))

print ('Kappa_train：', cohen_kappa_score(y3_train, randomforest.predict(x3_train)))
print ('Kappa_test：', cohen_kappa_score(y3_test, randomforest.predict(x3_test)))
print ('F1_train：', f1_score(y3_train, randomforest.predict(x3_train),average='weighted'))
print ('F1_test：', f1_score(y3_test, randomforest.predict(x3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y3_train, randomforest.predict(x3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y3_test, randomforest.predict(x3_test)))


# In[204]:


# we canfind the accuracy of randomforest is round 0.92 in range (400,2400), in range (400,900) the accuracy is round 0.85
# we can find the random forest seems not as good as the boost decision tree, now we consider to a improvement of boost decision tree.


# In[205]:


# The random forset used bagging to improve the method of decision tree, while the GBDT used the boosting to improve .
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
GradientBoosting1 = GradientBoostingClassifier(max_depth=3,n_estimators=50,learning_rate=0.05)
GradientBoosting = GradientBoostingClassifier(max_depth=3,n_estimators=30,learning_rate=0.03)


# In[206]:


GradientBoosting1.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, GradientBoosting1.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, GradientBoosting1.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, GradientBoosting1.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, GradientBoosting1.predict(x1_test)))
print ('F1_train：', f1_score(y1_train, GradientBoosting1.predict(x1_train),average='weighted'))
print ('F1_test：', f1_score(y1_test, GradientBoosting1.predict(x1_test),average='weighted'))


# In[207]:


GradientBoosting1.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, GradientBoosting1.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, GradientBoosting1.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, GradientBoosting1.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, GradientBoosting1.predict(x2_test)))
print ('F1_train：', f1_score(y2_train, GradientBoosting1.predict(x2_train),average='weighted'))
print ('F1_test：', f1_score(y2_test, GradientBoosting1.predict(x2_test),average='weighted'))


# In[208]:


GradientBoosting1.fit(x3_train, y3_train)
print ('accuracy_train：', accuracy_score(y3_train, GradientBoosting1.predict(x3_train)))
print ('accuracy_test：', accuracy_score(y3_test, GradientBoosting1.predict(x3_test)))

print ('Kappa_train：', cohen_kappa_score(y3_train, GradientBoosting1.predict(x3_train)))
print ('Kappa_test：', cohen_kappa_score(y3_test, GradientBoosting1.predict(x3_test)))
print ('F1_train：', f1_score(y3_train, GradientBoosting1.predict(x3_train),average='weighted'))
print ('F1_test：', f1_score(y3_test, GradientBoosting1.predict(x3_test),average='weighted'))


# In[209]:


GradientBoosting1.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, GradientBoosting1.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, GradientBoosting1.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, GradientBoosting1.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, GradientBoosting1.predict(xpca1_test)))
print ('F1_train：', f1_score(ypca1_train, GradientBoosting1.predict(xpca1_train),average='weighted'))
print ('F1_test：', f1_score(ypca1_test, GradientBoosting1.predict(xpca1_test),average='weighted'))


# In[210]:


GradientBoosting.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, GradientBoosting.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, GradientBoosting.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, GradientBoosting.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, GradientBoosting.predict(xpca2_test)))
print ('F1_train：', f1_score(ypca2_train, GradientBoosting.predict(xpca2_train),average='weighted'))
print ('F1_test：', f1_score(ypca2_test, GradientBoosting.predict(xpca2_test),average='weighted'))


# In[211]:


GradientBoosting.fit(xpca3_train, ypca3_train)
print ('accuracy_train：', accuracy_score(ypca3_train, GradientBoosting.predict(xpca3_train)))
print ('accuracy_test：', accuracy_score(ypca3_test, GradientBoosting.predict(xpca3_test)))

print ('Kappa_train：', cohen_kappa_score(ypca3_train, GradientBoosting.predict(xpca3_train)))
print ('Kappa_test：', cohen_kappa_score(ypca3_test, GradientBoosting.predict(xpca3_test)))
print ('F1_train：', f1_score(ypca3_train, GradientBoosting.predict(xpca3_train),average='weighted'))
print ('F1_test：', f1_score(ypca3_test, GradientBoosting.predict(xpca3_test),average='weighted'))


# In[213]:


GradientBoosting1.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, GradientBoosting1.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, GradientBoosting1.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, GradientBoosting1.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, GradientBoosting1.predict(xIca1_test)))
print ('F1_train：', f1_score(yIca1_train, GradientBoosting1.predict(xIca1_train),average='weighted'))
print ('F1_test：', f1_score(yIca1_test, GradientBoosting1.predict(xIca1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca1_train, GradientBoosting1.predict(xIca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca1_test, GradientBoosting1.predict(xIca1_test)))


# In[214]:


scores = cross_val_score(GradientBoosting1,X_ICA1,target, cv=20)
sum(scores)/20


# In[215]:


GradientBoosting1.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, GradientBoosting1.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, GradientBoosting1.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, GradientBoosting1.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, GradientBoosting1.predict(xIca2_test)))
print ('F1_train：', f1_score(yIca2_train, GradientBoosting1.predict(xIca2_train),average='weighted'))
print ('F1_test：', f1_score(yIca2_test, GradientBoosting1.predict(xIca2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca2_train, GradientBoosting1.predict(xIca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca2_test, GradientBoosting1.predict(xIca2_test)))


# In[216]:


scores = cross_val_score(GradientBoosting1, X_ICA2,target, cv=20)
sum(scores)/20


# In[217]:


GradientBoosting.fit(xIca3_train, yIca3_train)
print ('accuracy_train：', accuracy_score(yIca3_train, GradientBoosting.predict(xIca3_train)))
print ('accuracy_test：', accuracy_score(yIca3_test, GradientBoosting.predict(xIca3_test)))

print ('Kappa_train：', cohen_kappa_score(yIca3_train, GradientBoosting.predict(xIca3_train)))
print ('Kappa_test：', cohen_kappa_score(yIca3_test, GradientBoosting.predict(xIca3_test)))
print ('F1_train：', f1_score(yIca3_train, GradientBoosting.predict(xIca3_train),average='weighted'))
print ('F1_test：', f1_score(yIca3_test, GradientBoosting.predict(xIca3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca3_train, GradientBoosting1.predict(xIca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca3_test, GradientBoosting1.predict(xIca3_test)))


# In[218]:


scores = cross_val_score(GradientBoosting1,X_ICA3,Target, cv=20)
sum(scores)/20


# In[219]:


GradientBoosting1.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, GradientBoosting1.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, GradientBoosting1.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, GradientBoosting1.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, GradientBoosting1.predict(xlle1_test)))
print ('F1_train：', f1_score(ylle1_train, GradientBoosting1.predict(xlle1_train),average='weighted'))
print ('F1_test：', f1_score(ylle1_test, GradientBoosting1.predict(xlle1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle1_train, GradientBoosting1.predict(xlle1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle1_test, GradientBoosting1.predict(xlle1_test)))


# In[220]:


scores = cross_val_score(GradientBoosting1,X_lle1,target, cv=20)
sum(scores)/20


# In[223]:


GradientBoosting.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, GradientBoosting.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, GradientBoosting.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, GradientBoosting.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, GradientBoosting.predict(xlle2_test)))
print ('F1_train：', f1_score(ylle2_train, GradientBoosting1.predict(xlle2_train),average='weighted'))
print ('F1_test：', f1_score(ylle2_test, GradientBoosting1.predict(xlle2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, GradientBoosting1.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, GradientBoosting1.predict(xlle2_test)))


# In[222]:


scores = cross_val_score(GradientBoosting1,X_lle2,target, cv=20)
sum(scores)/20


# In[224]:


GradientBoosting.fit(xlle3_train, ylle3_train)
print ('accuracy_train：', accuracy_score(ylle3_train, GradientBoosting.predict(xlle3_train)))
print ('accuracy_test：', accuracy_score(ylle3_test, GradientBoosting.predict(xlle3_test)))

print ('Kappa_train：', cohen_kappa_score(ylle3_train, GradientBoosting.predict(xlle3_train)))
print ('Kappa_test：', cohen_kappa_score(ylle3_test, GradientBoosting.predict(xlle3_test)))
print ('F1_train：', f1_score(ylle3_train, GradientBoosting.predict(xlle3_train),average='weighted'))
print ('F1_test：', f1_score(ylle3_test, GradientBoosting.predict(xlle3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle3_train, GradientBoosting1.predict(xlle3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle3_test, GradientBoosting1.predict(xlle3_test)))


# In[225]:


scores = cross_val_score(GradientBoosting1,X_lle3,Target, cv=20)
sum(scores)/20


# In[226]:


# now we find the for ICA + GBDT, is a good method, but seems this may still nobetter than boost tree, and the effect almost same 
# with the boost + naive bayes.


# In[227]:


# I can also use sklearn to duild Netural Network, but it can be difficult to tell the meaning of the net.
# also the best net is difficult to find, I can just try some and find a roughly suitable one.
# I also find for DNN, reduce dimension is useless since the netural network can do this.


# In[228]:


# split train data,and test data at random
xpca1_train, xpca1_test, ypca1_train, ypca1_test = train_test_split(X_PCA1,target, test_size=0.3)
xIca1_train, xIca1_test, yIca1_train, yIca1_test = train_test_split(X_ICA1,target, test_size=0.3)
xlle1_train, xlle1_test, ylle1_train, ylle1_test = train_test_split(X_lle1,target, test_size=0.3)
x1_train, x1_test, y1_train, y1_test = train_test_split(wavelength1,target, test_size=0.3)

xpca2_train, xpca2_test, ypca2_train, ypca2_test = train_test_split(X_PCA2,target, test_size=0.3)
xIca2_train, xIca2_test, yIca2_train, yIca2_test = train_test_split(X_ICA2,target, test_size=0.3)
xlle2_train, xlle2_test, ylle2_train, ylle2_test = train_test_split(X_lle2,target, test_size=0.3)
x2_train, x2_test, y2_train, y2_test = train_test_split(wavelength2,target, test_size=0.3)

xpca3_train, xpca3_test, ypca3_train, ypca3_test = train_test_split(X_PCA3,Target, test_size=0.3)
xIca3_train, xIca3_test, yIca3_train, yIca3_test = train_test_split(X_ICA3,Target, test_size=0.3)
xlle3_train, xlle3_test, ylle3_train, ylle3_test = train_test_split(X_lle3,Target, test_size=0.3)
x3_train, x3_test, y3_train, y3_test = train_test_split(wavelength3,Target, test_size=0.3)


# In[229]:


from sklearn.neural_network import MLPClassifier
MLP1=MLPClassifier(random_state=1,activation ="relu",solver="lbfgs",max_iter=1000,beta_1=0.8,beta_2=0.8,hidden_layer_sizes=(40, ))
MLP2=MLPClassifier(random_state=1,activation ="relu",solver="lbfgs",max_iter=1000,beta_1=0.7,beta_2=0.7,hidden_layer_sizes=(20, ))


# In[230]:


MLP1.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, MLP1.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, MLP1.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, MLP1.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, MLP1.predict(x1_test)))
print ('F1_train：', f1_score(y1_train, MLP1.predict(x1_train),average='weighted'))
print ('F1_test：', f1_score(y1_test, MLP1.predict(x1_test),average='weighted'))


# In[231]:


MLP3=MLPClassifier(random_state=1,activation ="relu",solver="lbfgs",max_iter=1000,beta_1=0.8,beta_2=0.8,hidden_layer_sizes=(8, ))


# In[233]:


MLP3.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, MLP3.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, MLP3.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, MLP3.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, MLP3.predict(xIca1_test)))
print ('F1_train：', f1_score(yIca1_train, MLP3.predict(xIca1_train),average='weighted'))
print ('F1_test：', f1_score(yIca1_test, MLP3.predict(xIca1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca1_train, MLP3.predict(xIca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca1_test, MLP3.predict(xIca1_test)))


# In[234]:


scores = cross_val_score(MLP3,X_ICA1,target, cv=20)
sum(scores)/20


# In[235]:


import time
start =time.clock()

MLP3.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, MLP3.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, MLP3.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, MLP3.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, MLP3.predict(xIca1_test)))
print ('F1_train：', f1_score(yIca1_train, MLP3.predict(xIca1_train),average='weighted'))
print ('F1_test：', f1_score(yIca1_test, MLP3.predict(xIca1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca1_train, MLP3.predict(xIca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca1_test, MLP3.predict(xIca1_test)))

end=time.clock()
print('Running time: %s Seconds'%(end-start))


# In[236]:


MLP3.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, MLP3.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, MLP3.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, MLP3.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, MLP3.predict(xlle1_test)))
print ('F1_train：', f1_score(ylle1_train, MLP3.predict(xlle1_train),average='weighted'))
print ('F1_test：', f1_score(ylle1_test, MLP3.predict(xlle1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle1_train, MLP3.predict(xlle1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle1_test, MLP3.predict(xlle1_test)))


# In[237]:


# For netural network, range (400,900), the accuracy ca reach around 0.965, using ICA to reduce dimension


# In[238]:


MLP2.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, MLP2.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, MLP2.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, MLP2.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, MLP2.predict(x2_test)))
print ('F1_train：', f1_score(y2_train, MLP2.predict(x2_train),average='weighted'))
print ('F1_test：', f1_score(y2_test, MLP2.predict(x2_test),average='weighted'))


# In[239]:


MLP4=MLPClassifier(random_state=1,activation ="relu",solver="lbfgs",max_iter=500,beta_1=0.8,beta_2=0.8,hidden_layer_sizes=(8, ))


# In[240]:


MLP4.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, MLP4.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, MLP4.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, MLP4.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, MLP4.predict(xpca2_test)))
print ('F1_train：', f1_score(ypca2_train, MLP4.predict(xpca2_train),average='weighted'))
print ('F1_test：', f1_score(ypca2_test, MLP4.predict(xpca2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca2_train, MLP4.predict(xpca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca2_test, MLP4.predict(xpca2_test)))


# In[243]:


MLP4.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, MLP4.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, MLP4.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, MLP4.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, MLP4.predict(xIca2_test)))
print ('F1_train：', f1_score(yIca2_train, MLP4.predict(xIca2_train),average='weighted'))
print ('F1_test：', f1_score(yIca2_test, MLP4.predict(xIca2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca2_train, MLP4.predict(xIca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca2_test, MLP4.predict(xIca2_test)))


# In[244]:


scores = cross_val_score(MLP4,X_ICA2,target, cv=20)
sum(scores)/20


# In[245]:


MLP4.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, MLP4.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, MLP4.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, MLP4.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, MLP4.predict(xlle2_test)))
print ('F1_train：', f1_score(ylle2_train, MLP4.predict(xlle2_train),average='weighted'))
print ('F1_test：', f1_score(ylle2_test, MLP4.predict(xlle2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, MLP4.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, MLP4.predict(xlle2_test)))


# In[246]:


scores = cross_val_score(MLP4,X_lle2,target, cv=20)
sum(scores)/20


# In[247]:


# the reduce dimension in netural network here is a good choice.
# the accuracy is around 0.93


# In[248]:


MLP2.fit(x3_train, y3_train)
print ('accuracy_train：', accuracy_score(y3_train, MLP2.predict(x3_train)))
print ('accuracy_test：', accuracy_score(y3_test, MLP2.predict(x3_test)))

print ('Kappa_train：', cohen_kappa_score(y3_train, MLP2.predict(x3_train)))
print ('Kappa_test：', cohen_kappa_score(y3_test, MLP2.predict(x3_test)))
print ('F1_train：', f1_score(y3_train, MLP2.predict(x3_train),average='weighted'))
print ('F1_test：', f1_score(y3_test, MLP2.predict(x3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y3_train, MLP2.predict(x3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y3_test, MLP2.predict(x3_test)))


# In[249]:


import time
start =time.clock()

MLP2.fit(x3_train, y3_train)
print ('accuracy_train：', accuracy_score(y3_train, MLP2.predict(x3_train)))
print ('accuracy_test：', accuracy_score(y3_test, MLP2.predict(x3_test)))

print ('Kappa_train：', cohen_kappa_score(y3_train, MLP2.predict(x3_train)))
print ('Kappa_test：', cohen_kappa_score(y3_test, MLP2.predict(x3_test)))
print ('F1_train：', f1_score(y3_train, MLP2.predict(x3_train),average='weighted'))
print ('F1_test：', f1_score(y3_test, MLP2.predict(x3_test),average='weighted'))

end=time.clock()
print('Running time: %s Seconds'%(end-start))


# In[251]:


# then this method is good,ICA + DNN arruracy can reach 0.97 but has two bigest problem.
# the code runs slow,only classify 3 classes.


# In[252]:


MLP5=MLPClassifier(random_state=1,activation ="relu",solver="lbfgs",max_iter=200,beta_1=0.8,beta_2=0.8,hidden_layer_sizes=(5, ))


# In[253]:


MLP5.fit(xpca3_train, ypca3_train)
print ('accuracy_train：', accuracy_score(ypca3_train, MLP5.predict(xpca3_train)))
print ('accuracy_test：', accuracy_score(ypca3_test, MLP5.predict(xpca3_test)))

print ('Kappa_train：', cohen_kappa_score(ypca3_train, MLP5.predict(xpca3_train)))
print ('Kappa_test：', cohen_kappa_score(ypca3_test, MLP5.predict(xpca3_test)))
print ('F1_train：', f1_score(ypca3_train, MLP5.predict(xpca3_train),average='weighted'))
print ('F1_test：', f1_score(ypca3_test, MLP5.predict(xpca3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca3_train, MLP5.predict(xpca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca3_test, MLP5.predict(xpca3_test)))


# In[254]:


scores = cross_val_score(MLP5, X_PCA3,Target, cv=20)
sum(scores)/20


# In[255]:


MLP5.fit(xIca3_train, yIca3_train)
print ('accuracy_train：', accuracy_score(yIca3_train, MLP5.predict(xIca3_train)))
print ('accuracy_test：', accuracy_score(yIca3_test, MLP5.predict(xIca3_test)))

print ('Kappa_train：', cohen_kappa_score(yIca3_train, MLP5.predict(xIca3_train)))
print ('Kappa_test：', cohen_kappa_score(yIca3_test, MLP5.predict(xIca3_test)))
print ('F1_train：', f1_score(yIca3_train, MLP5.predict(xIca3_train),average='weighted'))
print ('F1_test：', f1_score(yIca3_test, MLP5.predict(xIca3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca3_train, MLP5.predict(xIca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca3_test, MLP5.predict(xIca3_test)))


# In[256]:


scores = cross_val_score(MLP5, X_ICA3,Target, cv=20)
sum(scores)/20


# In[257]:


MLP5.fit(xlle3_train, ylle3_train)
print ('accuracy_train：', accuracy_score(ylle3_train, MLP5.predict(xlle3_train)))
print ('accuracy_test：', accuracy_score(ylle3_test, MLP5.predict(xlle3_test)))

print ('Kappa_train：', cohen_kappa_score(ylle3_train, MLP5.predict(xlle3_train)))
print ('Kappa_test：', cohen_kappa_score(ylle3_test, MLP5.predict(xlle3_test)))
print ('F1_train：', f1_score(ylle3_train, MLP5.predict(xlle3_train),average='weighted'))
print ('F1_test：', f1_score(ylle3_test, MLP5.predict(xlle3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle3_train, MLP5.predict(xlle3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle3_test, MLP5.predict(xlle3_test)))


# In[258]:


scores = cross_val_score(MLP5, X_lle3,Target, cv=20)
sum(scores)/20


# In[259]:


# now we can find if we just classify 3 classes, the effect can be quite good using ICA to reduce the dimension,
# the problem is we can just classify 3 classes.


# In[260]:


# now the problem of DNN is it runs a little slow and the hidden layer is a little difficult to design.
# the range (400,900), accuracy around 0.95, is worse than (400,2400) accuracy around 0.97,
# although when remove the mud class the effect is very good, I still think no need to use this method since it can only classify 3 class


# In[261]:


from sklearn.model_selection import cross_val_score


# In[265]:


pip install xgboost


# In[263]:


pip install graphviz


# In[266]:


from pandas import DataFrame
from xgboost.sklearn import XGBClassifier
from xgboost import plot_tree


# In[267]:


# Then try a improvement of xgboost.it add bagging to the GBDT and has some other improvement.


# In[268]:


xgboost = XGBClassifier(
    n_estimators=50,
    learning_rate =0.3,
    max_depth=4,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.6,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=12,
    reg_lambda=1,
    seed=27,
    num_class= 4)


# In[269]:


xgboost.fit(x1_train, y1_train)
print ('accuracy_train：', accuracy_score(y1_train, xgboost.predict(x1_train)))
print ('accuracy_test：', accuracy_score(y1_test, xgboost.predict(x1_test)))

print ('Kappa_train：', cohen_kappa_score(y1_train, xgboost.predict(x1_train)))
print ('Kappa_test：', cohen_kappa_score(y1_test, xgboost.predict(x1_test)))
print ('F1_train：', f1_score(y1_train, xgboost.predict(x1_train),average='weighted'))
print ('F1_test：', f1_score(y1_test, xgboost.predict(x1_test),average='weighted'))


# In[271]:


xgboost.fit(x2_train, y2_train)
print ('accuracy_train：', accuracy_score(y2_train, xgboost.predict(x2_train)))
print ('accuracy_test：', accuracy_score(y2_test, xgboost.predict(x2_test)))

print ('Kappa_train：', cohen_kappa_score(y2_train, xgboost.predict(x2_train)))
print ('Kappa_test：', cohen_kappa_score(y2_test, xgboost.predict(x2_test)))
print ('F1_train：', f1_score(y2_train, xgboost.predict(x2_train),average='weighted'))
print ('F1_test：', f1_score(y2_test, xgboost.predict(x2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y2_train, xgboost.predict(x2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y2_test, xgboost.predict(x2_test)))


# In[274]:


scores = cross_val_score(xgboost, wavelength2,target, cv=20)
sum(scores)/20


# In[275]:


xgboost.fit(x3_train, y3_train)
print ('accuracy_train：', accuracy_score(y3_train, xgboost.predict(x3_train)))
print ('accuracy_test：', accuracy_score(y3_test, xgboost.predict(x3_test)))

print ('Kappa_train：', cohen_kappa_score(y3_train, xgboost.predict(x3_train)))
print ('Kappa_test：', cohen_kappa_score(y3_test, xgboost.predict(x3_test)))
print ('F1_train：', f1_score(y3_train, xgboost.predict(x3_train),average='weighted'))
print ('F1_test：', f1_score(y3_test, xgboost.predict(x3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y3_train, xgboost.predict(x3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y3_test, xgboost.predict(x3_test)))


# In[276]:


scores = cross_val_score(xgboost, wavelength3,Target, cv=20)
sum(scores)/20


# In[277]:


xgboost.fit(xpca1_train, ypca1_train)
print ('accuracy_train：', accuracy_score(ypca1_train, xgboost.predict(xpca1_train)))
print ('accuracy_test：', accuracy_score(ypca1_test, xgboost.predict(xpca1_test)))

print ('Kappa_train：', cohen_kappa_score(ypca1_train, xgboost.predict(xpca1_train)))
print ('Kappa_test：', cohen_kappa_score(ypca1_test, xgboost.predict(xpca1_test)))
print ('F1_train：', f1_score(ypca1_train, xgboost.predict(xpca1_train),average='weighted'))
print ('F1_test：', f1_score(ypca1_test, xgboost.predict(xpca1_test),average='weighted'))


# In[278]:


xgboost.fit(xpca2_train, ypca2_train)
print ('accuracy_train：', accuracy_score(ypca2_train, xgboost.predict(xpca2_train)))
print ('accuracy_test：', accuracy_score(ypca2_test, xgboost.predict(xpca2_test)))

print ('Kappa_train：', cohen_kappa_score(ypca2_train, xgboost.predict(xpca2_train)))
print ('Kappa_test：', cohen_kappa_score(ypca2_test, xgboost.predict(xpca2_test)))
print ('F1_train：', f1_score(ypca2_train, xgboost.predict(xpca2_train),average='weighted'))
print ('F1_test：', f1_score(ypca2_test, xgboost.predict(xpca2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca2_train, xgboost.predict(xpca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca2_test, xgboost.predict(xpca2_test)))


# In[279]:


xgboost.fit(xpca3_train, ypca3_train)
print ('accuracy_train：', accuracy_score(ypca3_train, xgboost.predict(xpca3_train)))
print ('accuracy_test：', accuracy_score(ypca3_test, xgboost.predict(xpca3_test)))

print ('Kappa_train：', cohen_kappa_score(ypca3_train, xgboost.predict(xpca3_train)))
print ('Kappa_test：', cohen_kappa_score(ypca3_test, xgboost.predict(xpca3_test)))
print ('F1_train：', f1_score(ypca3_train, xgboost.predict(xpca3_train),average='weighted'))
print ('F1_test：', f1_score(ypca3_test, xgboost.predict(xpca3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca3_train, xgboost.predict(xpca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca3_test, xgboost.predict(xpca3_test)))


# In[281]:


xgboost.fit(xIca1_train, yIca1_train)
print ('accuracy_train：', accuracy_score(yIca1_train, xgboost.predict(xIca1_train)))
print ('accuracy_test：', accuracy_score(yIca1_test, xgboost.predict(xIca1_test)))

print ('Kappa_train：', cohen_kappa_score(yIca1_train, xgboost.predict(xIca1_train)))
print ('Kappa_test：', cohen_kappa_score(yIca1_test, xgboost.predict(xIca1_test)))
print ('F1_train：', f1_score(yIca1_train, xgboost.predict(xIca1_train),average='weighted'))
print ('F1_test：', f1_score(yIca1_test, xgboost.predict(xIca1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca1_train, xgboost.predict(xIca1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca1_test, xgboost.predict(xIca1_test)))


# In[284]:


scores = cross_val_score(xgboost, X_ICA1,target, cv=20)
sum(scores)/20


# In[282]:


xgboost.fit(xIca2_train, yIca2_train)
print ('accuracy_train：', accuracy_score(yIca2_train, xgboost.predict(xIca2_train)))
print ('accuracy_test：', accuracy_score(yIca2_test, xgboost.predict(xIca2_test)))

print ('Kappa_train：', cohen_kappa_score(yIca2_train, xgboost.predict(xIca2_train)))
print ('Kappa_test：', cohen_kappa_score(yIca2_test, xgboost.predict(xIca2_test)))
print ('F1_train：', f1_score(yIca2_train, xgboost.predict(xIca2_train),average='weighted'))
print ('F1_test：', f1_score(yIca2_test, xgboost.predict(xIca2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca2_train, xgboost.predict(xIca2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca2_test, xgboost.predict(xIca2_test)))


# In[285]:


scores = cross_val_score(xgboost, X_ICA2,target, cv=20)
sum(scores)/20


# In[283]:


xgboost.fit(xIca3_train, yIca3_train)
print ('accuracy_train：', accuracy_score(yIca3_train, xgboost.predict(xIca3_train)))
print ('accuracy_test：', accuracy_score(yIca3_test, xgboost.predict(xIca3_test)))

print ('Kappa_train：', cohen_kappa_score(yIca3_train, xgboost.predict(xIca3_train)))
print ('Kappa_test：', cohen_kappa_score(yIca3_test, xgboost.predict(xIca3_test)))
print ('F1_train：', f1_score(yIca3_train, xgboost.predict(xIca3_train),average='weighted'))
print ('F1_test：', f1_score(yIca3_test, xgboost.predict(xIca3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca3_train, xgboost.predict(xIca3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca3_test, xgboost.predict(xIca3_test)))


# In[286]:


scores = cross_val_score(xgboost, X_ICA3,Target, cv=20)
sum(scores)/20


# In[287]:


xgboost.fit(xlle1_train, ylle1_train)
print ('accuracy_train：', accuracy_score(ylle1_train, xgboost.predict(xlle1_train)))
print ('accuracy_test：', accuracy_score(ylle1_test, xgboost.predict(xlle1_test)))

print ('Kappa_train：', cohen_kappa_score(ylle1_train, xgboost.predict(xlle1_train)))
print ('Kappa_test：', cohen_kappa_score(ylle1_test, xgboost.predict(xlle1_test)))
print ('F1_train：', f1_score(ylle1_train, xgboost.predict(xlle1_train),average='weighted'))
print ('F1_test：', f1_score(ylle1_test, xgboost.predict(xlle1_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle1_train, xgboost.predict(xlle1_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle1_test, xgboost.predict(xlle1_test)))


# In[288]:


xgboost.fit(xlle2_train, ylle2_train)
print ('accuracy_train：', accuracy_score(ylle2_train, xgboost.predict(xlle2_train)))
print ('accuracy_test：', accuracy_score(ylle2_test, xgboost.predict(xlle2_test)))

print ('Kappa_train：', cohen_kappa_score(ylle2_train, xgboost.predict(xlle2_train)))
print ('Kappa_test：', cohen_kappa_score(ylle2_test, xgboost.predict(xlle2_test)))
print ('F1_train：', f1_score(ylle2_train, xgboost.predict(xlle2_train),average='weighted'))
print ('F1_test：', f1_score(ylle2_test, xgboost.predict(xlle2_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle2_train, xgboost.predict(xlle2_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle2_test, xgboost.predict(xlle2_test)))


# In[289]:


xgboost.fit(xlle3_train, ylle3_train)
print ('accuracy_train：', accuracy_score(ylle3_train, xgboost.predict(xlle3_train)))
print ('accuracy_test：', accuracy_score(ylle3_test, xgboost.predict(xlle3_test)))

print ('Kappa_train：', cohen_kappa_score(ylle3_train, xgboost.predict(xlle3_train)))
print ('Kappa_test：', cohen_kappa_score(ylle3_test, xgboost.predict(xlle3_test)))
print ('F1_train：', f1_score(ylle3_train, xgboost.predict(xlle3_train),average='weighted'))
print ('F1_test：', f1_score(ylle3_test, xgboost.predict(xlle3_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle3_train, xgboost.predict(xlle3_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle3_test, xgboost.predict(xlle3_test)))


# In[240]:


# from these trying, I can find the xgboost has a good effect.
# ICA is good, so consider further for these two methods.
# first consider the dimension to choose.
# The get the mean of accuracy.
# I choose to consider range (400,900), and (400,2400) for the classify of four classes
# remove the mud class is a idea, but if we do so , we can just classify three classes, so if we can classify 4 classes well, we 
# have no need to consider this method.


# In[241]:


# PCA
pca=PCA(n_components=0.99)
X_Pca1=pca.fit_transform(wavelength1)
X_Pca2=pca.fit_transform(wavelength2)
X_Pca1.shape, X_Pca2.shape


# In[242]:


xgboost = XGBClassifier(
    n_estimators=50,
    learning_rate =0.3,
    max_depth=4,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.6,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=12,
    reg_lambda=1,
    seed=27,
    num_class= 4)


# In[232]:


from sklearn.metrics import accuracy_score
Scores=[]

for i in range(2,25):
    ICA = FastICA(n_components=i,random_state=100) 
    X_ICA=ICA.fit_transform(wavelength1)
    summary=0
    n=40
    for j in range(0,40):
        xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA,target, test_size=0.3)
        xgboost.fit(xIca_train, yIca_train)
        summary = accuracy_score(yIca_test, xgboost.predict(xIca_test))+summary
    a = summary/n
    Scores.append(a)  


# In[233]:


Scores


# In[234]:


N=array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
plt.plot(N,Scores)


# In[243]:


# so I can find the dimension can be choose as 23 , when the wavelength is in the range of (400nm,900nm).


# In[244]:


ICA1 = FastICA(n_components=23,random_state=100) 
X_ICA1=ICA1.fit_transform(wavelength1)
scores = cross_val_score(xgboost , X_ICA1,target, cv=30)
sum(scores)/30


# In[237]:


# another method to calculate the mean accuracy.


# In[238]:


ICA1 = FastICA(n_components=23,random_state=100) 
X_ICA1=ICA1.fit_transform(wavelength1)
summary=0
n=100
for j in range(0,100):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA1,target, test_size=0.3)
    xgboost.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, xgboost.predict(xIca_test))+summary
a = summary/n
a


# In[143]:


# Now can find the accuracy is around 0.95.


# In[145]:


from sklearn.metrics import accuracy_score
scores=[]

for i in range(2,25):
    ICA = FastICA(n_components=i,random_state=100) 
    X_ICA=ICA.fit_transform(wavelength2)
    summary=0
    n=50
    for j in range(0,50):
        xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA,target, test_size=0.3)
        xgboost.fit(xIca_train, yIca_train)
        summary = accuracy_score(yIca_test, xgboost.predict(xIca_test))+summary
    a = summary/n
    scores.append(a) 


# In[149]:


N=array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
plt.plot(N,Scores)


# In[96]:


# so I can find the dimension can be choose as 23 , when the wavelength is (400nm,2400nm).


# In[228]:


ICA2 = FastICA(n_components=23,random_state=100) 
X_ICA2=ICA2.fit_transform(wavelength2)
scores = cross_val_score(xgboost, X_ICA2,target, cv=30)
sum(scores)/30


# In[266]:


ICA2 = FastICA(n_components=23,random_state=100) 
X_ICA2=ICA2.fit_transform(wavelength2)
summary=0
n=100
for j in range(0,100):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA2,target, test_size=0.3)
    xgboost.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, xgboost.predict(xIca_test))+summary
b = summary/n
b


# In[154]:


# Now can find the accuracy is around 0.96.


# In[155]:


# Then calculate the time.


# In[156]:


import time
start =time.clock()

ICA1 = FastICA(n_components=23,random_state=100) 
X_ICA1=ICA1.fit_transform(wavelength1)
summary=0
n=50
for j in range(0,50):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA1,target, test_size=0.3)
    xgboost.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, xgboost.predict(xIca_test))+summary
a = summary/n

end=time.clock()

print('Running time: %s Seconds'%(end-start))


# In[157]:


import time
start =time.clock()

ICA2 = FastICA(n_components=23,random_state=100) 
X_ICA2=ICA2.fit_transform(wavelength2)
summary=0
n=50
for j in range(0,50):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA2,target, test_size=0.3)
    xgboost.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, xgboost.predict(xIca_test))+summary
b = summary/n

end=time.clock()

print('Running time: %s Seconds'%(end-start))


# In[158]:


# If also calculate the running time of GBDT using this method, we can find xgboost is much faster.


# In[159]:


import time
start =time.clock()

GradientBoosting1 = GradientBoostingClassifier(max_depth=3,n_estimators=50,learning_rate=0.05)
ICA1 = FastICA(n_components=23,random_state=100) 
X_ICA1=ICA1.fit_transform(wavelength1)
summary=0
n=50
for j in range(0,50):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA1,target, test_size=0.3)
    GradientBoosting1.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, GradientBoosting1.predict(xIca_test))+summary
a = summary/n

end=time.clock()

print('Running time: %s Seconds'%(end-start))


# In[160]:


import time
start =time.clock()

GradientBoosting1 = GradientBoostingClassifier(max_depth=3,n_estimators=50,learning_rate=0.05)
ICA2 = FastICA(n_components=23,random_state=100) 
X_ICA2=ICA2.fit_transform(wavelength2)
summary=0
n=50
for j in range(0,50):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA2,target, test_size=0.3)
    GradientBoosting1.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, GradientBoosting1.predict(xIca_test))+summary
b = summary/n
end=time.clock()

print('Running time: %s Seconds'%(end-start))


# In[161]:


# Now I can find the classification effect is quite good, around 0.96.
# and also find the suitable dimension.
# For all the work, I also find the the xgboost model a little overfitting, and the parameters can be improved.
# Now we consider this.


# In[162]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb


# In[163]:


# first consider the wavelength (400nm,900nm).
# first try to find the suitable value of n_estimators.


# In[164]:


xgboost = XGBClassifier(
    n_estimators=50,
    learning_rate =0.3,
    max_depth=4,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.6,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=12,
    reg_lambda=1,
    seed=27,
    num_class= 4)


# In[189]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength1)

cv_params = {'n_estimators': [30, 40, 50, 60, 70, 80, 90, 100]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.2, "subsample": 0.6, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 50, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[166]:


# Then lets think deeper for the n_estimators.


# In[190]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength1)

cv_params = {'n_estimators': [64,66,68,70,72,74,76]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.2, "subsample": 0.6, "colsample_bytree": 0.8,
                 "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 70, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[168]:


# # so now I can choose the n_estimator = 70.
# The we try to estimate the suitable value of max_depth and min_child_weight.


# In[191]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength1)

cv_params = {'max_depth': [2, 3, 4, 5, 6, 7], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.2, "subsample": 0.6, "colsample_bytree": 0.8,
                 "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 70, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[115]:


# so now I can choose the max_depth = 4, the min_child_weight = 1.
# Then we try to estimate the best value of gamma.


# In[192]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength1)

cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.2, "subsample": 0.6, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 70, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[117]:


# Now let us consider further.


# In[195]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength1)

cv_params = {'gamma': [0.06,0.08,0.1,0.12,0.14]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.1, "subsample": 0.6, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 70, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[173]:


# so now we can choose gamma = 0.08.
# Then we can try to estimate subsample and colsample_bytree.


# In[196]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength1)

cv_params = {'subsample': [0.4,0.5,0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.08, "subsample": 0.6, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 70, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[197]:


# Then we can choose subsample = 0.4 , and colsample_bytree = 0.7.
# Next step we try to estimate reg_lambda.


# In[198]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength1)

cv_params = {'reg_lambda': [0.05, 0.1, 0.5, 0.8, 1, 1.2, 1.5, 2, 3]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.08, "subsample": 0.4, "colsample_bytree": 0.7,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 70, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[177]:


# Now the suitablle value of reg_lambda is about 1.
# Next is the last step and I should estimate the suitable value of learning_rate.


# In[201]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength1)

cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.08, "subsample": 0.4, "colsample_bytree": 0.7,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 70, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[202]:


# so the learning_rate = 0.3 is a good choice.
# Now here is the new classifier model.
xgboost = XGBClassifier(
    n_estimators=70,
    learning_rate =0.3,
    max_depth=4,
    min_child_weight=1,
    gamma=0.08,
    subsample=0.4,
    colsample_bytree=0.7,
    objective= 'multi:softprob',
    nthread=12,
    reg_lambda=1,
    seed=27,
    num_class= 4)

ICA1 = FastICA(n_components=23,random_state=100) 
X_ICA1=ICA1.fit_transform(wavelength1)


# In[205]:


# So I can find the accuracy just improved a little, around 0.96.
# Now we can see the accuracy, kappa again, in addition we can also see the confuse matrix.


# In[226]:


from sklearn.metrics import f1_score
xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA,target, test_size=0.3)
xgboost.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, xgboost.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, xgboost.predict(xIca_test)))
print ('Kappa_train：', cohen_kappa_score(yIca_train, xgboost.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, xgboost.predict(xIca_test)))
print ('F1_train：', f1_score(yIca_train, xgboost.predict(xIca_train),average='weighted'))
print ('F1_test：', f1_score(yIca_test, xgboost.predict(xIca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, xgboost.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, xgboost.predict(xIca_test)))


# In[208]:


# Now consider the other wavelength range.


# In[230]:


Xgboost = XGBClassifier(
    n_estimators=50,
    learning_rate =0.3,
    max_depth=4,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.6,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=12,
    reg_lambda=1,
    seed=27,
    num_class= 4)


# In[231]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength2)

cv_params = {'n_estimators': [20, 30, 40, 50, 60,70 ,80, 90, 100]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.2, "subsample": 0.6, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 50, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[139]:


# Then think further for n_estimators.


# In[232]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength2)

cv_params = {'n_estimators': [54,56,58,60,62,64,66]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.2, "subsample": 0.6, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 60, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[141]:


# # so now I can choose the n_estimator = 60.
# The we try to estimate the suitable value of max_depth and min_child_weight.


# In[233]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength2)

cv_params = {'max_depth': [2, 3, 4, 5, 6, 7], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.2, "subsample": 0.6, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 60, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[143]:


# so now I can choose the max_depth = 2, the min_child_weight = 3.
# just a little improvement, so maybe no need to change
# Then we try to estimate the best value of gamma.


# In[234]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength2)

cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.2, "subsample": 0.6, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 60, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[146]:


# Then consider further for gamma.


# In[235]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength2)

cv_params = {'gamma': [0.14,0.16,0.18,0.2,0.22,0.24,0.26]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.2, "subsample": 0.6, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 60, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[147]:


# so now we can choose gamma = 0.18.
# Then we can try to estimate subsample and colsample_bytree.


# In[236]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength2)

cv_params = {'subsample': [0.4,0.5,0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.18, "subsample": 0.6, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 60, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[150]:


# Then we can choose subsample = 0.5 , and colsample_bytree = 0.7 .
# Next step we try to estimate reg_lambda.


# In[237]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength2)

cv_params = {'reg_lambda': [0.05, 0.1, 0.5, 0.8, 1, 1.2, 1.5, 2, 3]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.18, "subsample": 0.5, "colsample_bytree": 0.7,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 60, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[238]:


# Now the suitablle value of reg_lambda is about 1.
# Next is the last step and I should estimate the suitable value of learning_rate.


# In[239]:


ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength2)

cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
other_params = {"learning_rate": 0.3, "max_depth": 4, "min_child_weight": 1, "gamma": 0.18, "subsample": 0.5, "colsample_bytree": 0.7,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 60, "num_class" : 4}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=30)
optimized_GBM.fit(X_ICA, target)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[258]:


# so the learning_rate = 0.3 is a good choice.
# Now here is the new classifier model.
Xgboost = XGBClassifier(
    n_estimators=60,
    learning_rate =0.3,
    max_depth=4,
    min_child_weight=1,
    gamma=0.18,
    subsample=0.5,
    colsample_bytree=0.7,
    objective= 'multi:softprob',
    nthread=12,
    reg_lambda=1,
    seed=27,
    num_class= 4)

ICA = FastICA(n_components=23,random_state=100) 
X_ICA=ICA.fit_transform(wavelength2)


# In[263]:


# So I can find the accuracy just improved a little, around 0.96.
# Now we can see the accuracy, kappa again, in addition we can also see the confuse matrix and F1 score.


# In[264]:


from sklearn.metrics import f1_score
xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA,target, test_size=0.3)
Xgboost.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, Xgboost.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, Xgboost.predict(xIca_test)))
print ('Kappa_train：', cohen_kappa_score(yIca_train, Xgboost.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, Xgboost.predict(xIca_test)))
print ('F1_train：', f1_score(yIca_train, Xgboost.predict(xIca_train),average='weighted'))
print ('F1_test：', f1_score(yIca_test, Xgboost.predict(xIca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, Xgboost.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, Xgboost.predict(xIca_test)))


# In[225]:


# then we can also see how the tree is. Just as the coffee data code.
# now we don't plot the tree here.


# In[ ]:




