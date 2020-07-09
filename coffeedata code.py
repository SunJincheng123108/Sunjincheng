#!/usr/bin/env python
# coding: utf-8

# In[183]:


# coffee data summary


# In[2]:


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


# In[3]:


# here we read in the data


# In[4]:


Y=np.loadtxt('Reflection2_00001.txt').T[0,]
np.savetxt('Class',Y)
Y=np.loadtxt('Class')
X=pd.read_csv('Coffee.csv')
X=X[40:890]
Y[300],Y[1600]


# In[5]:


# then for x1-x9, we calculate the mean and the sigma=variance^0.5 in each group(class1,class2,...class9, but class6, class7 are empty)
# then we plot charts to see them


# In[6]:


x=np.std(X,axis=0)
x_mean=np.mean(X,axis=0)
X_1=np.vstack((X[120:180],X[190:220],X[520:560],X[570:640]))
x1=np.std(X_1,axis=0)
x1_mean=np.mean(X_1,axis=0)
X_2=np.vstack((X[220:260],X[290:360],X[400:490]))
x2=np.std(X_2,axis=0)
x2_mean=np.mean(X_2,axis=0)
X_3=np.vstack((X[260:290],X[360:400],X[490:520]))
x3=np.std(X_3,axis=0)
x3_mean=np.mean(X_3,axis=0)
X_4=np.vstack((X[640:760],X[770:800]))
x4=np.std(X_4,axis=0)
x4_mean=np.mean(X_4,axis=0)
X_5=X[800:850]
x5=np.std(X_5,axis=0)
x5_mean=np.mean(X_5,axis=0)
X_8=X[10:30]
x8=np.std(X_8,axis=0)
x8_mean=np.mean(X_8,axis=0)
X_9=X[40:50]
x9=np.std(X_9,axis=0)
x9_mean=np.mean(X_9,axis=0)


# In[7]:


# first for all the samples,and the samples in each class,let us see the sigma,we can find the data of begining and the end 
# are very large. these data must have some problems(too noisy), so wavelength less than 200nm and wavelength larger than 1000nm 
# are quite bad,so I must remove them.


# In[8]:


plt.plot(Y[23:2030],x[23:2030],color="black")


# In[9]:


plt.plot(Y[30:2030],x1[30:2030],color="red")
plt.plot(Y[30:2030],x2[30:2030],color="orange")
plt.plot(Y[30:2030],x3[30:2030],color="yellow")
plt.plot(Y[30:2030],x4[30:2030],color="green")
plt.plot(Y[30:2030],x5[30:2030],color="blue")
plt.plot(Y[30:2030],x8[30:2030],color="purple")
plt.plot(Y[30:2030],x9[30:2030],color="brown")


# In[10]:


# then we see the mean of all the samples and the samples in each classes, in different wavelength.
# we can find in some wavelength, 200-300nm and 900-1000nm, although the value under almost the same wavelength, for example,
# 201nm and 201.5nm, the value should be almost the same. But for the plot, we can see the at almost the same wavelength,
# the value we get can be a little different. So I think the data here is noisy too, this noisy is mainly for the wavelength.
# In order to get a better result, I think I can remove these data.
# in a word, I think the wavelength around 300nm-900nm can be a good chioce, now I just choose the wavelength roughly in 325nm-865nm.
# (so we can keep one thousand and three hundred dimensions)


# In[11]:


plt.plot(Y[22:2050],x_mean[22:2050],color="black")


# In[12]:


plt.plot(Y[30:2030],x1_mean[30:2030],color="red")
plt.plot(Y[30:2030],x2_mean[30:2030],color="orange")
plt.plot(Y[30:2030],x3_mean[30:2030],color="yellow")
plt.plot(Y[30:2030],x4_mean[30:2030],color="green")
plt.plot(Y[30:2030],x5_mean[30:2030],color="blue")
plt.plot(Y[30:2030],x8_mean[30:2030],color="purple")
plt.plot(Y[30:2030],x9_mean[30:2030],color="brown")


# In[13]:


x=X.T
X_n=x[300:1600].T
X_n.shape


# In[14]:


# first we should reduce the dimension, it have some benefits:
# make the codes runs faster, reduce the noisy and most of the time can improve the classifier effect.
# so I think reduce the dimension(deal with features) is a necessary step.

# since there are some unlabeled data, so LDA is not very suitable. LDA is used for the labeled data.
# So I choose to use unsupervised learning to reduce dimension.
# By the unsupervised learning methods, we can also mark the unlabeled data using the model we find in our work.


# In[15]:


# PCA and ICA are two linear methods, PCA is a very common method and ICA is a improvement of it, so I consider both of them.
# for the choose of dimension, we just first choose 10, although it may not the best, it can be a nice try and I also improve this 
# in our later work.(in fact, after 10 dimension, the model effect just have a little change by our trying)
# the choose of dimension can just have a little improvement after 10, so we can considr to improve this after we find the best model.


# In[16]:


pca=PCA(n_components=0.99)
X_PCA=pca.fit_transform(X_n)
ICA = FastICA(n_components=10,random_state=40) 
X_ICA=ICA.fit_transform(X_n)
X_PCA.shape


# In[17]:


# t-SNE and lle are nonlinear methods, if the data are noisy in a range, the lle can keep the most characteristics, so in this 
# condithion, PCA and ICA is not good, we can use lle.
# but here we have remove the noisy data, the lle may loss its benefits.


# In[18]:


X_tsne = TSNE(n_components=3, n_iter=500).fit_transform(X_n)
lle=LocallyLinearEmbedding(n_components=10,n_neighbors=40)
X_lle=lle.fit_transform(X_n)


# In[19]:


# here we set the target.


# In[20]:


from numpy import array


# In[21]:


# first use 1,2,3,...to present the different class:
# "1": health good
# "2": rust
# "3": infected green canopy
# "4": geisha
# "5": geisha cannopy
# "6": yellow
# "7": yellow green
# "8": Young_Rust_Yellow
# "9": Young_Rust_Green_Leaf

target_labeled=array(["8","8","8","8","8","8","8","8","8","8","8","8","8","8","8","8","8","8","8","8","9","9","9","9","9","9","9","9",
                      "9","9","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","2","2","2","2","2","2","2","2","2","2",
             "2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2",
             "3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3",
             "2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2",
             "2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2",
             "2","2","2","2","2","2","2","2","2","2","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3",
             "3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","2","2","2","2","2","2","2","2","2","2",
             "2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2",
             "2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2",
             "2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","2","3","3","3","3","3","3","3","3","3","3",
             "3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","3","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","4","4","4","4","4","4","4","4","4","4",
             "4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4",
             "4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4",
             "4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4",
             "4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4",
             "4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4",
             "5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5",
             "5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5"])
len(target_labeled)


# In[22]:


np.savetxt('New_X',X_n)
np.savetxt("PCA",X_PCA)
np.savetxt("ICA",X_ICA)
np.savetxt("LLE",X_lle)
np.savetxt('TSNE',X_tsne)
X_new=np.loadtxt('New_X')
X_Pca=np.loadtxt("PCA")
X_ICA=np.loadtxt("ICA")
X_lle=np.loadtxt("LLE")
X_tsne=np.loadtxt("TSNE")


# In[23]:


# reset the data for the labeled and unlabeled data


# In[24]:


X_tsne_labeled=np.vstack((X_tsne[10:30],X_tsne[40:50],X_tsne[120:180],X_tsne[190:560],X_tsne[570:760],X_tsne[770:850]))
X_tsne_unlabeled=np.vstack((X_tsne[0:10],X_tsne[30:40],X_tsne[50:120],X_tsne[180:190],X_tsne[560:570],X_tsne[760:770]))
X_Pca_labeled=np.vstack((X_Pca[10:30],X_Pca[40:50],X_Pca[120:180],X_Pca[190:560],X_Pca[570:760],X_Pca[770:850]))
X_Pca_unlabeled=np.vstack((X_Pca[0:10],X_Pca[30:40],X_Pca[50:120],X_Pca[180:190],X_Pca[560:570],X_Pca[760:770]))
X_ICA_labeled=np.vstack((X_ICA[10:30],X_ICA[40:50],X_ICA[120:180],X_ICA[190:560],X_ICA[570:760],X_ICA[770:850]))
X_ICA_unlabeled=np.vstack((X_ICA[0:10],X_ICA[30:40],X_ICA[50:120],X_ICA[180:190],X_ICA[560:570],X_ICA[760:770]))
X_lle_labeled=np.vstack((X_lle[10:30],X_lle[40:50],X_lle[120:180],X_lle[190:560],X_lle[570:760],X_lle[770:850]))
X_lle_unlabeled=np.vstack((X_lle[0:10],X_lle[30:40],X_lle[50:120],X_lle[180:190],X_lle[560:570],X_lle[760:770]))
X_labeled=np.vstack((X_new[10:30],X_new[40:50],X_new[120:180],X_new[190:560],X_new[570:760],X_new[770:850]))
X_unlabeled=np.vstack((X_new[0:10],X_new[30:40],X_new[50:120],X_new[180:190],X_new[560:570],X_new[760:770]))

X.shape,X_labeled.shape


# In[25]:


# in this part just see roughly the classification effect, split the train and test data
x_train, x_test, y_train, y_test = train_test_split(X_labeled,target_labeled, test_size=0.3)
xpca_train, xpca_test, ypca_train, ypca_test = train_test_split(X_Pca_labeled,target_labeled, test_size=0.3)
xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
xlle_train, xlle_test, ylle_train, ylle_test = train_test_split(X_lle_labeled,target_labeled, test_size=0.3)
xtsne_train, xtsne_test, ytsne_train, ytsne_test = train_test_split(X_tsne_labeled,target_labeled, test_size=0.3)


# In[26]:


from sklearn import svm
from sklearn.svm import SVC
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


# In[184]:


# accuracy= predict right/ all samples.
# kappa and F1 score are used to estimate the accuracy in each class.


# In[27]:


# there are some kinds of machine learning models, like, netural network, tree model(can be improved by ensemble), svm model, naivebayes,
# knn(unsupervised machine learning, classify the unlabeled samples, not suitable for my data), logistic regression clasifier( good but
# this model should classifier the positive data),but after reduce the dimension, the data can be negative, so the logistic regression 
# classifier may not a good choice.
# other models are just the improvement based on these models, like adding boosting or bagging.
# so we just cinsider four main kinds of models: svm, decision tree, naive bayes,netural network.


# In[28]:


# first let us see the naive bayes model


# In[29]:


gnb = GaussianNB()
gnb.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, gnb.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, gnb.predict(x_test)))

print ('Kappa_train：', cohen_kappa_score(y_train, gnb.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, gnb.predict(x_test)))
print ('F1_train：', f1_score(y_train, gnb.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, gnb.predict(x_test),average='weighted'))


# In[30]:


mnb = MultinomialNB()
mnb.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, mnb.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, mnb.predict(x_test)))

print ('Kappa_train：', cohen_kappa_score(y_train, mnb.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, mnb.predict(x_test)))
print ('F1_train：', f1_score(y_train, mnb.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, mnb.predict(x_test),average='weighted'))


# In[31]:


from sklearn.naive_bayes import ComplementNB
cnb = ComplementNB()
cnb.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, cnb.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, cnb.predict(x_test)))

print ('Kappa_train：', cohen_kappa_score(y_train, cnb.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, cnb.predict(x_test)))
print ('F1_train：', f1_score(y_train, cnb.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, cnb.predict(x_test),average='weighted'))


# In[32]:


# for other two kinds of naive bayes model, the data should be postive, only the GaussianNB can be used after reducing dimension.


# In[33]:


gnb = GaussianNB()
gnb.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, gnb.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, gnb.predict(xpca_test)))

print ('Kappa_train：', cohen_kappa_score(ypca_train, gnb.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, gnb.predict(xpca_test)))
print ('F1_train：', f1_score(ypca_train, gnb.predict(xpca_train),average='weighted'))
print ('F1_test：', f1_score(ypca_test, gnb.predict(xpca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, gnb.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, gnb.predict(xpca_test)))


# In[34]:


gnb = GaussianNB()
gnb.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, gnb.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, gnb.predict(xIca_test)))

print ('Kappa_train：', cohen_kappa_score(yIca_train, gnb.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, gnb.predict(xIca_test)))
print ('F1_train：', f1_score(yIca_train, gnb.predict(xIca_train),average='weighted'))
print ('F1_test：', f1_score(yIca_test, gnb.predict(xIca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, gnb.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, gnb.predict(xIca_test)))


# In[35]:


gnb = GaussianNB()
gnb.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, gnb.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, gnb.predict(xlle_test)))

print ('Kappa_train：', cohen_kappa_score(ylle_train, gnb.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, gnb.predict(xlle_test)))
print ('F1_train：', f1_score(ylle_train, gnb.predict(xlle_train),average='weighted'))
print ('F1_test：', f1_score(ylle_test, gnb.predict(xlle_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, gnb.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, gnb.predict(xlle_test)))


# In[36]:


# we can find the reducing dimension is a quite good method, then try to improve this model.
# we can also find this model seems not good nough, the bagging is a way to improvement overfitting, but we can find the naive bayes 
# model is not overfitting, so I think we can considr boosting, which is a method to improve the model acuracy.


# In[37]:


# try to adding boost method, boost is used for adding many models, the good models will have a higher weight, and the models have a 
# bad effect may have a light weight, so using this method our model can be better.
# then let us the different methods of reducing dimension + naive bayes model.


# In[38]:


bdt = AdaBoostClassifier(GaussianNB(),algorithm="SAMME",n_estimators=500, learning_rate=0.6)


# In[39]:


bdt.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, bdt.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, bdt.predict(x_test)))

print ('Kappa_train：', cohen_kappa_score(y_train, bdt.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, bdt.predict(x_test)))
print ('F1_train：', f1_score(y_train, bdt.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, bdt.predict(x_test),average='weighted'))


# In[40]:


bdt.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, bdt.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, bdt.predict(xpca_test)))

print ('Kappa_train：', cohen_kappa_score(ypca_train, bdt.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, bdt.predict(xpca_test)))
print ('F1_train：', f1_score(ypca_train, bdt.predict(xpca_train),average='weighted'))
print ('F1_test：', f1_score(ypca_test, bdt.predict(xpca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, bdt.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, bdt.predict(xpca_test)))


# In[41]:


bdt.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, bdt.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, bdt.predict(xIca_test)))

print ('Kappa_train：', cohen_kappa_score(yIca_train, bdt.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, bdt.predict(xIca_test)))
print ('F1_train：', f1_score(yIca_train, bdt.predict(xIca_train),average='weighted'))
print ('F1_test：', f1_score(yIca_test, bdt.predict(xIca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, bdt.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, bdt.predict(xIca_test)))


# In[185]:


# now we find for naive bays model, boosting +ICA + baive bayes model, is the best.
# then we consider to rpeat this method to calculate a average accuracy.


# In[42]:


summary=0
n=40
for j in range(0,40):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
    bdt.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, bdt.predict(xIca_test))+summary
summary/n


# In[43]:


bdt.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, bdt.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, bdt.predict(xlle_test)))

print ('Kappa_train：', cohen_kappa_score(ylle_train, bdt.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, bdt.predict(xlle_test)))
print ('F1_train：', f1_score(ylle_train, bdt.predict(xlle_train),average='weighted'))
print ('F1_test：', f1_score(ylle_test, bdt.predict(xlle_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, bdt.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, bdt.predict(xlle_test)))


# In[44]:


# here we can find the booting + ICA + naive bayes can have a good effect.


# In[45]:


# then see SVM model.


# In[46]:


SVM= SVC(kernel='rbf')
# when choose kernel, I find linear and sigmod is bad, so I choose rbf, which is often used.
# for gussian kernel (rbf), there are C and gamma to choose
# I first limit their range, then find the good combination roughly.

# for svm model, the kernel can be any styles, but too difficult to find a good one, we often choose gusssian kernel(but it may not best)
# then the range of gamma is also difficult to deal with, gamma = 1/(2*sigma^2), so if gamma is large, this model will be meaningless,
# (can only explain the sample itself) what I want to say is just the gamma is difficult to set.
# so this model may difficult to design.
 
# ICA and LLE not a good choice here, since the data is very small after transfermation, so the sigma should be small, the the gamma 
# can be large, but it can be difficult to set a good range(if small, effect will be bad, if large, very easy to loss meaning)
# so here we just consider two kinds of methods to reduce dimensions.


# In[47]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-3,1,40),gamma=np.logspace(-3,1,40))
clf = RandomizedSearchCV(SVM, distributions, random_state=42)
search1 = clf.fit(x_train, y_train)
search1.best_params_ 


# In[48]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-3,1,50),gamma=np.logspace(-3,1,50))
clf = RandomizedSearchCV(SVM, distributions, random_state=50)
search2 = clf.fit(xpca_train, ypca_train)
search2.best_params_ 


# In[49]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-3,1,50),gamma=np.logspace(-3,1,50))
clf = RandomizedSearchCV(SVM, distributions, random_state=50)
search4 = clf.fit(xtsne_train, ytsne_train)
search4.best_params_ 


# In[50]:


clf1= SVC(C=1,kernel='rbf',gamma=0.004)
clf1.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, clf1.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, clf1.predict(x_test)))

print ('Kappa_train：', cohen_kappa_score(y_train, clf1.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, clf1.predict(x_test)))
print ('F1_train：', f1_score(y_train, clf1.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, clf1.predict(x_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, clf1.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, clf1.predict(x_test)))


# In[51]:


# Then find may overfitting, so try to use bagging to improve my model.


# In[52]:


from sklearn.ensemble import BaggingClassifier
clf1= SVC(C=1,kernel='rbf',gamma=0.004)
bagging1 = BaggingClassifier(clf1,max_samples=0.6, max_features=0.6)
bagging1.fit(x_train,y_train)
print ('accuracy_train：', accuracy_score(y_train, bagging1.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, bagging1.predict(x_test)))
print ('Kappa_train：', cohen_kappa_score(y_train, bagging1.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, bagging1.predict(x_test)))
print ('F1_train：', f1_score(y_train, bagging1.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, bagging1.predict(x_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, bagging1.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, bagging1.predict(x_test)))


# In[53]:


# then we can find the problem of overfitting can be improved, but the model effect is just so so.
# so we consider to reduce the dimension now.


# In[54]:


clf2= SVC(C=3,kernel='rbf',gamma=0.004)
clf2.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, clf2.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, clf2.predict(xpca_test)))
print ('Kappa_train：', cohen_kappa_score(ypca_train, clf2.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, clf2.predict(xpca_test)))
print ('F1_train：', f1_score(ypca_train, clf2.predict(xpca_train),average='weighted'))
print ('F1_test：', f1_score(ypca_test, clf2.predict(xpca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, clf2.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, clf2.predict(xpca_test)))


# In[55]:


# adding bagging to improve the model too.


# In[56]:


from sklearn.ensemble import BaggingClassifier
clf2= SVC(C=3,kernel='rbf',gamma=0.004)
bagging2 = BaggingClassifier(clf2,max_samples=0.6, max_features=0.6)
bagging2.fit(xpca_train,ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, bagging2.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, bagging2.predict(xpca_test)))
print ('Kappa_train：', cohen_kappa_score(ypca_train, bagging2.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, bagging2.predict(xpca_test)))
print ('F1_train：', f1_score(ypca_train, bagging2.predict(xpca_train),average='weighted'))
print ('F1_test：', f1_score(ypca_test, bagging2.predict(xpca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, bagging2.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, bagging2.predict(xpca_test)))


# In[57]:


clf3= SVC(C=4.7,kernel='rbf',gamma=0.16)
clf3.fit(xtsne_train, ytsne_train)
print ('accuracy_train：', accuracy_score(ytsne_train, clf3.predict(xtsne_train)))
print ('accuracy_test：', accuracy_score(ytsne_test, clf3.predict(xtsne_test)))
print ('Kappa_train：', cohen_kappa_score(ytsne_train, clf3.predict(xtsne_train)))
print ('Kappa_test：', cohen_kappa_score(ytsne_test, clf3.predict(xtsne_test)))
print ('F1_train：', f1_score(ytsne_train, clf3.predict(xtsne_train),average='weighted'))
print ('F1_test：', f1_score(ytsne_test, clf3.predict(xtsne_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ytsne_train, clf3.predict(xtsne_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ytsne_test, clf3.predict(xtsne_test)))


# In[58]:


from sklearn.ensemble import BaggingClassifier
clf3= SVC(C=4.7,kernel='rbf',gamma=0.2)
bagging3 = BaggingClassifier(clf3,max_samples=0.6, max_features=0.8)
bagging3.fit(xtsne_train,ytsne_train)
print ('accuracy_train：', accuracy_score(ytsne_train, bagging3.predict(xtsne_train)))
print ('accuracy_test：', accuracy_score(ytsne_test, bagging3.predict(xtsne_test)))
print ('Kappa_train：', cohen_kappa_score(ytsne_train, bagging3.predict(xtsne_train)))
print ('Kappa_test：', cohen_kappa_score(ytsne_test, bagging3.predict(xtsne_test)))
print ('F1_train：', f1_score(ytsne_train, bagging3.predict(xtsne_train),average='weighted'))
print ('F1_test：', f1_score(ytsne_test, bagging3.predict(xtsne_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ytsne_train, bagging3.predict(xtsne_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ytsne_test, bagging3.predict(xtsne_test)))


# In[59]:


# for both ICA and LLE method, the test data effect is better than train data, so I think these methods to reduce the dimension is not 
# good choice, I removed them
# in a word, we can find the t-SNE + svm model can be a good choice, the effect is a little better than naive bayes model.


# In[60]:


# Then we consider the tree models, to avoid overfitting, roughly set a depth.
# consider different dimension reduction method, pca, Ica, lle.


# In[61]:


from sklearn import tree
clf_tree = tree.DecisionTreeClassifier(max_depth=12)
clf_tree1 = tree.DecisionTreeClassifier(max_depth=7)
clf_tree2 = tree.DecisionTreeClassifier(max_depth=9)


# In[62]:


clf_tree.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, clf_tree.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, clf_tree.predict(x_test)))
print ('Kappa_train：', cohen_kappa_score(y_train, clf_tree.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, clf_tree.predict(x_test)))
print ('F1_train：', f1_score(y_train, clf_tree.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, clf_tree.predict(x_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, clf_tree.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, clf_tree.predict(x_test)))


# In[63]:


clf_tree2.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, clf_tree2.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, clf_tree2.predict(xpca_test)))
print ('Kappa_train：', cohen_kappa_score(ypca_train, clf_tree2.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, clf_tree2.predict(xpca_test)))
print ('F1_train：', f1_score(ypca_train, clf_tree2.predict(xpca_train),average='weighted'))
print ('F1_test：', f1_score(ypca_test, clf_tree2.predict(xpca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, clf_tree2.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, clf_tree2.predict(xpca_test)))


# In[64]:


clf_tree2.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, clf_tree2.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, clf_tree2.predict(xIca_test)))

print ('Kappa_train：', cohen_kappa_score(yIca_train, clf_tree2.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, clf_tree2.predict(xIca_test)))
print ('F1_train：', f1_score(yIca_train, clf_tree2.predict(xIca_train),average='weighted'))
print ('F1_test：', f1_score(yIca_test, clf_tree2.predict(xIca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, clf_tree2.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, clf_tree2.predict(xIca_test)))


# In[65]:


clf_tree2.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, clf_tree2.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, clf_tree2.predict(xlle_test)))

print ('Kappa_train：', cohen_kappa_score(ylle_train, clf_tree2.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, clf_tree2.predict(xlle_test)))
print ('F1_train：', f1_score(ylle_train, clf_tree2.predict(xlle_train),average='weighted'))
print ('F1_test：', f1_score(ylle_test, clf_tree2.predict(xlle_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, clf_tree2.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, clf_tree2.predict(xlle_test)))


# In[66]:


# the train accuracy is very high but the test accuracy is not goog enough, so it may be overfitting.
# now we can find the ICA + decision tree is a good choice, but we can also consider to improve this model.


# In[67]:


# we can think adding boosting to the tree to improve the model, boosting is a good way to improve model.


# In[68]:


from sklearn.ensemble import AdaBoostClassifier


# In[69]:


bdt1 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=12),algorithm="SAMME",n_estimators=1000, learning_rate=0.6)


# In[70]:


bdt1.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, bdt1.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, bdt1.predict(x_test)))
print ('Kappa_train：', cohen_kappa_score(y_train, bdt1.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, bdt1.predict(x_test)))
print ('F1_train：', f1_score(y_train, bdt1.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, bdt1.predict(x_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, bdt1.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, bdt1.predict(x_test)))


# In[71]:


bdt2 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=7),algorithm="SAMME",n_estimators=20, learning_rate=0.1)


# In[72]:


bdt2.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, bdt2.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, bdt2.predict(xpca_test)))
print ('Kappa_train：', cohen_kappa_score(ypca_train, bdt2.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, bdt2.predict(xpca_test)))
print ('F1_train：', f1_score(ypca_train, bdt2.predict(xpca_train),average='weighted'))
print ('F1_test：', f1_score(ypca_test, bdt2.predict(xpca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, bdt2.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, bdt2.predict(xpca_test)))


# In[73]:


bdt3 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=7),algorithm="SAMME",n_estimators=10, learning_rate=0.1)


# In[74]:


bdt3.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, bdt3.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, bdt3.predict(xIca_test)))
print ('Kappa_train：', cohen_kappa_score(yIca_train, bdt3.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, bdt3.predict(xIca_test)))
print ('F1_train：', f1_score(yIca_train, bdt3.predict(xIca_train),average='weighted'))
print ('F1_test：', f1_score(yIca_test, bdt3.predict(xIca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, bdt3.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, bdt3.predict(xIca_test)))


# In[186]:


# calculate the time, very fast.


# In[75]:


import time
start =time.clock()

bdt3.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, bdt3.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, bdt3.predict(xIca_test)))
print ('Kappa_train：', cohen_kappa_score(yIca_train, bdt3.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, bdt3.predict(xIca_test)))

end=time.clock()
print('Running time: %s Seconds'%(end-start))


# In[76]:


bdt3.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, bdt3.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, bdt3.predict(xlle_test)))
print ('Kappa_train：', cohen_kappa_score(ylle_train, bdt3.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, bdt3.predict(xlle_test)))
print ('F1_train：', f1_score(ylle_train, bdt3.predict(xlle_train),average='weighted'))
print ('F1_test：', f1_score(ylle_test, bdt3.predict(xlle_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, bdt3.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, bdt3.predict(xlle_test)))


# In[77]:


# now we can find boost + ICA + decision tree is quite good, but to prove this I think I should try for some times, so let us consider
# further for this method.


# In[78]:


summary=0
n=40
for j in range(0,40):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
    bdt3.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, bdt3.predict(xIca_test))+summary
summary/n


# In[79]:


# so now we can find this method is around 0.955, a quite nice model.


# In[80]:


# then we consier bagging to improve decision tree( avoid overfiffing), so now we consider random forest model.
# for random forest model, also set a max_depth to avoid overfitting


# In[81]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[82]:


randomforest = RandomForestClassifier(max_depth=9)


# In[83]:


randomforest.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, randomforest.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, randomforest.predict(x_test)))
print ('Kappa_train：', cohen_kappa_score(y_train, randomforest.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, randomforest.predict(x_test)))
print ('F1_train：', f1_score(y_train, randomforest.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, randomforest.predict(x_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, randomforest.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, randomforest.predict(x_test)))


# In[84]:


randomforest.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, randomforest.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, randomforest.predict(xpca_test)))
print ('Kappa_train：', cohen_kappa_score(ypca_train, randomforest.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, randomforest.predict(xpca_test)))
print ('F1_train：', f1_score(ypca_train, randomforest.predict(xpca_train),average='weighted'))
print ('F1_test：', f1_score(ypca_test, randomforest.predict(xpca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, randomforest.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, randomforest.predict(xpca_test)))


# In[85]:


randomforest.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, randomforest.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, randomforest.predict(xIca_test)))
print ('Kappa_train：', cohen_kappa_score(yIca_train, randomforest.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, randomforest.predict(xIca_test)))
print ('F1_train：', f1_score(yIca_train, randomforest.predict(xIca_train),average='weighted'))
print ('F1_test：', f1_score(yIca_test, randomforest.predict(xIca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, randomforest.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, randomforest.predict(xIca_test)))


# In[187]:


# calculate the time, also very fast.


# In[86]:


import time
start =time.clock()

randomforest.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, randomforest.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, randomforest.predict(xIca_test)))
print ('Kappa_train：', cohen_kappa_score(yIca_train, randomforest.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, randomforest.predict(xIca_test)))

end=time.clock()
print('Running time: %s Seconds'%(end-start))


# In[87]:


randomforest.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, randomforest.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, randomforest.predict(xlle_test)))
print ('Kappa_train：', cohen_kappa_score(ylle_train, randomforest.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, randomforest.predict(xlle_test)))
print ('F1_train：', f1_score(ylle_train, randomforest.predict(xlle_train),average='weighted'))
print ('F1_test：', f1_score(ylle_test, randomforest.predict(xlle_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, randomforest.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, randomforest.predict(xlle_test)))


# In[88]:


# here we can find the ICA + random forest is good, now let us consider to repeat this to get a average data.


# In[89]:


summary=0
n=50
for j in range(0,50):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
    randomforest.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, randomforest.predict(xIca_test))+summary
summary/n


# In[90]:


# now we an find the accuracy can reach 0.965, even a little better than boost decision tree.


# In[91]:


# Then consider the GBDT, this is also a boost decision tree model, but have some improvement in althroghm.
# it has a more clear aim.


# In[92]:


from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
GradientBoosting1 = GradientBoostingClassifier(max_depth=3,n_estimators=60,learning_rate=0.05)


# In[93]:


GradientBoosting1.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, GradientBoosting1.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, GradientBoosting1.predict(x_test)))
print ('Kappa_train：', cohen_kappa_score(y_train, GradientBoosting1.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, GradientBoosting1.predict(x_test)))
print ('F1_train：', f1_score(y_train, GradientBoosting1.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, GradientBoosting1.predict(x_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, GradientBoosting1.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, GradientBoosting1.predict(x_test)))


# In[94]:


GradientBoosting1.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, GradientBoosting1.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, GradientBoosting1.predict(xpca_test)))
print ('Kappa_train：', cohen_kappa_score(ypca_train, GradientBoosting1.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, GradientBoosting1.predict(xpca_test)))
print ('F1_train：', f1_score(ypca_train, GradientBoosting1.predict(xpca_train),average='weighted'))
print ('F1_test：', f1_score(ypca_test, GradientBoosting1.predict(xpca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, GradientBoosting1.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, GradientBoosting1.predict(xpca_test)))


# In[95]:


GradientBoosting1.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, GradientBoosting1.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, GradientBoosting1.predict(xIca_test)))

print ('Kappa_train：', cohen_kappa_score(yIca_train, GradientBoosting1.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, GradientBoosting1.predict(xIca_test)))
print ('F1_train：', f1_score(yIca_train, GradientBoosting1.predict(xIca_train),average='weighted'))
print ('F1_test：', f1_score(yIca_test, GradientBoosting1.predict(xIca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, GradientBoosting1.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, GradientBoosting1.predict(xIca_test)))


# In[188]:


# calculate the time, over than 1 second, the effect is better than boost decision tree.


# In[96]:


import time
start =time.clock()

GradientBoosting1.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, GradientBoosting1.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, GradientBoosting1.predict(xIca_test)))

print ('Kappa_train：', cohen_kappa_score(yIca_train, GradientBoosting1.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, GradientBoosting1.predict(xIca_test)))

end=time.clock()
print('Running time: %s Seconds'%(end-start))


# In[97]:


GradientBoosting1.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, GradientBoosting1.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, GradientBoosting1.predict(xlle_test)))
print ('Kappa_train：', cohen_kappa_score(ylle_train, GradientBoosting1.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, GradientBoosting1.predict(xlle_test)))
print ('F1_train：', f1_score(ylle_train, GradientBoosting1.predict(xlle_train),average='weighted'))
print ('F1_test：', f1_score(ylle_test, GradientBoosting1.predict(xlle_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, GradientBoosting1.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, GradientBoosting1.predict(xlle_test)))


# In[98]:


# now we can find the ICA + GBDT is good, almost same as ICA + boost decision tree.then try to repeat to get a average value.


# In[99]:


summary=0
n=50
for j in range(0,50):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
    GradientBoosting1.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, GradientBoosting1.predict(xIca_test))+summary
summary/n


# In[100]:


# now we can find the ICA + GBDT is a little better than boost decision tree.


# In[101]:


# Then consider a improvement of GBDT, which is xgboost(extrem gradient boost decision tree), for this ,it runs faster than GBDT
# and also adding bagging in the classifier model. and there are some other improvement in this model, then lets's consider this.


# In[189]:


# use package(you shou download the package first.)


# In[102]:


pip install xgboost


# In[103]:


pip install graphviz


# In[104]:


from pandas import DataFrame
from xgboost.sklearn import XGBClassifier
from xgboost import plot_tree


# In[105]:


# first set a xgboost model


# In[106]:


xgboost = XGBClassifier(
    n_estimators=30,
    learning_rate =0.4,
    max_depth=4,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=12,
    reg_lambda=1,
    seed=27,
    num_class= 7)


# In[107]:


xgboost.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, xgboost.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, xgboost.predict(x_test)))
print ('Kappa_train：', cohen_kappa_score(y_train, xgboost.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, xgboost.predict(x_test)))
print ('F1_train：', f1_score(y_train, xgboost.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, xgboost.predict(x_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, xgboost.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, xgboost.predict(x_test)))


# In[108]:


# then see the time we need to use


# In[109]:


import time
start =time.clock()

xgboost.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, xgboost.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, xgboost.predict(x_test)))
print ('Kappa_train：', cohen_kappa_score(y_train, xgboost.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, xgboost.predict(x_test)))

end=time.clock()
print('Running time: %s Seconds'%(end-start))


# In[110]:


# here try reducing dimension + xgboost


# In[111]:


xpca_train, xpca_test, ypca_train, ypca_test = train_test_split(X_Pca_labeled,target_labeled, test_size=0.3)
xgboost.fit(xpca_train, ypca_train)


# In[112]:


xgboost.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, xgboost.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, xgboost.predict(xpca_test)))
print ('Kappa_train：', cohen_kappa_score(ypca_train, xgboost.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, xgboost.predict(xpca_test)))
print ('F1_train：', f1_score(ypca_train, xgboost.predict(xpca_train),average='weighted'))
print ('F1_test：', f1_score(ypca_test, xgboost.predict(xpca_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, xgboost.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, xgboost.predict(xpca_test)))


# In[113]:


xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
xgboost.fit(xIca_train, yIca_train)


# In[114]:


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


# In[ ]:


# see the time we need, we can find the time is short, so the xgboost is quite fast and good, then consider the accuracy.


# In[115]:


import time
start =time.clock()

xgboost.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, xgboost.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, xgboost.predict(xIca_test)))
print ('Kappa_train：', cohen_kappa_score(yIca_train, xgboost.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, xgboost.predict(xIca_test)))

end=time.clock()
print('Running time: %s Seconds'%(end-start))


# In[116]:


# also see the time we need, very fast, faster than GBDT.


# In[117]:


xlle_train, xlle_test, ylle_train, ylle_test = train_test_split(X_lle_labeled,target_labeled, test_size=0.3)
xgboost.fit(xlle_train, ylle_train)


# In[118]:


print ('accuracy_train：', accuracy_score(ylle_train, xgboost.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, xgboost.predict(xlle_test)))
print ('Kappa_train：', cohen_kappa_score(ylle_train, xgboost.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, xgboost.predict(xlle_test)))
print ('F1_train：', f1_score(ylle_train, xgboost.predict(xlle_train),average='weighted'))
print ('F1_test：', f1_score(ylle_test, xgboost.predict(xlle_test),average='weighted'))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, xgboost.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, xgboost.predict(xlle_test)))


# In[119]:


# now we can find the ICA + xgboost is quite good, then try to repeat to get a average value and the accuracy can even higher.
# for decision tree, ICA + xgboost is th best, compared with GBDT, Random Forest, Boost Decision tree.
# both the accuracy and time speed is very good.


# In[120]:


summary=0
n=50
for j in range(0,50):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
    xgboost.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, xgboost.predict(xIca_test))+summary
summary/n


# In[121]:


# At last, consider the deep netural network.
# This method needn't to reduce the dimension since the network can do it in the hidden layers.
# The problems is the best hidden layer is very difficult to find, it also runs a little slow.
# we can also compare the sffect between reducing and not reducing the dimension.


# In[122]:


# then let us set a model and see the effect.(this model may not the best one, but quite good still)


# In[123]:


from sklearn.neural_network import MLPClassifier
MLP=MLPClassifier(random_state=1,activation ="relu",solver="lbfgs",max_iter=600,beta_1=0.7,beta_2=0.7,hidden_layer_sizes=(70,))


# In[124]:


MLP.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, MLP.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, MLP.predict(x_test)))
print ('Kappa_train：', cohen_kappa_score(y_train, MLP.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, MLP.predict(x_test)))
print ('F1_train：', f1_score(y_train, MLP.predict(x_train),average='weighted'))
print ('F1_test：', f1_score(y_test, MLP.predict(x_test),average='weighted'))
print (MLP.n_layers_)
print (MLP.loss_)

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, MLP.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, MLP.predict(x_test)))


# In[125]:


# then we can consider to reduce the dimension then using netural network, but we should not set the hidden layer's first values too
# big since if we set it big, the process of reducing dimension can be meanless.


# In[126]:


MLP1=MLPClassifier(random_state=1,activation ="relu",solver="lbfgs",max_iter=600,beta_1=0.7,beta_2=0.7,hidden_layer_sizes=(16,))


# In[127]:


MLP1.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, MLP1.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, MLP1.predict(xpca_test)))
print ('Kappa_train：', cohen_kappa_score(ypca_train, MLP1.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, MLP1.predict(xpca_test)))
print ('F1_train：', f1_score(ypca_train, MLP1.predict(xpca_train),average='weighted'))
print ('F1_test：', f1_score(ypca_test, MLP1.predict(xpca_test),average='weighted'))
print (MLP1.n_layers_)
print (MLP1.loss_)

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, MLP1.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, MLP1.predict(xpca_test)))


# In[128]:


MLP2=MLPClassifier(random_state=1,activation ="relu",solver="lbfgs",max_iter=600,beta_1=0.7,beta_2=0.7,hidden_layer_sizes=(50,))


# In[129]:


MLP2.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, MLP2.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, MLP2.predict(xIca_test)))
print ('Kappa_train：', cohen_kappa_score(yIca_train, MLP2.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, MLP2.predict(xIca_test)))
print ('F1_train：', f1_score(yIca_train, MLP2.predict(xIca_train),average='weighted'))
print ('F1_test：', f1_score(yIca_test, MLP2.predict(xIca_test),average='weighted'))
print (MLP2.n_layers_)
print (MLP2.loss_)

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, MLP2.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, MLP2.predict(xIca_test)))


# In[130]:


# then we see the time of this model needs, 3 seconds, higher than other models bust lower than not reducing the dimension.
# but it is true if we don't reduce the dimension, the effect will be better.
# also, the classification effect can't compare with the xgboost, gbdt and random forest.


# In[131]:


import time
start =time.clock()

MLP2.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, MLP2.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, MLP2.predict(xIca_test)))
print ('Kappa_train：', cohen_kappa_score(yIca_train, MLP2.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, MLP2.predict(xIca_test)))

end=time.clock()
print('Running time: %s Seconds'%(end-start))


# In[132]:


# then repeat this to see the classification effect.


# In[190]:


# Ica + netural network, the accuracy is not very good in fact, and th speed is not very fast, too.


# In[133]:


summary=0
n=30
for j in range(0,30):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
    MLP2.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, MLP2.predict(xIca_test))+summary
summary/n


# In[134]:


MLP3 = MLPClassifier(random_state=1,activation ="relu",solver="lbfgs",max_iter=600,beta_1=0.7,beta_2=0.7,hidden_layer_sizes=(43,))


# In[135]:


MLP3.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, MLP3.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, MLP3.predict(xlle_test)))
print ('Kappa_train：', cohen_kappa_score(ylle_train, MLP3.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, MLP3.predict(xlle_test)))
print ('F1_train：', f1_score(ylle_train, MLP3.predict(xlle_train),average='weighted'))
print ('F1_test：', f1_score(ylle_test, MLP3.predict(xlle_test),average='weighted'))
print (MLP3.n_layers_)
print (MLP3.loss_)

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, MLP3.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, MLP3.predict(xlle_test)))


# In[136]:


# after these trying we can find for netural network, it is better to solve the high dimension data, reduce dimension can not improve
# the classificstion effect, since after we reducing the dimension we may loss some information. and the work of reduce dimension can
# be done in the hidden layer.

# if not reduce the dimension, the netural network seems have a good effect but it may consume some time.so compared with the xgboost,
# I think using xgboost is better.

# another problem is the hidden layer is quite difficult to design, compared with design the tree, network is quite complex.

# if we don't reduce the dimension, the netural network is good, accuracy is very high, (almost same with xgboost) but it runs too slow.


# In[137]:


summary=0
n=10
for j in range(0,10):
    x_train, x_test, y_train, y_test = train_test_split(X_labeled,target_labeled, test_size=0.3)
    MLP.fit(x_train, y_train)
    summary = accuracy_score(y_test, MLP.predict(x_test))+summary
summary/n


# In[138]:


# then let us see the time we need, as we know, ICA + xgboost need less than 1 second, and xgboost needs around 5-6 seconds.
# but the netural network need around 20 seconds, quite slow. But the effect are almost the same, so I prefer xgboost.


# In[139]:


import time
start =time.clock()

MLP.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, MLP.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, MLP.predict(x_test)))
print ('Kappa_train：', cohen_kappa_score(y_train, MLP.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, MLP.predict(x_test)))
print (MLP.n_layers_)
print (MLP.loss_)

end=time.clock()
print('Running time: %s Seconds'%(end-start))


# In[140]:


# from the above work, I find the ICA + xgboost can be the best model combination, the effect is almost same as netural network,
# but much faster. also fo xgboost model, it is easier to improve the model.
# Tnen I choose this method and try to improve this.In fact my model is good and no need to improve, but would just show the way
# of improving my model, so when we want to find a suitable model we can use this method.
# In fact changing parameters can just improve the model a little when our model is a suitable one.


# In[141]:


# First consider the best dimention of ICA method.


# In[142]:


xgboost = XGBClassifier(
    n_estimators=30,
    learning_rate =0.4,
    max_depth=4,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=12,
    reg_lambda=1,
    seed=27,
    num_class= 7)


# In[143]:


from sklearn.metrics import accuracy_score
Scores=[]

for i in range(2,20):
    ICA = FastICA(n_components=i,random_state=40) 
    X_ICA=ICA.fit_transform(X_n)
    X_ICA_labeled=np.vstack((X_ICA[10:30],X_ICA[40:50],X_ICA[120:180],X_ICA[190:560],X_ICA[570:760],X_ICA[770:850]))
    summary=0
    n=20
    for j in range(0,20):
        xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
        xgboost.fit(xIca_train, yIca_train)
        summary = accuracy_score(yIca_test, xgboost.predict(xIca_test))+summary
    a = summary/n
    Scores.append(a)  


# In[144]:


# see the score and find 12 dimension is a good choice.


# In[145]:


Scores


# In[146]:


N=array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])


# In[147]:


plt.plot(N,Scores)


# In[148]:


# from the above work, we can find the dimension 14 is a good choice.


# In[149]:


ICA = FastICA(n_components=14,random_state=40) 
X_ICA=ICA.fit_transform(X_n)
X_ICA_labeled=np.vstack((X_ICA[10:30],X_ICA[40:50],X_ICA[120:180],X_ICA[190:560],X_ICA[570:760],X_ICA[770:850]))


# In[150]:


# then to avoid overfitting, repeat the random split for 100 times and calculate the mean.
# Can also use cross_valid, but the effect won't be good since there are many classes but little samples, even some class with only 10 
# samples, so the cross_valid is not good here.


# In[151]:


summary=0
n=100
for j in range(0,100):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
    xgboost.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, xgboost.predict(xIca_test))+summary
a = summary/n
a


# In[152]:


from sklearn.model_selection import cross_val_score
a = cross_val_score(xgboost , X_ICA_labeled,target_labeled, cv=35) 
sum(a)/35


# In[153]:


# Then calculate the time.


# In[154]:


import time
start =time.clock()
clf = XGBClassifier(
    n_estimators=30,
    learning_rate =0.4,
    max_depth=4,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=12,
    reg_lambda=1,
    seed=27,
    num_class=7)
# your pragrama
summary=0
n=100
for j in range(0,100):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
    clf.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, clf.predict(xIca_test))+summary
a = summary/n

end=time.clock()

print('Running time: %s Seconds'%(end-start))


# In[155]:


# For all the work, I also find the the xgboost model a little overfitting, and the parameters can be improved.
# Now we consider this.
# here we use crossvaild method, this is not as good as the method we use before, but it is more convenient and the effect is almost 
# the same in fact.
# here we just show a method of improving xgboost, GBDT, random forest.


# In[156]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb


# In[157]:


# first try to find the suitable value of n_estimators.
# n_estimatore means the number of trees(group) added, adding these models, the effect will be better.


# In[158]:


ICA = FastICA(n_components=14,random_state=40) 
X_ICA=ICA.fit_transform(X_n)
X_ICA_labeled=np.vstack((X_ICA[10:30],X_ICA[40:50],X_ICA[120:180],X_ICA[190:560],X_ICA[570:760],X_ICA[770:850]))

cv_params = {'n_estimators': [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]}
other_params = {"learning_rate": 0.4, "max_depth": 4, "min_child_weight": 1, "gamma": 0.3, "subsample": 0.8, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 30, "num_class" : 7}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=35)
optimized_GBM.fit(X_ICA_labeled, target_labeled)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[159]:


cv_params = {'n_estimators': [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]}
other_params = {"learning_rate": 0.4, "max_depth": 4, "min_child_weight": 1, "gamma": 0.3, "subsample": 0.8, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 40, "num_class" : 7}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=35)
optimized_GBM.fit(X_ICA_labeled, target_labeled)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[160]:


# so now I can choose the n_estimator = 40
# The we try to estimate the suitable value of max_depth and min_child_weight.
# max_depth means the maximize deepth of the tree.


# In[161]:


cv_params = {'max_depth': [2, 3, 4, 5, 6, 7], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
other_params = {"learning_rate": 0.4, "max_depth": 4, "min_child_weight": 1, "gamma": 0.3, "subsample": 0.8, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 40, "num_class" : 7}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=35)
optimized_GBM.fit(X_ICA_labeled, target_labeled)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[162]:


# so now I can choose the max_depth = 4, the min_child_weight = 1.
# Then we try to estimate the best value of gamma.


# In[163]:


cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
other_params = {"learning_rate": 0.4, "max_depth": 4, "min_child_weight": 1, "gamma": 0.3, "subsample": 0.8, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 40, "num_class" : 7}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=35)
optimized_GBM.fit(X_ICA_labeled, target_labeled)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[164]:


# so now we can choose gamma = 0.1.
# Then we can try to estimate subsample and colsample_bytree.
# for subsample, it use the idea of bagging,a way to improve overfitting.


# In[165]:


cv_params = {'subsample': [0.4,0.5,0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
other_params = {"learning_rate": 0.4, "max_depth": 4, "min_child_weight": 1, "gamma": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 40, "num_class" : 7}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=35)
optimized_GBM.fit(X_ICA_labeled, target_labeled)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[166]:


# Then we can choose subsample = 0.5 , and colsample_bytree = 0.8 .
# Next step we try to estimate reg_lambda.


# In[167]:


cv_params = {'reg_lambda': [0.05, 0.1, 0.5, 0.8, 1, 1.2, 1.5, 2, 3]}
other_params = {"learning_rate": 0.4, "max_depth": 4, "min_child_weight": 1, "gamma": 0.1, "subsample": 0.5, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 40, "num_class" : 7}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=35)
optimized_GBM.fit(X_ICA_labeled, target_labeled)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[168]:


# Now the suitablle value of reg_lambda is about 1,
# Next is the last step and I should estimate the suitable value of learning_rate.
# learning_rate means the speed of learning.


# In[169]:


cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
other_params = {"learning_rate": 0.4, "max_depth": 4, "min_child_weight": 1, "gamma": 0.1, "subsample": 0.5, "colsample_bytree": 0.8,
                "objective": 'multi:softprob', "nthread": 12, "reg_lambda": 1, "seed": 27, "n_estimators": 40, "num_class" : 7}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, cv=35)
optimized_GBM.fit(X_ICA_labeled, target_labeled)

print('best parameter：{0}'.format(optimized_GBM.best_params_))
print('best score:{0}'.format(optimized_GBM.best_score_))


# In[170]:


# so the learning_rate = 0.4 is a good choice.


# In[171]:


# Now here is the new classifier model.


# In[172]:


Xgboost = XGBClassifier(
    n_estimators=40,
    learning_rate =0.4,
    max_depth=4,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.5,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=12,
    reg_lambda=1,
    seed=27,
    num_class = 7)

ICA = FastICA(n_components=14,random_state=40) 
X_ICA=ICA.fit_transform(X_n)
X_ICA_labeled=np.vstack((X_ICA[10:30],X_ICA[40:50],X_ICA[120:180],X_ICA[190:560],X_ICA[570:760],X_ICA[770:850]))


# In[173]:


# then we see the accuracy again, but we can find the change can just improve the model a little, the main work is to choose a good
# dimension reducing method and classifier model.


# In[174]:


summary=0
n=100
for j in range(0,100):
    xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
    Xgboost.fit(xIca_train, yIca_train)
    summary = accuracy_score(yIca_test, Xgboost.predict(xIca_test))+summary
average = summary/n
average


# In[175]:


# Now after improve the parameters of xgboost model, we can now find the classification accuracy has improved a little(around 0.01).
# Now we can see the accuracy, kappa again, in addition we can also see the confuse matrix and F1 score.
# Now the test accuracy can over 0.98 , the kappa is also around 0.97.


# In[176]:


from sklearn.metrics import f1_score
xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3)
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


# In[177]:


# then let us see how the tree is.


# In[178]:


pip install graphviz


# In[179]:


import xgboost as xgb
import pandas as pd


# In[180]:


target_labeled = pd.DataFrame(target_labeled)
target_labeled.columns=["class"]


# In[181]:


ICA_X = pd.DataFrame(X_ICA_labeled)
ICA_X.columns = ['W1','W2','W3','W4','W5','W6','W7','W8','W9','W10','W11','W12','W13','W14']

XGBoost = xgb.XGBClassifier(
    n_estimators=40,
    learning_rate =0.4,
    max_depth=4,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.5,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=12,
    reg_lambda=1,
    seed=27,
    num_class = 7)

XGBoost.fit(ICA_X,target_labeled)


# In[191]:


# see one of the tree in my model.


# In[182]:


fig,ax = plt.subplots()
fig.set_size_inches(60,40)
xgb.plot_tree(XGBoost,ax = ax,num_trees=2)


# In[ ]:




