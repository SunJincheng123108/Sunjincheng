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


# In[4]:


Y=np.loadtxt('Reflection2_00001.txt').T[0,]
X=pd.read_csv('Coffee.csv')


# In[6]:


X.head()


# In[7]:


Y[300],Y[1600]


# In[8]:


x=X.T
X_n=x[300:1600].T
X_n.shape


# In[9]:


pca=PCA(n_components=11)
X_PCA=pca.fit_transform(X_n)
ICA = FastICA(n_components=11,random_state=400) 
X_ICA=ICA.fit_transform(X_n)


# In[10]:


X_tsne = TSNE(n_components=3, n_iter=800).fit_transform(X_n)


# In[11]:


lle=LocallyLinearEmbedding(n_components=16,n_neighbors=40)
X_lle=lle.fit_transform(X_n)


# In[12]:


from numpy import array


# In[13]:


# first use 1,2,3,...to present the different class:
# "0": unlabeled
# "1": health good
# "2": rust
# "3": infected green canopy
# "4": geisha
# "5": geisha cannopy
# "6": yellow
# "7": yellow green
# "8": Young_Rust_Yellow
# "9": Young_Rust_Green_Leaf
target=array(["1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","0","0","0","0","0","0","0","0","0","0","8","8","8","8","8","8","8","8","8","8",
             "8","8","8","8","8","8","8","8","8","8","0","0","0","0","0","0","0","0","0","0","9","9","9","9","9","9","9","9","9","9",
             "0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0",
             "0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0",
             "0","0","0","0","0","0","0","0","0","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","0","0","0","0","0","0","0","0","0","0","1","1","1","1","1","1","1","1","1","1",
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
             "0","0","0","0","0","0","0","0","0","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","4","4","4","4","4","4","4","4","4","4",
             "4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4",
             "4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4",
             "4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4",
             "4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","0","0","0","0","0","0","0","0","0","0",
             "4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4",
             "5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5",
             "5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5"])

target_labeled=array(["1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1",
             "1","1","1","1","1","1","1","1","1","1","8","8","8","8","8","8","8","8","8","8",
             "8","8","8","8","8","8","8","8","8","8","9","9","9","9","9","9","9","9","9","9",
             "1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1",
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

target_unlabeled=array(["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0",
             "0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0",
             "0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0",
             "0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0",
             "0","0","0","0","0","0","0","0","0","0"])


# In[14]:


np.savetxt('New_X',X_n)
np.savetxt("PCA",X_PCA)
np.savetxt("ICA",X_ICA)
np.savetxt("LLE",X_lle)
np.savetxt('TSNE',X_tsne)


# In[15]:


X_new=np.loadtxt('New_X')
X_Pca=np.loadtxt("PCA")
X_ICA=np.loadtxt("ICA")
X_lle=np.loadtxt("LLE")
X_tsne=np.loadtxt("TSNE")


# In[16]:


X.shape


# In[17]:


X_tsne_labeled=np.vstack((X_tsne[0:40],X_tsne[50:70],X_tsne[80:90],X_tsne[160:220],X_tsne[230:600],X_tsne[610:800],X_tsne[810:890]))
X_tsne_unlabeled=np.vstack((X_tsne[40:50],X_tsne[70:80],X_tsne[90:160],X_tsne[220:230],X_tsne[600:610],X_tsne[800:810]))
X_Pca_labeled=np.vstack((X_Pca[0:40],X_Pca[50:70],X_Pca[80:90],X_Pca[160:220],X_Pca[230:600],X_Pca[610:800],X_Pca[810:890]))
X_Pca_unlabeled=np.vstack((X_Pca[40:50],X_Pca[70:80],X_Pca[90:160],X_Pca[220:230],X_Pca[600:610],X_Pca[800:810]))
X_ICA_labeled=np.vstack((X_ICA[0:40],X_ICA[50:70],X_ICA[80:90],X_ICA[160:220],X_ICA[230:600],X_ICA[610:800],X_ICA[810:890]))
X_ICA_unlabeled=np.vstack((X_ICA[40:50],X_ICA[70:80],X_ICA[90:160],X_ICA[220:230],X_ICA[600:610],X_ICA[800:810]))
X_lle_labeled=np.vstack((X_lle[0:40],X_lle[50:70],X_lle[80:90],X_lle[160:220],X_lle[230:600],X_lle[610:800],X_lle[810:890]))
X_lle_unlabeled=np.vstack((X_lle[40:50],X_lle[70:80],X_lle[90:160],X_lle[220:230],X_lle[600:610],X_lle[800:810]))
X_labeled=np.vstack((X_new[0:40],X_new[50:70],X_new[80:90],X_new[160:220],X_new[230:600],X_new[610:800],X_new[810:890]))
X_unlabeled=np.vstack((X_new[40:50],X_new[70:80],X_new[90:160],X_new[220:230],X_new[600:610],X_new[800:810]))


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(X_labeled,target_labeled, test_size=0.3,random_state=50)
xpca_train, xpca_test, ypca_train, ypca_test = train_test_split(X_Pca_labeled,target_labeled, test_size=0.3,random_state=50)
xIca_train, xIca_test, yIca_train, yIca_test = train_test_split(X_ICA_labeled,target_labeled, test_size=0.3,random_state=50)
xlle_train, xlle_test, ylle_train, ylle_test = train_test_split(X_lle_labeled,target_labeled, test_size=0.3,random_state=50)
xtsne_train, xtsne_test, ytsne_train, ytsne_test = train_test_split(X_tsne_labeled,target_labeled, test_size=0.3,random_state=50)


# In[19]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV


# In[20]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-3,3,50),gamma=np.logspace(-3,3,50))
clf = RandomizedSearchCV(SVM, distributions, random_state=50)


# In[21]:


search1 = clf.fit(x_train, y_train)
search1.best_params_ 


# In[22]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-3,3,50),gamma=np.logspace(-3,3,50))
clf = RandomizedSearchCV(SVM, distributions, random_state=50)


# In[23]:


search2 = clf.fit(xpca_train, ypca_train)
search2.best_params_ 


# In[24]:


clf= SVC(C=4.7,kernel='rbf',gamma=0.003)


# In[25]:


clf.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, clf.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, clf.predict(x_test)))

print ('Kappa_train：', cohen_kappa_score(y_train, clf.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, clf.predict(x_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, clf.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, clf.predict(x_test)))


# In[26]:


clf1= SVC(C=4.7,kernel='rbf',gamma=0.003)


# In[27]:


clf1.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, clf1.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, clf1.predict(xpca_test)))

print ('Kappa_train：', cohen_kappa_score(ypca_train, clf1.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, clf1.predict(xpca_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, clf1.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, clf1.predict(xpca_test)))


# In[28]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-3,1,50),gamma=np.logspace(-3,2,50))
clf = RandomizedSearchCV(SVM, distributions, random_state=50)


# In[29]:


search4 = clf.fit(xtsne_train, ytsne_train)
search4.best_params_ 


# In[30]:


clf2= SVC(C=4.7,kernel='rbf',gamma=0.6)


# In[31]:


clf2.fit(xtsne_train, ytsne_train)
print ('accuracy_train：', accuracy_score(ytsne_train, clf2.predict(xtsne_train)))
print ('accuracy_test：', accuracy_score(ytsne_test, clf2.predict(xtsne_test)))

print ('Kappa_train：', cohen_kappa_score(ytsne_train, clf2.predict(xtsne_train)))
print ('Kappa_test：', cohen_kappa_score(ytsne_test, clf2.predict(xtsne_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ytsne_train, clf2.predict(xtsne_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ytsne_test, clf2.predict(xtsne_test)))


# In[32]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-3,2,50),gamma=np.logspace(-3,2,50))
clf = RandomizedSearchCV(SVM, distributions, random_state=50)


# In[33]:


search3 = clf.fit(xIca_train, yIca_train)
search3.best_params_ 


# In[34]:


clf3= SVC(C=20,kernel='rbf',gamma=24)


# In[35]:


clf3.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, clf3.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, clf3.predict(xIca_test)))

print ('Kappa_train：', cohen_kappa_score(yIca_train, clf3.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, clf3.predict(xIca_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, clf3.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, clf3.predict(xIca_test)))


# In[36]:


SVM= SVC(kernel='rbf')
distributions = dict(C=np.logspace(-3,2,50),gamma=np.logspace(-3,2,50))
clf = RandomizedSearchCV(SVM, distributions, random_state=50)


# In[37]:


search5 = clf.fit(xlle_train, ylle_train)
search5.best_params_ 


# In[38]:


clf4= SVC(C=20,kernel='rbf',gamma=24)


# In[39]:


clf4.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, clf4.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, clf4.predict(xlle_test)))

print ('Kappa_train：', cohen_kappa_score(ylle_train, clf4.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, clf4.predict(xlle_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, clf4.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, clf4.predict(xlle_test)))


# In[40]:


from sklearn import tree
clf_tree = tree.DecisionTreeClassifier(max_depth=9)


# In[41]:


clf_tree.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, clf_tree.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, clf_tree.predict(x_test)))

print ('Kappa_train：', cohen_kappa_score(y_train, clf_tree.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, clf_tree.predict(x_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, clf_tree.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, clf_tree.predict(x_test)))


# In[42]:


clf_tree.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, clf_tree.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, clf_tree.predict(xpca_test)))

print ('Kappa_train：', cohen_kappa_score(ypca_train, clf_tree.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, clf_tree.predict(xpca_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, clf_tree.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, clf_tree.predict(xpca_test)))


# In[43]:


clf_tree.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, clf_tree.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, clf_tree.predict(xIca_test)))

print ('Kappa_train：', cohen_kappa_score(yIca_train, clf_tree.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, clf_tree.predict(xIca_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, clf_tree.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, clf_tree.predict(xIca_test)))


# In[44]:


clf_tree.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, clf_tree.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, clf_tree.predict(xlle_test)))

print ('Kappa_train：', cohen_kappa_score(ylle_train, clf_tree.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, clf_tree.predict(xlle_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, clf_tree.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, clf_tree.predict(xlle_test)))


# In[45]:


clf_tree.fit(xtsne_train, ytsne_train)
print ('accuracy_train：', accuracy_score(ytsne_train, clf_tree.predict(xtsne_train)))
print ('accuracy_test：', accuracy_score(ytsne_test, clf_tree.predict(xtsne_test)))

print ('Kappa_train：', cohen_kappa_score(ytsne_train, clf_tree.predict(xtsne_train)))
print ('Kappa_test：', cohen_kappa_score(ytsne_test, clf_tree.predict(xtsne_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ytsne_train, clf_tree.predict(xtsne_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ytsne_test, clf_tree.predict(xtsne_test)))


# In[46]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[47]:


randomforest = RandomForestClassifier(max_depth=8)


# In[48]:


randomforest.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, randomforest.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, randomforest.predict(x_test)))

print ('Kappa_train：', cohen_kappa_score(y_train, randomforest.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, randomforest.predict(x_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, randomforest.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, randomforest.predict(x_test)))


# In[49]:


randomforest.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, randomforest.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, randomforest.predict(xpca_test)))

print ('Kappa_train：', cohen_kappa_score(ypca_train, randomforest.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, randomforest.predict(xpca_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, randomforest.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, randomforest.predict(xpca_test)))


# In[50]:


randomforest.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, randomforest.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, randomforest.predict(xIca_test)))

print ('Kappa_train：', cohen_kappa_score(yIca_train, randomforest.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, randomforest.predict(xIca_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, randomforest.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, randomforest.predict(xIca_test)))


# In[51]:


randomforest.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, randomforest.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, randomforest.predict(xlle_test)))

print ('Kappa_train：', cohen_kappa_score(ylle_train, randomforest.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, randomforest.predict(xlle_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, randomforest.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, randomforest.predict(xlle_test)))


# In[52]:


from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
GradientBoosting1 = GradientBoostingClassifier(max_depth=2)
GradientBoosting2 = GradientBoostingClassifier(random_state=10,max_depth=3,min_samples_split=30,min_samples_leaf=10)


# In[53]:


GradientBoosting1.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, GradientBoosting1.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, GradientBoosting1.predict(x_test)))

print ('Kappa_train：', cohen_kappa_score(y_train, GradientBoosting1.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, GradientBoosting1.predict(x_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, GradientBoosting1.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, GradientBoosting1.predict(x_test)))


# In[54]:


GradientBoosting1.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, GradientBoosting1.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, GradientBoosting1.predict(xpca_test)))

print ('Kappa_train：', cohen_kappa_score(ypca_train, GradientBoosting1.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, GradientBoosting1.predict(xpca_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, GradientBoosting1.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, GradientBoosting1.predict(xpca_test)))


# In[55]:


GradientBoosting1.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, GradientBoosting1.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, GradientBoosting1.predict(xIca_test)))

print ('Kappa_train：', cohen_kappa_score(yIca_train, GradientBoosting1.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, GradientBoosting1.predict(xIca_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, GradientBoosting1.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, GradientBoosting1.predict(xIca_test)))


# In[56]:


GradientBoosting1.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, GradientBoosting1.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, GradientBoosting1.predict(xlle_test)))

print ('Kappa_train：', cohen_kappa_score(ylle_train, GradientBoosting1.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, GradientBoosting1.predict(xlle_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, GradientBoosting1.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, GradientBoosting1.predict(xlle_test)))


# In[57]:


GradientBoosting2.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, GradientBoosting2.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, GradientBoosting2.predict(xlle_test)))

print ('Kappa_train：', cohen_kappa_score(ylle_train, GradientBoosting2.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, GradientBoosting2.predict(xlle_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, GradientBoosting2.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, GradientBoosting2.predict(xlle_test)))


# In[58]:


from sklearn.neural_network import MLPClassifier


# In[59]:


help(MLPClassifier)


# In[60]:


MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20,40,20), random_state=1)


# In[61]:


MLP.fit(x_train, y_train)
print ('accuracy_train：', accuracy_score(y_train, MLP.predict(x_train)))
print ('accuracy_test：', accuracy_score(y_test, MLP.predict(x_test)))

print ('Kappa_train：', cohen_kappa_score(y_train, MLP.predict(x_train)))
print ('Kappa_test：', cohen_kappa_score(y_test, MLP.predict(x_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(y_train, MLP.predict(x_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(y_test, MLP.predict(x_test)))


# In[62]:


MLP.fit(xpca_train, ypca_train)
print ('accuracy_train：', accuracy_score(ypca_train, MLP.predict(xpca_train)))
print ('accuracy_test：', accuracy_score(ypca_test, MLP.predict(xpca_test)))

print ('Kappa_train：', cohen_kappa_score(ypca_train, MLP.predict(xpca_train)))
print ('Kappa_test：', cohen_kappa_score(ypca_test, MLP.predict(xpca_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ypca_train, MLP.predict(xpca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ypca_test, MLP.predict(xpca_test)))


# In[63]:


MLP1 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,20,10), random_state=1)


# In[64]:


MLP1.fit(xIca_train, yIca_train)
print ('accuracy_train：', accuracy_score(yIca_train, MLP1.predict(xIca_train)))
print ('accuracy_test：', accuracy_score(yIca_test, MLP1.predict(xIca_test)))

print ('Kappa_train：', cohen_kappa_score(yIca_train, MLP1.predict(xIca_train)))
print ('Kappa_test：', cohen_kappa_score(yIca_test, MLP1.predict(xIca_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(yIca_train, MLP1.predict(xIca_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(yIca_test, MLP1.predict(xIca_test)))


# In[65]:


MLP.fit(xlle_train, ylle_train)
print ('accuracy_train：', accuracy_score(ylle_train, MLP.predict(xlle_train)))
print ('accuracy_test：', accuracy_score(ylle_test, MLP.predict(xlle_test)))

print ('Kappa_train：', cohen_kappa_score(ylle_train, MLP.predict(xlle_train)))
print ('Kappa_test：', cohen_kappa_score(ylle_test, MLP.predict(xlle_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ylle_train, MLP.predict(xlle_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ylle_test, MLP.predict(xlle_test)))


# In[66]:


MLP.fit(xtsne_train, ytsne_train)
print ('accuracy_train：', accuracy_score(ytsne_train, MLP.predict(xtsne_train)))
print ('accuracy_test：', accuracy_score(ytsne_test, MLP.predict(xtsne_test)))

print ('Kappa_train：', cohen_kappa_score(ytsne_train, MLP.predict(xtsne_train)))
print ('Kappa_test：', cohen_kappa_score(ytsne_test, MLP.predict(xtsne_test)))

print ('confusion_matrix_train：')
print ( confusion_matrix(ytsne_train, MLP.predict(xtsne_train)))
print ('confusion_matrix_test：')
print ( confusion_matrix(ytsne_test, MLP.predict(xtsne_test)))


# In[ ]:




