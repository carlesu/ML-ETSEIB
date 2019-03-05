
# coding: utf-8

# # Dimensionality reduction: Principal Component Analysis

# In[6]:

import numpy as np
from sklearn import decomposition
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

pca = decomposition.PCA(n_components=2)
pca.fit(X)
Xproj = pca.transform(X)

# Variance fraction explained by each principal component (eigenvalue of the corresponding eigenvector)
pca.explained_variance_ratio_


# In[16]:

#Simple scatter plot (matplotlib):
import matplotlib.pyplot as plt
plt.scatter(Xproj[:,0], Xproj[:,1])
plt.show()


# In[26]:

get_ipython().magic(u'matplotlib inline')
#grouped scatter plot (seaborn):
import seaborn as sns
sns.scatterplot(x=Xproj[:,0], y=Xproj[:,1], hue=y)


# In[ ]:



