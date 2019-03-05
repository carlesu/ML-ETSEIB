# PCA with synthetic data, and with uci data
# Model evaluation: Performance measures (type I, type II)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# E1
#  T1
mu=[-1.2, 7.2]
sigma=[0.1, 2.3]
n=100000
data2=pd.concat([pd.DataFrame(np.random.normal(mu[0], sigma[0], n), columns=['1']), pd.DataFrame(np.random.normal(mu[1], sigma[1], n), columns=['2'])], axis=1)
plt.figure()
sns.distplot(data2['1']) # We can see the distribution 1
plt.figure()
sns.distplot(data2['2']) # We can see the distribution 2
sns.pairplot(data2) # Checking correlations on Distribution 1 vs Distribution 2

# T2 & T3
datatot=pd.concat([data2, 0.3*data2['1']+1.2*data2['2'], np.sqrt(np.absolute(data2['1']))+data2['2']], axis=1)
datatot.columns=['1', '2', '3', '4']

# T4
sns.pairplot(datatot) # Checking correlations on Distribution 1 vs Distribution 2 vs Distribution 3 vs Distribution 4

# T5
pca = PCA(n_components=4) # We do PCA on all variables, so we can decide after which ones we use
pca.fit(datatot) # We do PCA of our whole data
print('Explained Variance:')
for i,b in enumerate(pca.explained_variance_):
    print('Dist', i,':', b)
print('\nExplained Variance Ratio:')
for i,b in enumerate(pca.explained_variance_ratio_):
    print('Dist', i,':', b)
print('\nSingular Valures', pca.singular_values_)
# Plot from eigenvalues needed, homework
# T6
# What?
pca.transform(datatot)