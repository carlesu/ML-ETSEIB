# PCA with synthetic data, and with uci data
# Model evaluation: Performance measures (type I, type II)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import math


# 1.1
mu=[-1.2, 1.8]
sigma=[0.1, 0.5]
n=100000
data2=pd.concat([pd.DataFrame(np.random.normal(mu[0], sigma[0], n), columns=['1']),
                 pd.DataFrame(np.random.normal(mu[1], sigma[1], n), columns=['2'])], axis=1)
# We can see the distribution 1
distfig1 = plt.figure()
distfig1.suptitle('Distribution for generated mu=-1.2 sigma=0.1)')
sns.distplot(data2['1'])
# We can see the distribution 2
distfig2 = plt.figure()
distfig2.suptitle('Distribution for generated mu=1.8 sigma=0.5)')
sns.distplot(data2['2'])
# Checking correlations on Distribution 1 vs Distribution 2
sns.pairplot(data2)


# 1.2 & 1.3
# Generating the linear combination so we have 4 columns of data
datatot=pd.concat([data2, 0.3*data2['1']+1.2*data2['2'], np.sqrt(np.absolute(data2['1']))+data2['2']], axis=1)
datatot.columns=['1', '2', '3', '4']


# 1.4
# Checking correlations on Distribution 1 vs Distribution 2 vs Distribution 3 vs Distribution 4
sns.pairplot(datatot)


# 1.5
# PCA on all variables, so we can decide after which ones we use
pcaexplore = PCA(n_components=len(datatot.columns))
# PCA of our whole data
pcaexplore.fit(datatot)
# print the eigen values
print('Explained Variance:')
for i,b in enumerate(pcaexplore.explained_variance_):
    print('Dist', i,':', b)
# print the eigen values %
print('\nExplained Variance Ratio:')
eigenpercent = pcaexplore.explained_variance_ratio_
for i,b in enumerate(eigenpercent):
    print('Dist', i,':', b)
print('\nSingular Valures', pcaexplore.singular_values_)
# Plot of the eigen value in %, how much % explains each component
eigenpercentframe = pd.DataFrame(eigenpercent, columns=['percent'])
fig, ax = plt.subplots()
ax.set_ylim(min(0,min(eigenpercentframe['percent'])),max(eigenpercentframe['percent'])*1.1)
fig.suptitle('Eigenvalues %', fontsize=12)
barcollection = ax.bar(np.arange(eigenpercentframe['percent'].count()), eigenpercentframe['percent'])
acumulative = 0.0
# Define the % of acceptance to 0.95 (95%)
minimumpercent = 0.95
# Color in green the bars that explain our desired %
for i, bar in enumerate(barcollection):
    ax.text(bar.get_x() + bar.get_width()/3., 1.05*bar.get_height(), round(100*bar.get_height(), 4) )
    if acumulative < minimumpercent:
        # Set color green until 95 % is explained
        bar.set_color('g')
    acumulative += bar.get_height()

# We already made a bar plot about it, but automatically we find the minimum components(pcancomp) that explains the
# desired % of our variance, and perform the PCA dimensionality reduction.
sumpercent = 0
pcancomp = 0
for i,value in enumerate(eigenpercent):
    if sumpercent <= minimumpercent:
        pcancomp = i + 1
    sumpercent += value
pca = PCA(n_components=pcancomp)
pca.fit(datatot)
datareduced = pd.DataFrame(pca.transform(datatot))
datareduced.columns=['1']


# 1.6
# Let's get all the data in the principal components projection and plot it.
data_new = pd.DataFrame(pca.inverse_transform(datareduced))
data_new.columns=['1', '2', '3', '4']
sns.pairplot(datatot)
sns.pairplot(data_new)
# Find a way to  do a 3d plot


# 2.1, 2.2
# Getting iris data.
datairis = pd.read_csv('./files/iris.data', delimiter = ',', names = ['SL', 'SW', 'PL', 'PW', 'CLASS'])
# Eliminate class information
features = datairis.drop('CLASS', axis = 1)


# 2.3
# Check correlations
sns.pairplot(features)
# PCA on all variables, so we can decide after which ones we use
pcaexplore = PCA(n_components=len(features.columns))
# PCA of our whole data
pcaexplore.fit(features)
# print the eigen values
print('Explained Variance:')
for i,b in enumerate(pcaexplore.explained_variance_):
    print('Dist', i,':', b)
# print the eigen values %
print('\nExplained Variance Ratio:')
eigenpercent = pcaexplore.explained_variance_ratio_
for i,b in enumerate(eigenpercent):
    print('Dist', i,':', b)
print('\nSingular Valures', pcaexplore.singular_values_)
# Plot of the eigen value in %, how much % explains each component
eigenpercentframe = pd.DataFrame(eigenpercent, columns=['percent'])
fig, ax = plt.subplots()
ax.set_ylim(min(0,min(eigenpercentframe['percent'])),max(eigenpercentframe['percent'])*1.1)
fig.suptitle('Eigenvalues %', fontsize=12)
barcollection = ax.bar(np.arange(eigenpercentframe['percent'].count()), eigenpercentframe['percent'])
acumulative = 0.0
# Define the % of acceptance
minimumpercent = 0.95
# Color in green the bars that explain our desired %
for i, bar in enumerate(barcollection):
    ax.text(bar.get_x() + bar.get_width()/3., 1.05*bar.get_height(), round(100*bar.get_height(), 4) )
    if acumulative < minimumpercent:
        # Set color green until 95 % is explained
        bar.set_color('g')
    acumulative += bar.get_height()
# We already made a bar plot about it, but automatically we find the minimum components(pcancomp) that explains the
# desired % of our variance, and perform the PCA dimensionality reduction.
# We already made a bar plot about it, but automatically we find the minimum components(pcancomp) that explains the
# desired % of our variance, and perform the PCA dimensionality reduction.
sumpercent = 0
pcancomp = 0
for i,value in enumerate(eigenpercent):
    if sumpercent <= minimumpercent:
        pcancomp = i + 1
    sumpercent += value
pca = PCA(n_components=pcancomp)
pca.fit(features)
datareduced = pd.DataFrame(pca.transform(features))
datareduced.columns=['1', '2']
# We get all the data but expressed in the PCA1 and PCA2, because que did a PCA to 2 components. data_new performs it.
data_new = pd.DataFrame(pca.inverse_transform(datareduced))
data_new.columns=['1', '2', '3', '4']
sns.pairplot(features)
sns.pairplot(data_new)
sns.scatterplot(x=datareduced['1'], y=datareduced['2'], data=datareduced)

# 3.1
# 1-calculate the distance or similarity between samples.
# 2-find best way to plot the variability of the data.
# 3-plot data.
# It's similar to PCA, but it is done for distances, while PCA it is done for variance.
# It will be done only for the iris data.
mds = MDS(n_components=2, dissimilarity='euclidean')
mds.fit(features)
mdsreduced = mds.fit_transform(features)
mdsreducedframe = pd.DataFrame(mdsreduced)
mdsreducedframe.columns = ['1', '2']
sns.scatterplot(x=mdsreducedframe['1'], y=mdsreducedframe['2'], data=mdsreducedframe)

# 3.2
# Let's get sample 3, 42, 86
l=[]
# Sample 3 - 42
l.append([math.sqrt((float(features.iloc[[3]]['SL'])-float(features.iloc[[42]]['SL']))**2 +
          (float(features.iloc[[3]]['SW'])-float(features.iloc[[42]]['SW']))**2 +
          (float(features.iloc[[3]]['PL'])-float(features.iloc[[42]]['PL']))**2 +
          (float(features.iloc[[3]]['PW'])-float(features.iloc[[42]]['PW']))**2),
          math.sqrt((float(mdsreducedframe.iloc[[3]]['1'])-float(mdsreducedframe.iloc[[42]]['1']))**2 +
          (float(mdsreducedframe.iloc[[3]]['2'])-float(mdsreducedframe.iloc[[42]]['2']))**2)])

# Sample 3 - 86
l.append([math.sqrt((float(features.iloc[[3]]['SL'])-float(features.iloc[[86]]['SL']))**2 +
          (float(features.iloc[[3]]['SW'])-float(features.iloc[[86]]['SW']))**2 +
          (float(features.iloc[[3]]['PL'])-float(features.iloc[[86]]['PL']))**2 +
          (float(features.iloc[[3]]['PW'])-float(features.iloc[[86]]['PW']))**2),
          math.sqrt((float(mdsreducedframe.iloc[[3]]['1'])-float(mdsreducedframe.iloc[[86]]['1']))**2 +
          (float(mdsreducedframe.iloc[[3]]['2'])-float(mdsreducedframe.iloc[[86]]['2']))**2)])

# Sample 42 - 86
l.append([math.sqrt((float(features.iloc[[42]]['SL'])-float(features.iloc[[86]]['SL']))**2 +
          (float(features.iloc[[42]]['SW'])-float(features.iloc[[86]]['SW']))**2 +
          (float(features.iloc[[42]]['PL'])-float(features.iloc[[86]]['PL']))**2 +
          (float(features.iloc[[42]]['PW'])-float(features.iloc[[86]]['PW']))**2),
          math.sqrt((float(mdsreducedframe.iloc[[42]]['1'])-float(mdsreducedframe.iloc[[86]]['1']))**2 +
          (float(mdsreducedframe.iloc[[42]]['2'])-float(mdsreducedframe.iloc[[86]]['2']))**2)])

# l =[[0.29999999999999954, 0.2825962161379616],
# [4.042276586281548, 4.063181086144194],
# [4.306971093471606, 4.3401561131452615]] So pretty solid, they are very similar.