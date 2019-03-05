import pandas as pd
import seaborn as sns
import statsmodels.api as sm
# from sklearn import manifold


# E1

#  T1 We will be using pandas library as pd.
data = pd.read_csv('./files/iris.data', delimiter = ',', names = ['SL', 'SW', 'PL', 'PW', 'CLASS'])


# E2

#  T1
sns.catplot(x="CLASS", y="SW", data=data, kind = 'box') # Box-plot of SW
sns.catplot(x="CLASS", y="SL", data=data, kind = 'bar') # Bar-plot of SL
sns.catplot(x="PL", data=data, kind = 'count', hue = 'CLASS') # Histogram of PL

#  T2
labels = ['SL', 'SW', 'PL', 'PW']
for i in range(len(labels)):
    print('DATA FOR ', labels[i],':')
    print('Number of non-NA/null observations for:', labels[i], data.count()[i])
    print('Maximum value for:', labels[i], data.max()[i])
    print('Minimum value for:', labels[i], '', data.max()[i])
    print('Mean for:', labels[i], data.mean()[i])
    print('Standard deviation for', labels[i], data.std()[i])
    print('Kurtosis for:', labels[i], data.kurt()[i])
    print('IQR for:', labels[i], data.quantile(0.75)[i]-data.quantile(0.25)[i])
    print('')
print('Covariance matrix')
print(data.cov())

#  T3 Scatter plot of 2 variables (SL & SW)
sns.catplot(x="SL", y="SW", data=data, hue = 'CLASS') # Scatter-plot, SL vs SW

#  T4 q-q plot of 2 variables (SL & SW), we need statsmodels
sns.scatterplot(x=data['PW'].sort_values(), y=data['SL'].sort_values(), data=data) # Same sample sizes, sorted data, 'PW' vs 'SL', the result is a qqplot, with same sample size
sm.qqplot_2samples(data['SL'],data['PW'])  # Result should be the same as before
# sm.qqplot_2samples(data['SL'],data['PW'],line='45') # Line does not fit, and shouldn't be 45 Ask professor.

#  T5, T6
sns.pairplot(data, hue="CLASS") # Scatter plot matrix

#  T7 Apply multidimensional scaling (MDS) to project the d-dimensional data in 2-d data
# mds = manifold.MDS(n_components=2, metric=Flase)
# pos = mds.fit(data[['SL','SW','PL','PW']]).embedding_