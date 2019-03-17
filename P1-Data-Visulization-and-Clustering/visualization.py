import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


# 1.1 We will be using pandas library as pd.
data = pd.read_csv('./files/iris.data', delimiter = ',', names = ['SL', 'SW', 'PL', 'PW', 'CLASS'])

# 2.1
# Box-plot of SW
sns.catplot(x="CLASS", y="SW", data=data, kind = 'box')
# Bar-plot of SL
sns.catplot(x="CLASS", y="SL", data=data, kind = 'bar')
# Histogram of PL
sns.catplot(x="PL", data=data, kind = 'count', hue = 'CLASS')

# 2.2
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

# 2.3 Scatter plot of 2 variables (SL & SW)
# Scatter-plot, SL vs SW
sns.catplot(x="SL", y="SW", data=data, hue = 'CLASS')

# 2.4 q-q plot of 2 variables (SL & SW), we need statsmodels
# Same sample sizes, sorted data, 'PW' vs 'SL', the result is a qqplot, with same sample size
sm.qqplot_2samples(data['SL'],data['PW'])

#  2.5, 2.6
# Scatter plot matrix
sns.pairplot(data, hue='CLASS')

# 2.7 Apply multidimensional scaling (MDS) to project the d-dimensional data in 2-d data
X = data.iloc[:,:4]
embedding = MDS(n_components=2)
x_transformes = embedding.fit_transform(X[:100])
x_transformes.shape
plt.scatter(x_transformes[:,0],x_transformes[:,1])