clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPLORATORY ANALYSIS OF DATA:
% PRML R. Benitez 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load Iris database. Access the database at the UCI Machine Learning Repository:
% https://archive.ics.uci.edu/ml/datasets/iris

load fisheriris; % MATLAB built-in dataset
% 1. sepal length in cm
% 2. sepal width in cm
% 3. petal length in cm
% 4. petal width in cm
% 5. class:
% -- Iris Setosa
% -- Iris Versicolour
% -- Iris Virginica

% 1. DATA VISUALIZATION: 

% 1.1 basic plot univariate:
figure;
plot(meas(:,1),'ro-');

% 1.2 histogram: 
figure; 
histogram(meas(:,3),30);

% 1.3 boxplot: quantiles 25%, 50%, 75% and outliers
figure; 
boxplot(meas(:,1));

% 1.4 scatter plot two variables:
figure;
scatter(meas(:,1),meas(:,2),'ro');

% 1.5 scatter plot matrix 
figure;
gplotmatrix(meas,[],species);

% 1.6 grouped scatter plot matrix 
figure;
gplotmatrix(meas,[],species,{'r','g','b'},[],10,'on','variable',{'SL','SW','PL','PW'});

% 1.7 Correlation plot: scatter plot matrix with statistical test of pairwise linear correlations 
figure;corrplot(meas,'VarNames',{'SL','SW','PL','PW'},'testR','on');

% 1.8 quantile-quantile plot (q-q plot): Check two variables have the same distribution
figure;
qqplot(meas(:,1),meas(:,2));

% 1.9 quantile-quantile plot (q-q plot): Check a variable is normally
% distributed
figure;
qqplot(meas(:,1));

% 2. CLUSTER ANALYSIS:

% 2.1 k-means clustering:
k = 3; % Number of clusters 
T1 = kmeans(meas,k);
% Represent clustering results using grouped scatter plot matrix:
figure;
gplotmatrix(meas,[],T1);

% 2.2. Hierarchical clustering:
D = pdist(meas); % pairwise distances between observations
Y = squareform(D); 
Z = linkage(Y); % group into clusters 
figure;
dendrogram(Z); % Represent dendrogram
k = 3; % Number of clusters 
T2 = cluster(Z,'maxclust',k); % Cluster the data thresholding the dendrogram in order to have k clusters

% Represent clustering results using grouped scatter plot matrix:
figure;
gplotmatrix(meas,[],T2);

%2.3 Clustering with a Gaussian Mixture Model:
options = statset('MaxIter',1000);
k = 3; % Number of clusters 
obj = gmdistribution.fit(meas,k);% Build a GMM model with k gaussians

prob = obj.posterior(meas); % Compute the probability that each observation belongs to each cluster
[val T3] = max(prob'); % Assign each observation to the more likely cluster

% Represent clustering results using grouped scatter plot matrix:
figure;
gplotmatrix(meas,[],T3);

% 2.4 Evaluate the clustering results 
s1 = mean(silhouette(meas,T1,'cityblock'));
s2 = mean(silhouette(meas,T2,'cityblock'));
s3 = mean(silhouette(meas,T3,'cityblock'));

% 2.4 Model selection: How many clusters? 
krange = 1:6; % testing range
evakmeans = evalclusters(meas,'kmeans','CalinskiHarabasz','KList',krange); % optimal kmeans
evahier = evalclusters(meas,'linkage','CalinskiHarabasz','KList',krange); % optimal hierarchical
evagmm = evalclusters(meas,'gmdistribution','CalinskiHarabasz','KList',krange); % optimal GMM
% Explicit test of the GMM model selection uwing BIC: 
BIC = [];
for k = krange
   gmm{k} = gmdistribution.fit(meas,k);
   BIC = [BIC gmm{k}.BIC];
end; 
[minBIC ibest]= min(BIC);

% 3. DIMENSIONALITY REDUCTION

% 3.1 Principal Component Analysis (PCA):
[evec,PCA_projA,eval] = pca(meas);

% 3.2 Alternate method: diagonalize the covariance matrix of the data:
[eve eva] = eig(cov(meas));

% 3.3 Scree plot: Plot relevnce of each component (variance explained):
100*eval/sum(eval)

