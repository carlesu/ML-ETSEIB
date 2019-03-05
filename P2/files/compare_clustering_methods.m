close all;
clear all;
clc; 

% Generate k 2D Gaussians as observations:
rng(123456);
NOBS = 100;
MU1 = [1 2];
SIGMA1 = [2 0; 0 .5];
MU2 = [-3 -5];
SIGMA2 = [1 0; 0 1];
X = [mvnrnd(MU1,SIGMA1,NOBS);
mvnrnd(MU2,SIGMA2,NOBS)];

figure; 
scatter(X(:,1),X(:,2),'o');axis equal


% kmeans clustering
[idx,C] = kmeans(X,2);
figure; 
plot(X(idx==1,1),X(idx==1,2),'ro'); hold on;
plot(X(idx==2,1),X(idx==2,2),'b^');
plot(C(:,1),C(:,2),'kx','MarkerSize',14);  

% Hierarchical clustering:
Y = pdist(X,'euclid'); 
Z = linkage(Y,'average'); 
figure; dendrogram(Z);
T = cluster(Z,'maxclust',2); 
figure; 
plot(X(T==1,1),X(T==1,2),'ro'); hold on;
plot(X(T==2,1),X(T==2,2),'b^');
plot(C(:,1),C(:,2),'kx','MarkerSize',14);  

% Gaussian Mixture Models:
options = statset('Display','final');
obj = gmdistribution.fit(X,2,'Options',options);
hold on
h = ezcontour(@(x,y)pdf(obj,[x y]),[-8 6],[-8 6]);
hold off

% Cosider models with G=1:NC and Choose the model that presents the largest BIC and AIC: 
AIC = [];
BIC = [];

NC = 3;
obj1 = cell(1,NC);
options = statset('MaxIter',1000);
for k = 1:NC
    obj1{k} = gmdistribution.fit(X,k,'Options',options);
    AIC = [AIC  obj1{k}.AIC];
    BIC = [BIC  obj1{k}.BIC]  
end

[minAIC,numComponentsAIC] = min(AIC);
[minBIC,numComponentsBIC] = min(BIC);

figure; plot(1:NC, AIC,'ro-'); hold on;
plot(1:NC,BIC,'ks-');




