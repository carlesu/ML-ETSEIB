%% PRML Lab session 
% Application of Discriminant Analysis to classification of image data:

% Add Toolbox to the MATLAB path:
addpath('dipum_toolbox_2.0.2');
%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.10 $  $Date: 2004/12/15 20:15:38 $

% Load images: 
Iblue = imread('Images/Fig1302(a)_blue.tif');
Igreen = imread('Images/Fig1302(b)_green.tif');
Ired = imread('Images/Fig1302(c)_red.tif');
InearIR = imread('Images/Fig1302(d)_near_IR.tif');

%% Select a sample region of each class (water, urban, vegatation): 
% Use instruction mask = roipoly(I) with one of the images: 
mask_water = roipoly(Iblue);
mask_urban = roipoly(Iblue);
mask_vegetation = roipoly(Iblue);

%% Group the 4 images in a signle stack:
stack = cat(3,Iblue,Igreen,Ired,InearIR);

% Crop the stack using each mask: Each crop has a dimension equal to NPIXxNCHANNELS, where NPIX is 
% the number of pixels of the selected region and NCHANNELS the number of
% channels (4): 
stack_crop_water = imstack2vectors(stack,mask_water);
stack_crop_urban = imstack2vectors(stack,mask_urban);
stack_crop_vegetation = imstack2vectors(stack,mask_vegetation);

% Create an image with all the pixels (a matrix full os 1's)
mask_all = ones(size(Iblue));

% Cropping the stacked image (4 channels) with masl_all, so we get all the
% image.
stack_all = imstack2vectors(stack, mask_all);

% compute covariance matrices and means of each training crop: 
[cov_water mean_water] = covmatrix(stack_crop_water);
[cov_urban mean_urban] = covmatrix(stack_crop_urban);
[cov_vegetation mean_vegetation] = covmatrix(stack_crop_vegetation);

%% train a Gaussian discriminant classifier: 
% Use toolbox instruction d = bayesgauss(X, CA, MA, P):
%   D = BAYESGAUSS(X, CA, MA, P) computes the Bayes decision
%   functions of the n-dimensional patterns in the rows of X. 
%   CA is an array of size n-by-n-by-W containing W covariance
%   matrices of size n-by-n, where W is the number of classes.
%   MA is an array of size n-by-W, whose columns are the corres-
%   ponding mean vectors. A cov. matrix and a mean vector must be 
%   specified for each class, even is some are equal.  X is of size 
%   K-by-n, where K is the number of patterns to be classified. P is 
%   a 1-by-W array, containing the probabilities of occurrence of 
%   each class.  If P is not included in the argument, the classes 
%   are assumed to be equally likely.  
%   D, is a column vector of length K. Its ith element is the class
%   number assigned to the ith vector in X during classification.  

% Construct a stack with all the covariance matrices and means.
CA = cat(3,cov_water,cov_urban,cov_vegetation);
MA = cat(2,mean_water,mean_urban,mean_vegetation);

result_watercrop = bayesgauss(stack_crop_water,CA,MA); 
result_urbancrop = bayesgauss(stack_crop_urban,CA,MA); 
result_vegetationcrop = bayesgauss(stack_crop_vegetation,CA,MA);


% Let's tell the full image, each pixel class.
result_all = bayesgauss(stack_all, CA, MA);

% We need to reshape the vector result all, we have a list of pixels
% classified, but in vector form, we have to shape it like the original
% image, for example Iblue.
Ipredict = reshape(result_all, size(Iblue));

% We build original image, in order to display it and compare it .
Ioriginal = Ired;
Ioriginal(:,:,2) = Igreen;
Ioriginal(:,:,3) = Iblue;
% Another way could be Ioriginal = car(3, Ired, Igreem, Iblue).

% Plotting result.
figure(1);
ax1 = subplot(1,2,1); imagesc(Ioriginal);
ax2 = subplot(1,2,2); imagesc(Ipredict);

%% Now let's try k-means method:
% k-means gets R G B IR info from each pixel, and asociates together (clusterizes) the
% pixels with similar combinations for R G B IR.
n_clusters = 3;
prediction_k = kmeans(double(stack_all), n_clusters);
Ipredict2 = reshape(prediction_k, size(Iblue));
% Plotting.
figure(2);
ax1 = subplot(1,2,1); imagesc(Ioriginal);
ax2 = subplot(1,2,2); imagesc(Ipredict2);

%% Let's try GMM;
% We create a gmdistribution object.
options = statset('MaxIter',1000);
GMModel = fitgmdist(double(stack_all), 3, 'Options', options);
% Let's get the prediction about each pixel.
prediction_gmm = cluster(GMModel,double(stack_all));
Ipredict3 = reshape(prediction_gmm, size(Iblue));
% Plotting
figure(3);
ax1 = subplot(1,2,1); imagesc(Ioriginal);
ax2 = subplot(1,2,2); imagesc(Ipredict3);

%% Let's find best cluster quantity for BIC and AIC minimization.
n = 6;
AIC = zeros(1,n);
BIC = zeros(1,n);
gm = cell(1,n);
options = statset('MaxIter',1000);
for k = 1:n
    gm{k} = fitgmdist(double(stack_all), k, 'Options', options);
    AIC(k)= gm{k}.AIC;
    BIC(k) = gm{k}.BIC;
end
[minAIC,numComponentsAIC] = min(AIC);
numComponentsAIC
[minBIC,numComponentsBIC] = min(BIC);
numComponentsBIC
% Seems that best cluster size is 6. Too much detail, but let's test it.
% k-means 6 clusters.
n_clusters = numComponentsBIC;
prediction_k = kmeans(double(stack_all), n_clusters);
Ipredict4 = reshape(prediction_k, size(Iblue));
figure(4);
ax1 = subplot(1,2,1); imagesc(Ioriginal);
ax2 = subplot(1,2,2); imagesc(Ipredict4);
% GMM 6 clusters.
GMModel = fitgmdist(double(stack_all), n_clusters);
prediction_gmm = cluster(GMModel,double(stack_all));
Ipredict5 = reshape(prediction_gmm, size(Iblue));
figure(5);
ax1 = subplot(1,2,1); imagesc(Ioriginal);
ax2 = subplot(1,2,2); imagesc(Ipredict5);


%% compute classification performance in the training set:
class_water_as_water = 100*numel(find(result_watercrop == 1))/length(result_watercrop); 
class_water_as_urban = 100*numel(find(result_watercrop == 2))/length(result_watercrop); 
class_water_as_vegetation = 100*numel(find(result_watercrop == 3))/length(result_watercrop); 

class_urban_as_water = 100*numel(find(result_urbancrop == 1))/length(result_urbancrop); 
class_urban_as_urban = 100*numel(find(result_urbancrop == 2))/length(result_urbancrop); 
class_urban_as_vegetation = 100*numel(find(result_urbancrop == 3))/length(result_urbancrop); 

class_vegetation_as_water = 100*numel(find(result_vegetationcrop == 1))/length(result_vegetationcrop); 
class_vegetation_as_urban = 100*numel(find(result_vegetationcrop == 2))/length(result_vegetationcrop); 
class_vegetation_as_vegetation = 100*numel(find(result_vegetationcrop == 3))/length(result_vegetationcrop); 

% Now validate the classification results by applying the classifier to sets of pixels of
% each class different to those used for the training: 



