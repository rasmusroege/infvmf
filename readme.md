# Infinite von Mises-Fisher Mixture Modeling in MATLAB
Gibbs sampler with split-merge moves for the infinite von Mises-Fisher clustering model. 

The script demo_sphere.m is a demo clustering points on the unit sphere. In general, if x is a (M,1) cell array where each entry is a (P,N) matrix with N observations in a P dimensional space, then the clustering is performed and will store the inferred clustering the variable z by the commands:

K=10;
N=size(x{1},2);
m=ivmfmodel(x,randi(K,N,1));
z=infsample(x,m);

A paper using this code will hopefully get published and may be referenced as: 

Roege, R. E., Madsen, K. H., Schmidt, M. N., Moerup, M. (Dec. 2016) Infinite von Mises-Fisher Modeling of Whole-Brain fMRI Data, Submitted.
