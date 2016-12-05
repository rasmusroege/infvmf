% demoscript for clustering 3D spherical data using the von Mises-Fisher clustering model
addpath models/;
addpath utils/;

% generate data

rng(5);

N=600;
K=6;
T=3;
tauk=[15 20 40 50 50 15];
tau0=0.1;
mu0=[0 0 1];

z=kron(1:K,ones(1,N/K))';
muk={zeros(T,K)};
x={zeros(T,N)};
for k=1:K
	muk{1}=vmfrand(T,K,tau0,mu0);
	x{1}(:,z==k)=vmfrand(T,N/K,tauk(k),muk{1}(:,k));
end

% run inference
o=struct();
o.maxiter=10;
o.zt=z;
m=vmfmodel(x,randi(K,N,1),K);
infsample(x,m,o);

% plot sphere
figure;hold on;
[xx,yy,zz] = sphere(50);
surface(xx,yy,zz,'facecolor','w','facealpha',0.5,'EdgeColor',0.85*[1 1 1],'linewidth',2);
hold on;
axis equal; axis off;
% plot data
mcolors=jet(max(max(m.par.z),K));
for k=1:K
    h{k}=plot3(muk{1}(1,k),muk{1}(2,k),muk{1}(3,k),'o','markeredgecolor',mcolors(k,:),'markerfacecolor',[0 0 0],'markersize',20,'linewidth',5);
end
for i=1:N
    plot3(x{1}(1,i),x{1}(2,i),x{1}(3,i),'o','markerfacecolor',mcolors(m.par.z(i),:),'markeredgecolor',mcolors(z(i),:),'markersize',14,'linewidth',3);
end
