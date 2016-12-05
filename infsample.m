function [z,llh,noc,cputime,strout,amis,best_sample]=infsample(X,model,opts)
% [z,llh,noc,cputime,strout,amis,best_sample]=infsample(X,model,opts)

[~,N]=size(X{1});

if nargin<3
    opts=struct;
end
% if isfield(opts,'covType'); par.covType=opts.covType; else par.covType='diag'; end;
if isfield(opts,'Kinit'); Kinit=opts.Kinit; else Kinit=ceil(log(N)); end
if isfield(model.par,'z'); z=model.par.z; else z=ceil(Kinit*rand(N,1)); end
if isfield(opts,'maxiter'); maxiter=opts.maxiter; else; maxiter=3   ;end;
if isfield(opts,'storeExpectations'); storeExpectations=opts.storeExpectations; else storeExpectations=true; end
if isfield(opts,'debug'); debug=opts.debug; else debug=false; end
if isfield(opts,'UseSequentialAllocation'); UseSequentialAllocation=opts.UseSequentialAllocation; else par.UseSequentialAllocation=true; end
if isfield(opts,'startiter'); startiter=opts.startiter; else, startiter=1;end;
if isfield(opts,'llh'); llh=opts.llh; else, llh=[];end;
if isfield(opts,'noc'); noc=opts.noc; else, noc=[];end;
if isfield(opts,'cputime'); cputime=opts.cputime; else, cputime=[];end;
if isfield(opts,'strout'); strout=opts.strout; else, strout={};end;
if isfield(opts,'optim'); optim=opts.optim; else;  optim=0; end;
if isfield(opts,'zt'); zt=opts.zt; else; zt=[]; end;
if isfield(opts,'best_sample');best_sample=opts.best_sample;else;best_sample=model;end;
if isfield(opts,'verbose');verbose=opts.verbose;else;verbose=1;end;
amis=[];
if isempty(zt)
    spacer = [repmat([repmat('-',1,14) '+'],1,4) repmat('-',1,13)];
    dheader = sprintf(' %12s | %12s | %12s | %12s | %12s ','Iteration','logP', 'dlogP/|logP|', 'noc', class(model));
else
    spacer = [repmat([repmat('-',1,14) '+'],1,4) sprintf('%12.3f',calcami(model.par.z,zt)) repmat('-',1,13)];
    dheader = sprintf(' %12s | %12s | %12s | %12s | %12s | %12s','Iteration','logP', 'dlogP/|logP|', 'noc', 'ami', class(model));    
end
getnoc=@(x)length(unique(x.par.z));
if isempty(model.par) && ~isa(model,'gmmgp_model')
    m_tmp=feval(str2func(class(model)),X,z);
    model.par=m_tmp.par;
elseif isempty(model.par)
    m_tmp=feval(str2func(class(model)),X,z,Kinit);
    model.par=m_tmp.par;    
end
model.calcss(X);
if debug && startiter==1
    fprintf('model:\n');
    model,
    fprintf('debug: 1\n');
end
ollh=model.llh;
% Main loop
for iter=startiter:startiter+maxiter-1
    if mod(iter,30)==1 && verbose
        disp(spacer);disp(dheader);disp(spacer);
    end

    % Gibbs sample
    tic;
    if iter>=startiter+maxiter-1-optim
        gibbs_sample(X,model,randperm(N),debug,[],[],optim);
    else
        gibbs_sample(X,model,randperm(N),debug);
    end
    if isa(model,'AbsFiniteModel')
        for k=1:1
%             split_sample(X,model,debug);
        end
    else
        for k=1:max(max(model.par.z),5)
            split_merge_sample(X,model,debug);
        end
    end
    model.calcss(X);
    if iter>0
        sample_hyperparameters(X,model,1);
    end
    
    model.calcss(X);
    cputime=[cputime;toc];
    llh=[llh;model.llh];
    noc=[noc;getnoc(model)];
    
    if model.llh>best_sample.llh
        best_sample=copy(model);
    end
    
    dllh=max((llh(end)-llh(max(end-1,1)))/abs(llh(end)),(llh(end)-ollh)/abs(llh(end)));
    if isempty(zt)
        strout{iter}=sprintf(' %12.0f | %12.4e | %12.4e | %12.0f | %s\n',iter,llh(end), dllh, getnoc(model),datestr(now));
    else
        modelstr=class(model);
        amis=[amis;calcami(model.par.z,zt)];
        strout{iter}=sprintf(' %12.0f | %12.4e | %12.4e | %12.0f | %12.3f | %s\n',iter,llh(end), dllh, getnoc(model),amis(end),datestr(now));
    end
    if verbose
        fprintf('%s',strout{end});
    end
end

%--------------------------------------------------------------------
function model=sample_hyperparameters(X,model,debug)
samplrstrs=model.get_samplers;
for sampler=1:length(samplrstrs)
    feval(str2func(samplrstrs{sampler}),model,X,1);
    if debug
        d=copy(model);d.calcss(X);
        if abs(d.llh-model.llh)/max(abs(d.llh),1)>1e-9
            error(sprintf('error in sampler: %s',samplrstrs{sampler}));
        end
    end
end

%--------------------------------------------------------------------
function split_sample(X,model,debug)
z=model.par.z;
UseSequentialAllocation=1;

N=model.par.N;
i1=ceil(N*rand);
i2=ceil(N*rand);
while z(i2)==z(i1)
    i2=ceil(N*rand);
end
z_t=z;

comp=[z(i1) z(i2)];
idx=(z==z(i1) | z==z(i2));
if UseSequentialAllocation
    z_t(idx)=0;
end
z_t(i1)=comp(1);
z_t(i2)=comp(2);
idx(i1)=false;
idx(i2)=false;
model_split=model.initLaunch(X,z_t,comp);
if sum(idx)>0
    for reps=1:3
        %             [z_t,par_t,logP_t,logQ]
        [logQ]=gibbs_sample(X,model_split,find(idx)',debug,comp);
    end
    if norm(model_split.par.z-model.par.z,'fro')==0
        return;
    end
    model_copy=copy(model_split);
    [logQrev]=gibbs_sample(X,model_copy,find(idx)',debug,comp,z);
else
    logQ=0;
end
if rand<exp(model_split.llh-model.llh-logQ+logQrev);
    disp(['resampled components ' num2str(z(i1)) 'and ' num2str(z(i2)) ' with delta llh: ' num2str(model_split.llh-model.llh)]);
    model.par.z=model_split.par.z;
    model.ss=model_split.ss;
    model.par.nk=model_split.par.nk;
    model.logPc=model_split.logPc;
    model.logZ=model_split.logZ;
    model.logP=model_split.logP;
    model_split.delete();
end

%--------------------------------------------------------------------
function split_merge_sample(X,model,debug)
UseSequentialAllocation=1;

nsubjects=length(X);
[~,N]=size(X{1});
i1=ceil(N*rand);
i2=ceil(N*rand);
while i2==i1
    i2=ceil(N*rand);
end

if model.par.z(i1)==model.par.z(i2) % Split move
    % generate split configuration
    z_t=model.par.z;
    comp=[model.par.z(i1) max(model.par.z)+1];
    idx=(z_t==model.par.z(i1));
    if UseSequentialAllocation
        z_t(idx)=0;
    else
        z_t(idx)=comp(ceil(2*rand(sum(idx),1)));
    end
    z_t(i1)=comp(1);
    z_t(i2)=comp(2);
    idx(i1)=false;
    idx(i2)=false;
    
    model_split=model.initLaunch(X,z_t,comp);
    
    if sum(idx)>0
        for reps=1:3
%             [z_t,par_t,logP_t,logQ]
            [logQ]=gibbs_sample(X,model_split,find(idx)',debug,comp);
        end
    else
        logQ=0;
    end
    dllh=model_split.llh-model.llh;
    if rand<exp(dllh-logQ)
        disp(['split component ' num2str(model.par.z(i1)) ' with delta llh: ' num2str(dllh)]);
        model.par.z=model_split.par.z;
        model.ss=model_split.ss;
        model.par.nk=model_split.par.nk;
        model.logPc=model_split.logPc;
        model.logZ=model_split.logZ;
        model.logP=model_split.logP;
        model_split.delete();
    end
else% merge move
    % generate merge configuration
    if model.par.z(i1)>model.par.z(i2);tmp=i2;i2=i1;i1=tmp;clear tmp;end;
    comp=[model.par.z(i1) model.par.z(i2)];
    z_merge=model.par.z;
    idx=(z_merge==z_merge(i1) | z_merge==z_merge(i2));
    z_merge(idx)=z_merge(i1);
    z_merge(z_merge>model.par.z(i2))=z_merge(z_merge>model.par.z(i2))-1;
    model_merge=model.initMerge(X,z_merge,comp);
    model_merge.updateLogZ;
    
    accept_rate=rand();
    if accept_rate<exp(model_merge.llh-model.llh)
        z_t=model.par.z;
        if UseSequentialAllocation
            z_t(idx)=0;
        else
            z_t(idx)=comp(ceil(2*rand(sum(idx),1)));
        end
        z_t(i1)=model.par.z(i1);
        z_t(i2)=model.par.z(i2);
        idx(i1)=false;
        idx(i2)=false;
        model_launch=model.initLaunch(X,z_t,comp);
        if sum(idx)>0
            for reps=1:2
                gibbs_sample(X,model_launch,find(idx)',debug,comp);
            end
            [logQ]=gibbs_sample(X,model_launch,find(idx)',debug,comp,model.par.z);
        else
            logQ=0;
        end
        
        if accept_rate<exp(model_merge.llh-model.llh+logQ);
            disp(['merged component ' num2str(model.par.z(i1)) ' with component ' num2str(model.par.z(i2)) ' with delta llh: ' num2str(model_merge.llh-model.llh)]);
            model.par.z=model_merge.par.z;
            model.ss=model_merge.ss;
            model.par.nk=model_merge.par.nk;
            model.logPc=model_merge.logPc;
            model.logZ=model_merge.logZ;
            model.logP=model_merge.logP;
            model_launch.delete();
            model_merge.delete();
        end
    end
end
% remove empty clusters
model.remove_empty_clusters();
z=model.par.z;
if debug
    dmodel=model.copy();
    dmodel.par.z=z;
    dmodel.calcss(X);
    if abs(dmodel.llh-model.llh)/abs(dmodel.llh)>1e-9
        error(sprintf('error in split-merge'));
    end
end

%----------------------------------------------------------------------------

function [logQ]=gibbs_sample(X,model,sample_idx,debug,comp,forced,optim)
if nargin<7
    optim=0;
end
if nargin<6
    forced=[];
    optim=0;
end
if nargin<5
    comp=[];
    optim=0;
end
logQ=0;
% gibbs sample clusters
for n=sample_idx
    if model.par.z(n)~=0
        model.remove_observation(X,n)
    end
    
    % Evaluate the assignment of i'th covariance matrix to all clusters
    [categoricalDist,logPnew,logDiff,addss]=model.compute_categorical(X,n,comp);
    
    if debug 
        if isa(model,'AbsFiniteModel')
            if isempty(comp)
                noc1=ceil(model.par.K*rand);
                noc2=ceil(model.par.K*rand);
                aa=logDiff+log(model.par.nk+model.par.alpha/model.par.K);
                qdiff_s=aa(noc1)-aa(noc2);
            else
                noc1=comp(1);
                noc2=comp(2);
                aa=logDiff+log(model.par.nk(comp)+model.par.alpha/model.par.K);
                qdiff_s=aa(1)-aa(2);
            end
        else
            if isempty(comp)
                noc1=ceil((max(model.par.z)+1)*rand);
                noc2=ceil((max(model.par.z)+1)*rand);
                tmp=[noc1 noc2];
                while min(model.par.nk(tmp(tmp<=length(model.par.nk))))<=0
                    noc1=ceil((max(model.par.z)+1)*rand);
                    noc2=ceil((max(model.par.z)+1)*rand);
                    tmp=[noc1 noc2];
                end
                
                aa=logDiff+log([model.par.nk;model.par.alpha]);
                qdiff_s=aa(noc1)-aa(noc2);
            else
                noc1=comp(1);
                noc2=comp(2);
                aa=logDiff+log(model.par.nk(comp));
                qdiff_s=aa(1)-aa(2);
            end
        end

        m1=model.copy();
        m1.par.z(n)=noc1;
        m1.remove_empty_clusters();
        m1.calcss(X);

        m2=model.copy();
        m2.par.z(n)=noc2;
        m2.remove_empty_clusters();
        m2.calcss(X);
        
        qdiff=m1.llh-m2.llh;

        if abs(qdiff_s-qdiff)/max(abs(m1.llh),10)>1e-8
            error('gibbs_sampling, %s',class(model));
        end
        
        if any(imag(logPnew(:)))
            error('gibbs_sampling, %s',class(model));
        end
    end
    
    % sample from posterior
    if isempty(comp)
        if optim
            [~,knew]=max(categoricalDist);
            model.par.z(n)=knew;
        else
            model.par.z(n)=find(rand<cumsum(categoricalDist/sum(categoricalDist)),1,'first');
        end
        model.logPc(model.par.z(n),:)=logPnew(model.par.z(n),:);
    else
        if ~isempty(forced)
            model.par.z(n)=forced(n);
        else
            model.par.z(n)=comp(find(rand<cumsum(categoricalDist/sum(categoricalDist)),1,'first'));
        end
        q_tmp=logDiff-max(logDiff)+log(model.par.nk(comp));
        q_tmp=q_tmp-log(sum(exp(q_tmp)));
        logQ=logQ+q_tmp(model.par.z(n)==comp);
        model.logPc(model.par.z(n),:)=logPnew(comp==model.par.z(n),:);        
    end

    % Update sufficient statistics
    model.add_observation(n,model.par.z(n),addss,comp);
    
    % remove empty clusters
    model.remove_empty_clusters();
end
model.updateLogZ;
model.updateLogP(X);
if debug
    dmodel=model.copy();
    dmodel.calcss(X);
    if abs(dmodel.llh-model.llh)/abs(dmodel.llh)>1e-6
        error(sprintf('error in gibbs sample'));
    end
end