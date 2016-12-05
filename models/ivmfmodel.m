classdef ivmfmodel < AbsInfiniteModel & AbsVMF & handle
    
    properties
        par;
        ss;
        logPc;
        logP;
        logZ;
    end
    
    methods
        function obj=ivmfmodel(x,z)
            if nargin==0
                super_args={};
            else
                super_args={x,z};
            end
            obj@AbsInfiniteModel(super_args{:});
            obj.par.hypersamplers=[obj.par.hypersamplers {'sample_tau0','sample_alpha','sample_ab'}];
            if nargin==0
                return;
            end
            obj.par.M=3;
            [a,b]=size(z);
            if b~=1 || a<=1
                error('z must be a column vector');
            end
            obj.par.a=6*ones(obj.par.nsubjects,1);
            obj.par.b=4.7*ones(obj.par.nsubjects,1);
            for l=1:obj.par.nsubjects
                obj.par.sub(l).tau0=0.1;
                obj.par.sub(l).mu0=mean(x{l},2)/norm(mean(x{l},2));
            end
            obj.calcss(x);
        end
        
        function returnobj=copy(obj)
            returnobj=ivmfmodel();
            returnobj.par=obj.par;
            returnobj.ss=obj.ss;
            returnobj.logPc=obj.logPc;
            returnobj.logP=obj.logP;
            returnobj.logZ=obj.logZ;
        end
        
        
        function merge_obj=initMerge(obj,x,z,comp)
            merge_obj=copy(obj);
            merge_obj.par.nk(comp(1))=sum(merge_obj.par.nk(comp));
            merge_obj.par.z=z;
            for l=1:obj.par.nsubjects
                k=comp(1);
                merge_obj.ss(l).xk(:,k)=sum(x{l}(:,merge_obj.par.z==k),2);
                gammak=sqrt(sum(bsxfun(@plus,merge_obj.par.sub(l).mu0*merge_obj.par.sub(l).tau0,merge_obj.ss(l).xk(:,k)*merge_obj.ss(l).xi).^2));
                logvec=merge_obj.logphiintegrand(merge_obj.par.nk(k),gammak,l);
                merge_obj.ss(l).phi(k)=log(sum(exp(logvec-max(logvec))))+max(logvec)-log(obj.par.M);
                merge_obj.logPc(k,l)=merge_obj.ss(l).logconst+merge_obj.ss(l).phi(k);
                
                k=comp(2);
                merge_obj.ss(l).xk(:,k)=[];
                merge_obj.ss(l).phi(k)=[];
                
            end
            k=comp(2);
            merge_obj.logPc(k,:)=[];
            merge_obj.par.nk(k)=[];
            merge_obj.updateLogZ();
        end
        
        function launch_obj=initLaunch(obj,x,z,comp)
            launch_obj=copy(obj);
            launch_obj.par.nk(comp(1),1)=sum(z==comp(1));
            launch_obj.par.nk(comp(2),1)=sum(z==comp(2));
            launch_obj.par.z=z;
            for l=1:obj.par.nsubjects
                for k=comp
                    launch_obj.ss(l).xk(:,k)=sum(x{l}(:,launch_obj.par.z==k),2);
                    gammak=sqrt(sum(bsxfun(@plus,launch_obj.par.sub(l).mu0*launch_obj.par.sub(l).tau0,launch_obj.ss(l).xk(:,k)*launch_obj.ss(l).xi).^2));
                    logvec=launch_obj.logphiintegrand(launch_obj.par.nk(k),gammak,l);
                    launch_obj.ss(l).phi(k,1)=log(sum(exp(logvec-max(logvec))))+max(logvec)-log(obj.par.M);
                end
                launch_obj.logPc(comp,l)=launch_obj.ss(l).logconst+launch_obj.ss(l).phi(comp);
            end
            launch_obj.updateLogZ();
        end
        
        function remove_empty_clusters(obj)
            idx_empty=find(obj.par.nk==0)';
            if ~isempty(idx_empty)
                for j_empty = idx_empty
                    obj.par.nk(j_empty)=[];
                    obj.par.z(obj.par.z>j_empty)=obj.par.z(obj.par.z>j_empty)-1;
                    for l=1:obj.par.nsubjects
                        obj.ss(l).phi(j_empty)=[];
                        obj.ss(l).xk(:,j_empty)=[];
                    end
                    obj.logPc(j_empty,:)=[];
                end
            end
        end
        
        % add observation n to cluster k
        function add_observation(obj,n,k,addss,~)
            if k>length(obj.par.nk)
                obj.par.nk(k,1)=1;
                for l=1:obj.par.nsubjects
                    obj.ss(l).xk(:,k)=addss(l).xk(:,k);
                    obj.ss(l).phi(k,1)=addss(l).phi(k);
                    obj.logPc(k,l)=obj.ss(l).logconst+obj.ss(l).phi(k);
                end
            else
                obj.par.nk(k)=obj.par.nk(k)+1;
                for l=1:obj.par.nsubjects
                    obj.ss(l).xk(:,k)=addss(l).xk(:,k);
                    obj.ss(l).phi(k,1)=addss(l).phi(k);
                    obj.logPc(k,l)=obj.ss(l).logconst+obj.ss(l).phi(k);
                end
            end
        end
        
        function [categoricalDist,logPnew,logdiff,addss]=compute_categorical(obj,x,n,comp)
            K=max(obj.par.z);
            addss=obj.ss;
            
            if obj.par.T/2-1<10
                % low dimensional - use built in bessel function
                if isempty(comp)
                    logPnew=zeros(obj.par.nsubjects,K+1);
                    M=obj.par.M;
                    for l=1:obj.par.nsubjects
                        mu0tmp=obj.par.sub(l).tau0*obj.par.sub(l).mu0;
                        addss(l).xk=bsxfun(@plus,[addss(l).xk zeros(obj.par.T,1)],x{l}(:,n));
                        sumxk=sum(bsxfun(@times,mu0tmp,addss(l).xk));
                        sumxk2=sum(addss(l).xk.^2);
                        gammak=sqrt(sum(mu0tmp.^2)+2*sumxk'*addss(l).xi+sumxk2'*addss(l).xi.^2);
                        logvec=obj.logphiintegrand([obj.par.nk;0]+1,gammak,l);
                        addss(l).phi=log(sum(exp(bsxfun(@minus,logvec,max(logvec,[],2))),2))+max(logvec,[],2)-log(M);
                        logPnew(l,:)=obj.ss(l).logconst+addss(l).phi;
                    end
                    logPnew=logPnew';
                    logdiff=sum(logPnew-[obj.logPc;zeros(1,obj.par.nsubjects)],2);
                    categoricalDist=[obj.par.nk;obj.par.alpha].*exp(logdiff-max(logdiff));
                else
                    logPnew=zeros(2,obj.par.nsubjects);
                    M=obj.par.M;
                    for l=1:obj.par.nsubjects
                        mu0tmp=obj.par.sub(l).tau0*obj.par.sub(l).mu0;
                        addss(l).xk(:,comp)=bsxfun(@plus,addss(l).xk(:,comp),x{l}(:,n));
                        sumxk=sum(bsxfun(@times,mu0tmp,addss(l).xk(:,comp)));
                        sumxk2=sum(addss(l).xk(:,comp).^2);
                        gammak=sqrt(sum(mu0tmp.^2)+2*sumxk'*addss(l).xi+sumxk2'*addss(l).xi.^2);
                        logvec=obj.logphiintegrand(obj.par.nk(comp)+1,gammak,l);
                        addss(l).phi(comp)=log(sum(exp(bsxfun(@minus,logvec,max(logvec,[],2))),2))+max(logvec,[],2)-log(M);
                        logPnew(:,l)=obj.ss(l).logconst+addss(l).phi(comp);
                    end
                    logdiff=sum(logPnew-obj.logPc(comp,:),2);
                    categoricalDist=obj.par.nk(comp).*exp(logdiff-max(logdiff));                    
                end
            else
                l2pi=log(2*pi);
                l2=log(2);
                d=(obj.par.T/2-1);
                l2d3d=log((2*d+3/2)/(2*(d+1)));
                dp5=d+1/2;
                dp1sq=(d+1)^2;
                
                if isempty(comp)
                    logPnew=zeros(obj.par.nsubjects,K+1);
                    M=obj.par.M;
                    nkk=[obj.par.nk;0]+1;
                    s=obj.par.sub;
                    sss=obj.ss;
                    for l=1:obj.par.nsubjects
                        glogcdtauk=sss(l).logcdtauk;
                        tmp=nkk*glogcdtauk+(d+1)*l2pi+1/2*l2+(d+1/2)*l2d3d-1/2*l2pi;
                        addss(l).xk=bsxfun(@plus,[addss(l).xk zeros(obj.par.T,1)],x{l}(:,n));
                        mu0tmp=obj.par.sub(l).tau0*obj.par.sub(l).mu0;
                        sumxk=sum(bsxfun(@times,mu0tmp,addss(l).xk));
                        sumxk2=sum(addss(l).xk.^2);
                        gammakp=sqrt(sum(mu0tmp.^2)+2*sumxk'*addss(l).xi+sumxk2'*addss(l).xi.^2);
                        sqp=sqrt(gammakp.^2+dp1sq);
                        lgammakp=log(gammakp);
                        logvecp=tmp-d*lgammakp+sqp+dp5*(lgammakp-log(dp5+sqp))-1/2*lgammakp;
                        addss(l).phi=log(sum(exp(bsxfun(@minus,logvecp,max(logvecp,[],2))),2))+max(logvecp,[],2)-log(M);
                        logPnew(l,:)=sss(l).logconst+addss(l).phi;
                    end
                    logPnew=logPnew';
                    logdiff=sum(logPnew-[obj.logPc;zeros(1,obj.par.nsubjects)],2);
                    categoricalDist=[obj.par.nk;obj.par.alpha].*exp(logdiff-max(logdiff));
                else
                    logPnew=zeros(2,obj.par.nsubjects);
                    for l=1:obj.par.nsubjects
                        addss(l).xk(:,comp)=bsxfun(@plus,obj.ss(l).xk(:,comp),x{l}(:,n));
                        mu0tmp=obj.par.sub(l).tau0*obj.par.sub(l).mu0;
                        sumxk=sum(bsxfun(@times,mu0tmp,addss(l).xk(:,comp)));
                        sumxk2=sum(addss(l).xk(:,comp).^2);
                        gammakp=sqrt(sum(mu0tmp.^2)+2*sumxk'*addss(l).xi+sumxk2'*addss(l).xi.^2);
                        sqp=sqrt(gammakp.^2+dp1sq);
                        lgammakp=log(gammakp);
                        tmp=(obj.par.nk(comp)+1)*obj.ss(l).logcdtauk+((obj.par.T/2-1)+1)*log(2*pi)+1/2*log(2)+((obj.par.T/2-1)+1/2)*log((2*(obj.par.T/2-1)+3/2)/(2*((obj.par.T/2-1)+1)))-1/2*log(2*pi);
                        logvecp=tmp-d*lgammakp+sqp+dp5*(lgammakp-log(dp5+sqp))-1/2*lgammakp;
                        addss(l).phi(comp)=log(sum(exp(bsxfun(@minus,logvecp,max(logvecp,[],2))),2))+max(logvecp,[],2)-log(obj.par.M);
                        logPnew(:,l)=obj.ss(l).logconst+addss(l).phi(comp);
                    end
                    logdiff=sum(logPnew-obj.logPc(comp,:),2);
                    categoricalDist=obj.par.nk(comp).*exp(logdiff-max(logdiff));
                end
            end
        end
        
        function remove_observation(obj,x,n)
            k=obj.par.z(n);
            obj.par.z(n)=0;
            obj.par.nk(k)=obj.par.nk(k)-1;
            if obj.par.nk(k)==0
                obj.par.nk(k)=[];
                obj.par.z(obj.par.z>k)=obj.par.z(obj.par.z>k)-1;
                for l=1:obj.par.nsubjects
                    obj.ss(l).phi(k)=[];
                    obj.ss(l).xk(:,k)=[];
                end
                obj.logPc(k,:)=[];
            else
                for l=1:obj.par.nsubjects
                    obj.ss(l).xk(:,k)=obj.ss(l).xk(:,k)-x{l}(:,n);
                    gammak=sqrt(sum(bsxfun(@plus,obj.par.sub(l).mu0*obj.par.sub(l).tau0,obj.ss(l).xk(:,k)*obj.ss(l).xi).^2));
                    logvec=obj.logphiintegrand(obj.par.nk(k),gammak,l);
                    obj.ss(l).phi(k)=log(sum(exp(logvec-max(logvec))))+max(logvec)-log(obj.par.M);
                    obj.logPc(k,l)=obj.ss(l).logconst+obj.ss(l).phi(k);
                end
            end
        end
        
        function calcss(obj,x,subjects)
            updatesamplingpoints=1;
            if nargin==2
                updatesamplingpoints=0;
                subjects=1:obj.par.nsubjects;
                val=unique(obj.par.z)';
                val=setdiff(val,0);
                obj.par.nk=zeros(size(val))';
                for k=val
                    obj.par.nk(k,1)=sum(obj.par.z==k);
                end
                K=length(obj.par.nk);
                obj.logPc=zeros(K,obj.par.nsubjects);
            end
            K=length(obj.par.nk);
            val=unique(obj.par.z)';
            val=setdiff(val,0);
            
            updateLogP(obj,x);
            if updatesamplingpoints || ~isa(obj.ss,'struct')
                for l=subjects
                    logcd=@(T,x)(T/2-1)*log(x)-(T/2)*log(2*pi)-logbesseli(T/2-1,x);
                    logprior=@(T,x,a,b)a*logcd(T,x)-logcd(T,b*x);
                    try
                        x0=mean(obj.ss(l).xi);
                    catch e
                        x0=max(fminsearch(@(x)obj.logprior(obj.par.T,10,obj.par.a(l),obj.par.b(l)),10),1);
                    end
                    obj.ss(l).xi=obj.samplefromprior(obj.par.a(l),obj.par.b(l),obj.par.M,x0)';
                    obj.ss(l).logcdtauk=obj.logcd(obj.ss(l).xi);
                    obj.ss(l).logcdbtau=obj.logcd(obj.par.b(l)*obj.ss(l).xi);
                end
            end
            
            for l=subjects
                
                obj.ss(l).logconst=obj.logcd(obj.par.sub(l).tau0);
                obj.ss(l).xk=zeros(obj.par.T,K);
                obj.ss(l).phi=zeros(K,1);
                
                for k=val
                    obj.ss(l).xk(:,k)=sum(x{l}(:,obj.par.z==k),2);
                end
                for k=val
                    gammak=sqrt(sum(bsxfun(@plus,obj.par.sub(l).mu0*obj.par.sub(l).tau0,obj.ss(l).xk(:,k)*obj.ss(l).xi).^2));
                    logvec=obj.logphiintegrand(obj.par.nk(k),gammak,l);
                    obj.ss(l).phi(k)=log(sum(exp(logvec-max(logvec))))+max(logvec)-log(obj.par.M);
                end
                obj.logPc(:,l)=obj.ss(l).logconst+obj.ss(l).phi;
            end
            obj.updateLogZ();
        end
        
        function updateLogP(obj,x)
        end
        
        function predictive_llh=pred(obj,xtest)
            m=obj;
            M=10;K=length(m.par.nk);
            logpred_k=zeros(K,1);
            lp=@(x,m,k)m.par.nk(k)*m.logcd(x)-m.logcd(sqrt(sum(bsxfun(@plus,m.par.sub(1).tau0*m.par.sub(1).mu0,(x'*m.ss(1).xk(:,k)')').^2)))+m.par.a*m.logcd(x)-m.logcd(m.par.b*x);
            for k=1:K
                tauk=mhsampler(@(x)lp(x,m,k),mean(m.ss(1).xi),M,200,10);
                gammak=sqrt(sum(bsxfun(@plus,bsxfun(@times,tauk',m.ss(1).xk(:,k)),m.par.sub(1).tau0+m.par.sub(1).mu0).^2,1)');
                gammakstar=zeros(M,size(xtest,2));
                for i=1:size(xtest,2)
                    gammakstar(:,i)=sqrt(sum(bsxfun(@plus,bsxfun(@times,tauk',m.ss(1).xk(:,k)+xtest(:,i)),m.par.sub(1).tau0+m.par.sub(1).mu0).^2,1)');
                end
                lp_tmp=bsxfun(@minus,m.logcd(tauk)+m.logcd(gammak),m.logcd(gammakstar));
                tmp=log(m.par.nk(k)+m.par.alpha)-log(m.par.N+m.par.alpha)+log(sum(exp(bsxfun(@minus,lp_tmp,max(lp_tmp)))))+max(lp_tmp)-log(M);
                logpred_k(k)=log(sum(exp(tmp-max(tmp))))+max(tmp);
            end
            predictive_llh=log(sum(exp(logpred_k-max(logpred_k))))+max(logpred_k);
        end
        
        %         function predictive_llh=pred(obj,xtest)
        %             m=obj;
        %             M=10;K=length(m.par.nk);
        %             logpred_k=zeros(K+1,1);
        %             lp=@(x,m,k)m.par.nk(k)*m.logcd(x)-m.logcd(sqrt(sum(bsxfun(@plus,m.par.sub(1).tau0*m.par.sub(1).mu0,(x'*m.ss(1).xk(:,k)')').^2)))+m.par.a*m.logcd(x)-m.logcd(m.par.b*x);
        %             for k=1:K
        %                 tauk=mhsampler(@(x)lp(x,m,k),mean(m.ss(1).xi),M,200,10);
        %                 gammak=sqrt(sum(bsxfun(@plus,bsxfun(@times,tauk',m.ss(1).xk(:,k)),m.par.sub(1).tau0+m.par.sub(1).mu0).^2,1)');
        %                 gammakstar=zeros(M,size(xtest,2));
        %                 for i=1:size(xtest,2)
        %                     gammakstar(:,i)=sqrt(sum(bsxfun(@plus,bsxfun(@times,tauk',m.ss(1).xk(:,k)+xtest(:,i)),m.par.sub(1).tau0+m.par.sub(1).mu0).^2,1)');
        %                 end
        %                 lp_tmp=bsxfun(@minus,m.logcd(tauk)+m.logcd(gammak),m.logcd(gammakstar));
        %                 tmp=log(m.par.nk(k))-log(m.par.N+m.par.alpha)+log(sum(exp(bsxfun(@minus,lp_tmp,max(lp_tmp)))))+max(lp_tmp)-log(M);
        %                 logpred_k(k)=log(sum(exp(tmp-max(tmp))))+max(tmp);
        %             end
        %             lp=@(x,m,k)m.par.a*m.logcd(x)-m.logcd(m.par.b*x);
        %             tauk=mhsampler(@(x)lp(x,m,k),mean(m.ss(1).xi),M,200,10);
        %             gammak=m.par.sub(1).tau0;
        %             gammakstar=zeros(M,size(xtest,2));
        %             for i=1:size(xtest,2)
        %                 gammakstar(:,i)=sqrt(sum(bsxfun(@plus,bsxfun(@times,tauk',xtest(:,i)),m.par.sub(1).tau0+m.par.sub(1).mu0).^2,1)');
        %             end
        %             lp_tmp=bsxfun(@minus,m.logcd(tauk)+m.logcd(gammak),m.logcd(gammakstar));
        %             tmp=log(m.par.alpha)-log(m.par.N+m.par.alpha)+log(sum(exp(bsxfun(@minus,lp_tmp,max(lp_tmp)))))+max(lp_tmp)-log(M);
        %             logpred_k(K+1)=log(sum(exp(tmp-max(tmp))))+max(tmp);
        %             predictive_llh=log(sum(exp(logpred_k-max(logpred_k))))+max(logpred_k);
        %         end
    end
end

