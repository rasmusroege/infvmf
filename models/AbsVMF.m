classdef AbsVMF < handle
    properties
        
    end
    
    methods
        function obj=AbsVMF()
        end
        
        function logQ=sample_tau0(obj,x,maxiter)
            logQ=0;
            stepsize=0.1;
            m=copy(obj);
            for i=1:maxiter
                l=1;
                newtau0=exp(log(obj.par.sub(l).tau0)+stepsize*randn());
                
                for l=1:obj.par.nsubjects
                    m.par.sub(l).tau0=newtau0;
                end
                m.calcss(x,1:obj.par.nsubjects);
                if rand<exp(m.llh-obj.llh)
                    logQ=logQ+m.llh-obj.llh;
                    obj.par.sub=m.par.sub;
                    obj.ss=m.ss;
                    obj.logPc=m.logPc;
                end
            end
        end
        
        function logQ=sample_ab(obj,x,maxiter)
            logQ=0;
            stepsize=0.1;
            abdiff=1e-6;
            for i=1:maxiter
                m=copy(obj);
                l=1;
                newa=exp(log(obj.par.a(l))+stepsize*randn());
                while newa<=obj.par.b(l)+abdiff
                    newa=exp(log(obj.par.a(l))+stepsize*randn());
                end
                m.par.a=newa*ones(size(m.par.a));
                m.calcss(x,1:m.par.nsubjects);
                if rand<exp(m.llh-obj.llh)
                    logQ=logQ+m.llh-obj.llh;
                    obj.par.a=m.par.a;
                    obj.ss=m.ss;
                    obj.logPc=m.logPc;
                    
                end
                m=copy(obj);
                newb=obj.par.a(l)+1;
                while newb>=obj.par.a(l)-abdiff
                    newb=exp(log(obj.par.b(l))+stepsize*randn());
                end
                m.par.b=newb*ones(size(m.par.a));
                m.calcss(x,1:m.par.nsubjects)
                if rand<exp(m.llh-obj.llh)
                    logQ=logQ+m.llh-obj.llh;
                    obj.par.b=m.par.b;
                    obj.ss=m.ss;
                    obj.logPc=m.logPc;             
                end
            end
        end  
        
        function samples=samplefromprior(obj,a,b,M,x0)
            T=obj.par.T;
            nburnin=1000;
            stepsize=0.2;
            trimming=20;
            for i=1:nburnin
                x=exp(log(x0)+stepsize*randn);
                if rand<exp(obj.logprior(T,x,a,b)-obj.logprior(T,x0,a,b))
                    x0=x;
                end
            end
            x0=x;
            stepsize=0.1;
            samples=zeros(M,1);
            for i=1:M
                for j=1:trimming
                    x=exp(log(x0)+stepsize*randn);
                    while rand>exp(obj.logprior(T,x,a,b)-obj.logprior(T,x0,a,b))
                        x=exp(log(x0)+stepsize*randn);
                    end
                    x0=x;
                end
                x0=x;
                samples(i)=x;
            end
        end        
        
        function samples=mhsampler(f,x0,M,nburnin,trimming)
            stepsize=0.1;
            for i=1:nburnin
                x=exp(log(x0)+stepsize*randn);
                if rand<exp(f(x)-f(x0))
                    x0=x;
                end
            end
            x0=x;
            stepsize=0.1;
            samples=zeros(M,1);
            for i=1:M
                for j=1:trimming
                    x=exp(log(x0)+stepsize*randn);
                    while rand>exp(f(x)-f(x0))
                        x=exp(log(x0)+stepsize*randn);
                    end
                    x0=x;
                end
                x0=x;
                samples(i)=x;
            end
        end
        
        function res=logcd(obj,x)
            d=obj.par.T/2-1;
            res=d*log(x)-obj.par.T/2*log(2*pi)-obj.logbesseli(d,x);
        end        
        
        function logphi=logphiintegrand(obj,nk,gammak,l)
            logphi=nk*obj.ss(l).logcdtauk-obj.logcd(gammak);
        end        
        
        function logb=logbesseli(obj,v,x)
            if v<10
                logb=zeros(size(x));
                logb(x<100)=log(besseli(v,x(x<100)));
                logb(x>=100)=log(besseli(v,x(x>=100),1))+x(x>=100);
            else
                sq=sqrt(x.^2+(v+1)^2);
                logb=sq+(v+1/2)*log(x./(v+1/2+sq))-1/2*log(x/2)+(v+1/2)*log((2*v+3/2)/(2*(v+1)))-1/2*log(2*pi);
            end
        end
        
        function y=logprior(obj,T,x,a,b)
            y=a*obj.logcd(x)-obj.logcd(b*x);
        end        
    end
end