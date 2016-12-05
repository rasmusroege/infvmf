classdef AbsVMF_sh < handle
    properties
        
    end
    
    methods
        function obj=AbsVMF_sh()
        end
        
        
        function logQ=sample_tau0(obj,x,maxiter)
            logQ=0;
            stepsize=0.1;
            m=copy(obj);
            for i=1:maxiter
                for l=1:obj.par.nsubjects
                    newtau0=exp(log(obj.par.sub(l).tau0)+stepsize*randn());
                    m.par.sub(l).tau0=newtau0;
                    m.calcss(x,l);
                    if rand<exp(m.llh-obj.llh)
                        logQ=logQ+m.llh-obj.llh;
                        obj.par.sub(l).tau0=newtau0;
                        obj.ss(l)=m.ss(l);
                        obj.logPc(:,l)=m.logPc(:,l);
                    end
                end
            end
        end
        
        function logQ=sample_ab(obj,x,maxiter)
            logQ=0;
            stepsize=0.1;
            abdiff=0.0;
            m=copy(obj);
            for i=1:maxiter
                for l=1:obj.par.nsubjects
                    newa=exp(log(obj.par.a(l))+stepsize*randn());
                    while newa<=obj.par.b(l)+abdiff
                        newa=exp(log(obj.par.a(l))+stepsize*randn());
                    end
                    m.par.a(l)=newa;
                    m.calcss(x,l);
                    if rand<exp(m.llh-obj.llh)
                        logQ=logQ+m.llh-obj.llh;
                        obj.par.a(l)=newa;
                        obj.ss(l)=m.ss(l);
                        obj.logPc(:,l)=m.logPc(:,l);
                    else
                        m.par.a(l)=obj.par.a(l);
                        m.ss(l)=obj.ss(l);
                    end
                    newb=obj.par.a(l)+1;
                    while newb>=obj.par.a(l)-abdiff
                        newb=exp(log(obj.par.b(l))+stepsize*randn());
                    end
                    m.par.b(l)=newb;
                    m.calcss(x,l);
                    if rand<exp(m.llh-obj.llh)
                        logQ=logQ+m.llh-obj.llh;
                        obj.par.b(l)=newb;
                        obj.ss(l)=m.ss(l);
                        obj.logPc(:,l)=m.logPc(:,l);
                    else
                        m.par.b(l)=obj.par.b(l);
                        m.ss(l)=obj.ss(l);
                    end
                end
            end
        end
        
    end
end