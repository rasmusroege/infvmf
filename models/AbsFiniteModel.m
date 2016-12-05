classdef AbsFiniteModel < handle
    methods
        function obj=AbsFiniteModel(x,z,K)
            obj.par.hypersamplers={'sample_alpha'};
            if nargin==0
                return;
            end
            obj.par.K=K;
            obj.par.nsubjects=length(x);            
            obj.par.z=z;
            obj.par.alpha=5*K;
            [obj.par.T,obj.par.N]=size(x{1});
        end
        
        function samplerstr=get_samplers(obj)
            samplerstr=obj.par.hypersamplers;
        end        

        function logQ=sample_alpha(obj,~,maxiter)
            logQ=0;
            step_size=0.2;
            for i=1:maxiter
                alpha_new=exp(log(obj.par.alpha)+step_size*randn());
                new_logZ=gammaln(alpha_new)-gammaln(alpha_new+obj.par.N)-obj.par.K*gammaln(alpha_new/obj.par.K)+sum(gammaln(alpha_new/obj.par.K+obj.par.nk));
                if rand()<exp(new_logZ-obj.logZ)
                    logQ=logQ+new_logZ-obj.logZ;
                    obj.par.alpha=alpha_new;
                    obj.logZ=new_logZ;
                end
            end
        end

        function updateLogZ(obj)
            obj.logZ=gammaln(obj.par.alpha)-gammaln(obj.par.alpha+obj.par.N)-obj.par.K*gammaln(obj.par.alpha/obj.par.K)+sum(gammaln(obj.par.alpha/obj.par.K+obj.par.nk));
        end

        function llh=llh(obj)
            llh=sum(obj.logPc(:))+sum(obj.logP)+obj.logZ;
        end
    end
end
