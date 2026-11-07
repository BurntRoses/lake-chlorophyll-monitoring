"""Training-only feature expression and reference statistical models."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import SplineTransformer, OneHotEncoder

class CommonExpression:
    """All numeric/categorical preprocessing is fitted solely on each training side."""
    def __init__(self, feature_spec, config):
        self.spec=feature_spec
        self.config=config

    def fit(self, frame):
        self.numeric=[x for x in self.spec.feature_name if x!='ecoregion']
        a=frame[self.numeric].to_numpy(dtype=np.float64)
        a[~np.isfinite(a)]=np.nan
        with np.errstate(all='ignore'):
            self.lower=np.nanquantile(a,self.config['training_quantile_clip'][0],axis=0)
            self.upper=np.nanquantile(a,self.config['training_quantile_clip'][1],axis=0)
            self.median=np.nanmedian(a,axis=0)
        self.median=np.nan_to_num(self.median,nan=0.0)
        self.lower=np.where(np.isfinite(self.lower),self.lower,0.0)
        self.upper=np.where(np.isfinite(self.upper),self.upper,0.0)
        self.spline_names=[x for x in self.config['spline_features'] if x in self.numeric and self.upper[self.numeric.index(x)]>self.lower[self.numeric.index(x)]]
        self.spline_indices=[self.numeric.index(x) for x in self.spline_names]
        z=self._impute(a)
        self.spline=SplineTransformer(n_knots=self.config['spline_knots'],degree=self.config['spline_degree'],knots='quantile',include_bias=False,extrapolation='constant') if self.spline_indices else None
        if self.spline is not None:self.spline.fit(z[:,self.spline_indices])
        self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
        self.encoder.fit(frame[['ecoregion']].fillna('__missing__').astype(str))
        raw,names,roles=self._expand(a,frame)
        self.keep=np.nanstd(raw,axis=0)>1e-10
        self.names=np.asarray(names)[self.keep].tolist()
        self.dynamic=np.asarray(roles)[self.keep]=='dynamic'
        self.mean=raw[:,self.keep].mean(axis=0)
        self.scale=raw[:,self.keep].std(axis=0)
        return self

    def _impute(self,a):
        z=np.where(np.isfinite(a),a,self.median)
        return np.clip(z,self.lower,self.upper)

    def _expand(self,a,frame):
        z=self._impute(a)
        lookup=self.spec.set_index('feature_name').role.to_dict()
        parts=[z,~np.isfinite(a)];names=self.numeric+[x+'__missing' for x in self.numeric]
        roles=[lookup[x] for x in self.numeric]*2
        if self.spline is not None:
            s=self.spline.transform(z[:,self.spline_indices]);parts.append(s)
            width=s.shape[1]//len(self.spline_names)
            names += [f'{x}__spline{k}' for x in self.spline_names for k in range(width)]
            roles += [lookup[x] for x in self.spline_names for k in range(width)]
        for x,y in self.config['interactions']:
            if x in self.numeric and y in self.numeric:
                xi=self.numeric.index(x);yi=self.numeric.index(y)
                xz=(z[:,xi]-self.median[xi])/max(self.upper[xi]-self.lower[xi],1e-5)
                yz=(z[:,yi]-self.median[yi])/max(self.upper[yi]-self.lower[yi],1e-5)
                parts.append((xz*yz)[:,None]);names.append(x+'__x__'+y)
                roles.append('dynamic' if lookup[x]=='dynamic' or lookup[y]=='dynamic' else 'base')
        cats=self.encoder.transform(frame[['ecoregion']].fillna('__missing__').astype(str));parts.append(cats)
        names += self.encoder.get_feature_names_out().tolist();roles += ['base']*cats.shape[1]
        return np.hstack(parts),names,roles

    def transform(self,frame):
        a=frame[self.numeric].to_numpy(dtype=np.float64);a[~np.isfinite(a)]=np.nan
        z,_,_=self._expand(a,frame)
        return np.asarray((z[:,self.keep]-self.mean)/self.scale,dtype=np.float64,order='C')




class StandardElasticNet:
    """The same unweighted elastic-net logistic objective as sklearn/SAGA.

    An exact positive-negative L1 split permits bounded L-BFGS optimization.
    The sklearn C parameter maps to alpha=1/(C*n) for mean log loss and an
    unpenalized intercept. Saved models contain coefficient arrays.
    """
    def __init__(self, C=0.3, l1_ratio=0.15, tol=1e-6, max_iter=12000):
        self.C=C;self.l1_ratio=l1_ratio;self.tol=tol;self.max_iter=max_iter

    def fit(self, X, y, initial_coefficient=None):
        n,p=X.shape;alpha=1.0/(self.C*n)
        # w=u-v, u>=0, v>=0: an exact smooth, bound-constrained formulation
        # of the L1 term. BLAS matrix products avoid long serial coordinate sweeps.
        def objective(theta):
            w=theta[:p]-theta[p:2*p];z=X@w+theta[-1]
            residual=expit(z)-y
            g=X.T@residual/n+alpha*(1-self.l1_ratio)*w
            value=(np.mean(np.logaddexp(0,z)-y*z)+alpha*self.l1_ratio*theta[:2*p].sum()+
                   alpha*(1-self.l1_ratio)/2*(w@w))
            return value,np.r_[g+alpha*self.l1_ratio,-g+alpha*self.l1_ratio,residual.mean()]
        initial=np.r_[np.zeros(2*p),np.log(y.mean()/(1-y.mean()))]
        if initial_coefficient is not None:
            w,b=initial_coefficient
            if np.asarray(w).shape!=(p,):raise ValueError('Warm-start feature dimension differs')
            initial=np.r_[np.maximum(w,0),np.maximum(-w,0),float(b)]
        iteration=[0]
        def callback(theta):
            iteration[0]+=1
            if iteration[0]%100==0:print('ELASTIC_NET_ITERATION',iteration[0],flush=True)
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',
                        bounds=[(0,None)]*(2*p)+[(None,None)],callback=callback,
                        options={'maxiter':self.max_iter,'ftol':1e-15,'gtol':self.tol*.1,'maxls':40})
        self.coef_=(result.x[:p]-result.x[p:2*p])[None,:]
        self.intercept_=float(result.x[-1])
        self._check_solution(X,y,alpha)
        return self


    def _check_solution(self,X,y,alpha):
        z=X@self.coef_[0]+self.intercept_
        gradient=X.T@(expit(z)-y)/len(y)+alpha*(1-self.l1_ratio)*self.coef_[0]
        nz=self.coef_[0]!=0
        violations=np.maximum(np.abs(gradient)-alpha*self.l1_ratio,0)
        violations[nz]=np.abs(gradient[nz]+alpha*self.l1_ratio*np.sign(self.coef_[0,nz]))
        self.kkt_max=float(max(violations.max(),abs(np.mean(expit(z)-y))))
        self.objective=float(np.mean(np.logaddexp(0,z)-y*z)+
            alpha*(self.l1_ratio*np.abs(self.coef_).sum()+
                   (1-self.l1_ratio)/2*np.square(self.coef_).sum()))
        self.converged=self.kkt_max<=self.tol

    def predict_proba(self,X):
        p=expit(X@self.coef_[0]+self.intercept_)
        return np.column_stack([1-p,p])
