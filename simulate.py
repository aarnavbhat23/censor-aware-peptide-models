"""An explicit synthetic sanity check, separate from the literature experiment."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr


def main():
    rng=np.random.default_rng(419);truth=rng.normal(1.5,.6,20000);cutoff=1.2
    censored=truth>cutoff;observed=np.minimum(truth,cutoff)
    def objective(parameters):
        mean,logscale=parameters;scale=np.exp(logscale);z=(observed-mean)/scale
        values=.5*z*z+logscale+.5*np.log(2*np.pi)
        values[censored]=-log_ndtr((mean-cutoff)/scale)
        return values.mean()
    result=minimize(objective,[observed.mean(),np.log(observed.std())],method='BFGS')
    if not result.success:raise RuntimeError(result.message)
    report={'synthetic':True,'seed':419,'rows':len(truth),'true_mean':1.5,'true_scale':.6,'censor_at':cutoff,
            'censored_fraction':float(censored.mean()),'endpoint_mean':float(observed.mean()),
            'endpoint_scale':float(observed.std()),'censor_aware_mean':float(result.x[0]),
            'censor_aware_scale':float(np.exp(result.x[1])),
            'note':'Known iid Gaussian simulation; this does not establish performance on literature data.'}
    Path('results/synthetic-sanity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
