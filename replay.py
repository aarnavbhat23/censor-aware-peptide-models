import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from likelihood import observed_nll
from benchmark import summarize


def main():
    rows=pd.read_csv('results/predictions.csv');report=json.loads(Path('results/metrics.json').read_text())
    for fold in report['folds']:assert fold['exact_sequence_train_test_overlap']==0
    actual=observed_nll(torch.tensor(rows['mean'].to_numpy(),dtype=torch.float32),
        torch.tensor(rows.sigma.to_numpy(),dtype=torch.float32),torch.tensor(rows.lower.to_numpy(),dtype=torch.float32),
        torch.tensor(rows.upper.to_numpy(),dtype=torch.float32)).numpy()
    np.testing.assert_allclose(actual,rows.nll.to_numpy(),atol=1e-5,rtol=1e-5)
    for model,group in rows.groupby('model'):
        for key,value in summarize(group).items():
            assert np.isclose(value,report['models'][model][key]),(model,key)
    synthetic=json.loads(Path('results/synthetic-sanity.json').read_text())
    assert abs(synthetic['censor_aware_mean']-synthetic['true_mean'])<.04
    assert abs(synthetic['endpoint_mean']-synthetic['true_mean'])>.3
    print('Replayed censored likelihoods, exact-only errors, bound violations and simulation sanity check.')


if __name__=='__main__':main()
