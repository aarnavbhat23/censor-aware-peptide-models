"""Publication-held-out evaluation, purging exact sequences from all fitting."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from likelihood import CensoredHead,observed_nll
from observations import parse_mic


def load(frame,embedding):
    lookup={str(s):v for s,v in zip(embedding['sequences'],embedding['vectors'])}
    accepted,excluded=[] ,[]
    for i,row in frame.iterrows():
        try: observation=parse_mic(row.mic)
        except ValueError:
            excluded.append({'source_row':int(i),'reason':'unsupported_unit_or_format','raw_mic':row.mic});continue
        accepted.append({'source_row':int(i),'sequence':row.sequence,'paper':row.paper_doi,
                         'lower':observation.lower,'upper':observation.upper,'operator':observation.operator})
    data=pd.DataFrame(accepted)
    return data,np.array([lookup[s] for s in data.sequence],dtype='float32'),excluded


def purged_masks(data,test_paper,validation_paper):
    test=(data.paper==test_paper).to_numpy()
    validation=(data.paper==validation_paper).to_numpy(copy=True)
    forbidden=set(data.sequence[test | validation])
    train=(~(test | validation)) & ~data.sequence.isin(forbidden).to_numpy()
    # Avoid test-to-validation exact matches as well.
    validation &= ~data.sequence.isin(set(data.sequence[test])).to_numpy()
    return train,validation,test


def fit(x,lower,upper,train,validation,mode,seed=417):
    torch.manual_seed(seed)
    loc,scale=x[train].mean(0),np.maximum(x[train].std(0),.05)
    features=torch.tensor((x-loc)/scale);lo=torch.tensor(lower,dtype=torch.float32);hi=torch.tensor(upper,dtype=torch.float32)
    endpoint=torch.where(torch.isfinite(lo),lo,hi)
    model=CensoredHead(x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    best,stale,state,trace=float('inf'),0,None,[]
    for epoch in range(250):
        model.train();optimizer.zero_grad();mean,sigma=model(features[train])
        a,b=(lo[train],hi[train]) if mode=='censored_likelihood' else (endpoint[train],endpoint[train])
        loss=observed_nll(mean,sigma,a,b).mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();model.eval()
        with torch.no_grad():
            mean,sigma=model(features[validation]);value=float(observed_nll(mean,sigma,lo[validation],hi[validation]).mean())
        trace.append({'epoch':epoch+1,'training_objective':float(loss.detach()),'validation_observation_nll':value})
        if value<best:
            best,stale,state=value,0,copy.deepcopy(model.state_dict())
        else:stale+=1
        if stale==25:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():mean,sigma=model(features)
    return mean.numpy(),sigma.numpy(),{'state':state,'location':torch.tensor(loc),'scale':torch.tensor(scale)},trace


def summarize(rows):
    exact=rows.operator=='=';bound=~exact
    mse=float(np.mean((rows.loc[exact,'mean']-rows.loc[exact,'lower'])**2)) if exact.any() else None
    violation=((rows.loc[bound,'mean']<rows.loc[bound,'lower']) | (rows.loc[bound,'mean']>rows.loc[bound,'upper']))
    return {'rows':len(rows),'exact_rows':int(exact.sum()),'censored_rows':int(bound.sum()),
            'observation_nll':float(rows.nll.mean()),'exact_only_rmse':float(np.sqrt(mse)) if mse is not None else None,
            'mean_outside_observed_bound_fraction':float(violation.mean()) if bound.any() else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data/assays.tsv'))
    args=parser.parse_args();torch.set_num_threads(4)
    data,x,excluded=load(pd.read_csv(args.data,sep='\t'),np.load('data/esm35m.npz'))
    papers=sorted(data.paper.unique());out=Path('results');out.mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    records,folds,histories=[],[],{}
    for index,paper in enumerate(papers):
        validation=next(p for p in papers[index+1:]+papers[:index] if p!=paper)
        train,val,test=purged_masks(data,paper,validation)
        fold={'test_paper':paper,'validation_paper':validation,'train_rows':int(train.sum()),
              'validation_rows':int(val.sum()),'test_rows':int(test.sum()),
              'exact_sequence_train_test_overlap':len(set(data.sequence[train])&set(data.sequence[test]))}
        if train.sum()<100 or val.sum()<5:
            fold['status']='excluded_insufficient_training_or_validation';folds.append(fold);continue
        fold['status']='evaluated';fold['selected_epochs']={}
        for name in ['endpoint_gaussian','censored_likelihood']:
            mean,sigma,checkpoint,trace=fit(x,data.lower.to_numpy(),data.upper.to_numpy(),train,val,name)
            checkpoint['width']=x.shape[1];torch.save(checkpoint,out/'checkpoints'/f'fold-{index}-{name}.pt')
            histories[f'{index}/{name}']=trace;fold['selected_epochs'][name]=int(np.argmin([h['validation_observation_nll'] for h in trace]))+1
            nll=observed_nll(torch.tensor(mean[test]),torch.tensor(sigma[test]),torch.tensor(data.lower.to_numpy()[test],dtype=torch.float32),
                             torch.tensor(data.upper.to_numpy()[test],dtype=torch.float32)).numpy()
            for i,n in zip(np.flatnonzero(test),nll):
                records.append({'source_row':int(data.source_row[i]),'paper':paper,'model':name,'operator':data.operator[i],
                                'lower':float(data.lower[i]),'upper':float(data.upper[i]),'mean':float(mean[i]),'sigma':float(sigma[i]),'nll':float(n)})
        folds.append(fold);print('completed',index,paper,flush=True)
    predictions=pd.DataFrame(records);predictions.to_csv(out/'predictions.csv',index=False)
    by_paper=[];report={'source_rows':len(pd.read_csv(args.data,sep='\t')),'eligible_rows':len(data),
        'unique_sequences':data.sequence.nunique(),'papers':len(papers),'excluded_rows':len(excluded),
        'censored_rows':int((data.operator!='=').sum()),'source_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'folds':folds,'models':{}}
    for name,rows in predictions.groupby('model'):
        report['models'][name]=summarize(rows)
        for paper,subset in rows.groupby('paper'):by_paper.append({'paper':paper,'model':name,**summarize(subset)})
    paper_frame=pd.DataFrame(by_paper);paper_frame.to_csv(out/'per-paper.csv',index=False)
    pivot=paper_frame.pivot(index='paper',columns='model',values='observation_nll')
    delta=(pivot.censored_likelihood-pivot.endpoint_gaussian).to_numpy()
    bootstrap=np.random.default_rng(418).choice(delta,size=(5000,len(delta)),replace=True).mean(1)
    report['paper_balanced_nll_difference']={'censored_minus_endpoint':float(delta.mean()),
        'paper_bootstrap_95_interval':np.quantile(bootstrap,[.025,.975]).tolist()}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
    (out/'training-history.json').write_text(json.dumps(histories,indent=2)+'\n')
    print(json.dumps(report['models'],indent=2))


if __name__=='__main__':main()
