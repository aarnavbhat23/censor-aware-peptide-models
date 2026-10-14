import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModel,AutoTokenizer
from encode import MODEL,REVISION
from likelihood import CensoredHead


def main():
    parser=argparse.ArgumentParser();parser.add_argument('sequence');parser.add_argument('--fold',type=int,default=0)
    args=parser.parse_args()
    if set(args.sequence)-set('ACDEFGHIKLMNPQRSTVWY') or not 8<=len(args.sequence)<=100:
        parser.error('Expected 8–100 canonical amino acids')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    encoder=AutoModel.from_pretrained(MODEL,revision=REVISION,add_pooling_layer=False).eval()
    tokens=tokenizer([args.sequence],return_tensors='pt',return_special_tokens_mask=True);special=tokens.pop('special_tokens_mask')
    with torch.no_grad():
        hidden=encoder(**tokens).last_hidden_state;mask=tokens['attention_mask'].bool() & ~special.bool()
        vector=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
    predictions={}
    for name in ['endpoint_gaussian','censored_likelihood']:
        checkpoint=torch.load(Path('results/checkpoints')/f'fold-{args.fold}-{name}.pt',map_location='cpu',weights_only=True)
        model=CensoredHead(checkpoint['width']).eval();model.load_state_dict(checkpoint['state'])
        with torch.no_grad():mean,sigma=model((vector-checkpoint['location'])/checkpoint['scale'])
        predictions[name]={'mean_log10_mic_uM':mean.item(),'gaussian_scale':sigma.item()}
    print(json.dumps({'fold':args.fold,'models':predictions,'note':'Marginal literature-table model; lacks organism and peptide chemistry conditioning. No efficacy or safety guarantee.'},indent=2))


if __name__=='__main__':main()
