import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from observations import parse_mic

MODEL='facebook/esm2_t12_35M_UR50D'
REVISION='6fbf070e65b0b7291e7bbcd451118c216cff79d8'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data/assays.tsv'))
    parser.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu')
    args=parser.parse_args();sequences=set()
    for row in csv.DictReader(args.data.open(),delimiter='\t'):
        try: parse_mic(row['mic'])
        except ValueError: continue
        sequences.add(row['sequence'])
    sequences=sorted(sequences)
    if any(set(s)-set('ACDEFGHIKLMNPQRSTVWY') or not 8<=len(s)<=100 for s in sequences):
        raise ValueError('Unexpected peptide alphabet or length')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION,add_pooling_layer=False).to(args.device).eval()
    vectors=[]
    with torch.inference_mode():
        for offset in range(0,len(sequences),32):
            tokens=tokenizer(sequences[offset:offset+32],padding=True,return_tensors='pt',return_special_tokens_mask=True)
            special=tokens.pop('special_tokens_mask').to(args.device)
            tokens={k:v.to(args.device) for k,v in tokens.items()}
            hidden=model(**tokens).last_hidden_state;mask=tokens['attention_mask'].bool() & ~special.bool()
            vectors.extend(((hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)).cpu().numpy())
    output=Path('data/esm35m.npz');np.savez_compressed(output,sequences=np.array(sequences),vectors=np.array(vectors))
    output.with_suffix('.manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'sequences':len(sequences),
        'dimension':480,'pooling':'Final layer residue mean, excluding special tokens and padding',
        'sha256':hashlib.sha256(output.read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
