import hashlib
import urllib.request
from pathlib import Path
import truststore

URL='https://raw.githubusercontent.com/aarnavbhat23/spear-wetlab-false-positive-audit/5ac295c08f48aef43c991ba40a706f0e2fedb5b7/data/df_wetlab_heatmap_results.tsv'
SHA256='f873b6a70745d1c29151b908b39b36ce9addfb67cac0b56ede4e9c6382555458'


if __name__=='__main__':
    truststore.inject_into_ssl();data=urllib.request.urlopen(URL,timeout=120).read()
    if hashlib.sha256(data).hexdigest()!=SHA256:raise ValueError('Source checksum mismatch')
    Path('data').mkdir(exist_ok=True);Path('data/assays.tsv').write_bytes(data);print('Verified upstream literature-extracted assay table')
