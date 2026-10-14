# ChEMBL snapshot attribution

ChEMBL data is from https://www.ebi.ac.uk/chembl — the version is **ChEMBL_37**, API release date **2026-05-01**. [ChEMBL licensing](https://chembl.github.io/chembl-licensing/): [Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/).

`chembl-herg-v37.jsonl.gz` preserves 17,099 API activity records for target CHEMBL240, standard IC50 in nM, including relations, upper values, document and assay identifiers. `results/chembl-herg/source-manifest.json` records the page URLs, checksums, target and API release. The compressed snapshot is deterministic (gzip modification time 0); it is data, not covered by the repository's MIT code license. Source-derived activity/split/prediction tables retain ChEMBL attribution and CC BY-SA terms.

Processing changes are documented in `chembl_benchmark.py`: exclude validity/duplicate flags, invalid or missing observations, missing documents, invalid structures, and repeated molecule/assay/bounds; standardize nM to µM without changing inequalities; canonicalize molecules; use document holdouts with molecule purge. All excluded activity IDs and reasons are released. This selects an assay population; it does not create newly measured biology.
