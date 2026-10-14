# Extension protocol: observation semantics versus prediction

Fixed before fitting extension models on 2026-10-14. Earlier peptide results have already been seen. This is a versioned analysis plan, not an independently registered preregistration.

Question: does preserving exact/one-sided/finite-interval observation semantics improve prediction under source shift, and when does the result depend on the assay population and training objective?

1. Implement finite-interval Gaussian likelihood using stable log-CDF/log-survival differences. Validate against independent numerical integration, extreme tails, shrinking intervals and gradients. Preserve the original exact-density semantics.
2. Fetch all ChEMBL CHEMBL240 hERG IC50 records in standard nM units, retaining original relations, upper values, document and assay IDs, release version and raw checksums. Do not use pChEMBL values to reconstruct censoring. Exclude missing/invalid/nonpositive/unsupported observations and record exclusions; do not choose rows by predictive performance.
3. This is an independent database and assay domain, NOT an independent peptide MIC replication. Do not combine its outcomes with the peptide table or claim peptide generalization from hERG.
4. Compare matched endpoint-Gaussian and observation-likelihood heads with identical fingerprint features and architecture; include a train-only constant Gaussian. Fit train-only feature preprocessing. Use document-held-out splits and remove any test/validation molecule from training. Run split seeds 901/902/903 with model seed 417, validation-only stopping; record all counts and predictions.
5. Primary metric: held-out observation NLL, with exact/left/right/interval counts and metrics separated. Exact-only RMSE and bound compatibility are secondary, never a surrogate for unknown values. Report per-document differences and split variability; no superiority claim if mixed/uncertain.
6. Keep peptide first-release results intact. Generate deterministic replays and retain the independent source/data license (ChEMBL CC BY-SA 3.0) separately from code MIT.

Outstanding independent validation: review extraction and source assumptions with another researcher; add a second peptide-specific assay source; neither is satisfied by adding an IC50 table.
