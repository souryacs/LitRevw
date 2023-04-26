# SNP / QTL / GWAS and functional variants


## Genotyping

[Meta Imputation - Yu et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35508176/) - Integrates multiple genotype imputation output. Uses weighted average of meta analysis.

## Colocalization / Fine-mapping

[SusieR - Zou et al. PLoS Genetics 2022](https://pubmed.ncbi.nlm.nih.gov/35853082/) Implements the sum of single effects ([SUSiE](https://academic.oup.com/jrsssb/article/82/5/1273/7056114)) model for fine mapping, and describes the differences of fine-mapping techniques when individual lelevl genotype data are available, and when only the summary statistics are available (more common - SusieRSS approach).


## QTL / SNP enrichment analysis

[Torres et al. AJHG 2021](https://pubmed.ncbi.nlm.nih.gov/33186544/) Tissue-of-action (TOA) scores of T2D GWAS, using fine-mapped variants, independent fine-mapped GWAS loci, reference coding annotations. A weighted sum of annotations for all fine-mapped SNPs are used for tissue-specific enrichment computation.






## QTL - Polygenic Risk Scores (PRS)

[Smail et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35588732/) utilizes rare variants and outlier genes (having Z score expression > certain threshold) to characterize the phenotypic effect of these rare variants.

[TL-PRS: Zhao et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/36240765/) Constructing cross-population polgenic risk scores using transfer learning.

[VIPRS - Zabad et al. AJHG 2023](https://pubmed.ncbi.nlm.nih.gov/37030289/) Approximates Bayesian computation of PRS by replacing MCMC with variational inference (VI), a deterministic class of algorithms replacing the posterior inference by an optimization problem (applied to LMM, Fine-mapping, association, enrichment).


## QTL - TWAS

[METRO - Li et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35334221/) TWAS using multi-ancestry (population) data - different allele frequencies and LD matrix.

[INTACT: Okamoto et al. AJHG 2023](https://pubmed.ncbi.nlm.nih.gov/36608684/) - Integration of TWAS and colocalization to identify causal genes. Posterior probability of a gene being causal : prior probability of colocalization * bayes factor (BF) from TWAS * prior of TWAS.


## Identifying disease-risk / causal variants (and) target genes

[3DFAACTS-SNP: Liu et al. Epigenomics and Chromatin 2022](https://epigeneticsandchromatin.biomedcentral.com/articles/10.1186/s13072-022-00456-5) proposed T1D causal variant identification pipeline: 1) Bayesian fine mapping, 2) Overlap with ATAC-seq peaks (open chromatin region), 3) HiC ineractions, 4) FOXP3 binding sites.

[MA-FOCUS - Lu et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35931050/) - Fine mapping and TWAS using multiple ancestry information. Integrates GWAS, eQTL and LD information, and assumes that causal genes are shared across ancestry.

[PALM - Yu et al. Bioinformatics 2023](https://pubmed.ncbi.nlm.nih.gov/36744920/) Uses functional annotations to prioritize GWAS variants.
Supports multiple functional annotations. Computes gradient boosting and tree based likelihood to prioritize the GWAS SNPs.

[MOVE - Allesoe et al. Nat Biotech 2023](https://pubmed.ncbi.nlm.nih.gov/36593394/) Defines MOVE - Multi-omics variational autoencoder including data from multiple omics from 789 sample cohort (vertical integration) and applies VAE, and defines the association between T2D with the latent space features. Significance is computed by t-test, and by feature purturbation (0/1) technique.



