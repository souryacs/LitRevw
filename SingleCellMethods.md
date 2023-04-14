# Computational approaches for Single Cell Data

## Multimodal Data Integration / pipelines

**Seurat - Different versions**

  &emsp; [Satija et al. Nat Biotech 2015](https://pubmed.ncbi.nlm.nih.gov/25867923/) First paper on Seurat. Talks about utilizing spatial and scRNA-seq datasets.
  
  &emsp; [Butler et al. Nat Biotech 2018](https://pubmed.ncbi.nlm.nih.gov/29608179/) Proposes CCA, specially Diagonal CCA to integrate multiple scRNA-seq datasets.
  
  &emsp; [Stuart et al. Cell 2019](https://pubmed.ncbi.nlm.nih.gov/31178118/) Proposes scTransform + VST + IntegrateAnchors and IntegrateFeatures, to integrate scRNA-seq, scATAC-seq or CITE-seq datasets.
  
  &emsp; [Stuart et al. Nat Revw Genet 2019](https://pubmed.ncbi.nlm.nih.gov/30696980/) Review paper on Seurat.
  
  &emsp; [Hao et al. Cell 2021](https://pubmed.ncbi.nlm.nih.gov/34062119/) Proposes WNN for multimodal data integration.
  
  &emsp; [Hao et al. bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.02.24.481684v1) Dictionary learning for multimodal data integration.

[VIMCCA - Wang et al. Bioinformatics 2023](https://pubmed.ncbi.nlm.nih.gov/36622018/) Variational inference method - generalizing CCA. Multi-view latent variable. CCA is modeled by multi-view latent variable and variational distribution. Maximizing log likelihood is modeled as maximizing evidence lower bound (ELBO). It has 2 components - KL divergence, and reconstruction error. SGVB estimator using monte carlo simulator is used to estimate the ELBO.




## Cell Annotation








## Spatial Transcriptomics









