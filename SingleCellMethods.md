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



## Single Cell RNA-seq




## Single Cell ATAC-seq

[ArchR - Granja et al. Nature Genetics 2021](https://pubmed.ncbi.nlm.nih.gov/33633365/) scATAC-seq processing method, and integrates scRNA-seq data. Supports: 1) Doublet detection by first synthesizing artificial dounlets and then using their nearest neighbors as estimated doublets, 2) Optimized iterative LSI for dimension reduction by applying LSI on most variable features, 3) Gene scores using ATAC-seq and TSS information to predict dummy of gene expression.

[Signac - Stuart et al. Nature Methods 2021](https://pubmed.ncbi.nlm.nih.gov/34725479/) scATAC-seq processing method, and integrates scRNA-seq data. Supports: 1) Peak calling from individual samples and then merging (to retain cell type specific peaks) and shows that it retains all cellranger peaks. 2) Dimension reduction using LSI. The TF-IDF matrix is computed using the total counts of a cell, total counts for a peak in a cell, total number of cells, and total number of counts for a given peak across all cells. The TF-IDF matrix (after log transformation) are applied to SVD. 3) Integration with scRNA-seq data is done by FindTransferAnchors function in Seurat. 4) Computes gene activity score and performs peak to gene linkage.

[chromVAR - Schep et al. Nat Meth 2017](https://pubmed.ncbi.nlm.nih.gov/28825706/) Using scATAC-seq data, measures the gain / loss of chromatin accessibility within peaks sharing the same motif or annotation.




## Cell Annotation








## Spatial Transcriptomics








## Single cell eQTL, ASE 

[g-ChromVAR, Ulirsch et al. Nat Genet 2021](https://pubmed.ncbi.nlm.nih.gov/30858613/) presents g-chromVAR, method to identify GWAS variant enrichment among closely related tissues / cell types, using scATAC-seq data. It computes the bias-corrected Z-scores to estimate trait-relevance for each single cell by integrating the probability of variant causality and quantitative strength of chromatin accessibility.

[scAVENGE, Yu et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/35668323/) Uses network propagation on causal variants to identify their relevant cell types using single cell resolution. Using the g-chromVAR output, i.e. single cell based trait-relevance scores, we rank the cells and select the top cells as seed cells, which are used for network propagation algorithm. Using random walk algorithm, we reach the stationary state of network connectivity among these cells, and the final trait relevance scores (TRS) are computed for each cell.








