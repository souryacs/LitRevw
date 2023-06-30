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

[TotalVI - Gayoso et al. Nat Meth 2021](https://pubmed.ncbi.nlm.nih.gov/33589839/) scVI on multi-omic setting (CITE-seq), its joint low dimensional representations and the parameters are inferred by VAE framework.

## Single Cell RNA-seq

[LIGER - Welch et al. Cell 2019](https://pubmed.ncbi.nlm.nih.gov/31178122/) Performs integrative nonnegative matrix factorization (INMF) for single-cell RNA-seq data integration.

[MEFISTO - Velten et al. Nat Meth 2022](https://pubmed.ncbi.nlm.nih.gov/35027765/) Uses factor analysis and extends the method MOFA to account for spatio-temporal variation of scRNA-seq data.

[scVI - Lopez et al. Nat Meth 2018](https://pubmed.ncbi.nlm.nih.gov/30504886/) Models scRNA-seq by ZINB distribution, but uses NN to infer its parameters. Performs batch correction. 

[scissor - Sun et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/34764492/) Uses a network regression model to identify the cell populations/clusters associated with a given phenotype. Input: scRNA-seq matrix, bulk RNA-seq matrix, and phenotype matrix/vector (can be binary, continuous, based on that the regression model would be defined). The correlation between single cell expression and bulk RNA-seq gene expression data are computed to produce a correlation matrix which is then applied to a regression framework with respect to the given phenotype.


## Single Cell ATAC-seq

[ArchR - Granja et al. Nature Genetics 2021](https://pubmed.ncbi.nlm.nih.gov/33633365/) scATAC-seq processing method, and integrates scRNA-seq data. Supports: 1) Doublet detection by first synthesizing artificial dounlets and then using their nearest neighbors as estimated doublets, 2) Optimized iterative LSI for dimension reduction by applying LSI on most variable features, 3) Gene scores using ATAC-seq and TSS information to predict dummy of gene expression.

[Signac - Stuart et al. Nature Methods 2021](https://pubmed.ncbi.nlm.nih.gov/34725479/) scATAC-seq processing method, and integrates scRNA-seq data. Supports: 1) Peak calling from individual samples and then merging (to retain cell type specific peaks) and shows that it retains all cellranger peaks. 2) Dimension reduction using LSI. The TF-IDF matrix is computed using the total counts of a cell, total counts for a peak in a cell, total number of cells, and total number of counts for a given peak across all cells. The TF-IDF matrix (after log transformation) are applied to SVD. 3) Integration with scRNA-seq data is done by FindTransferAnchors function in Seurat. 4) Computes gene activity score and performs peak to gene linkage.

[chromVAR - Schep et al. Nat Meth 2017](https://pubmed.ncbi.nlm.nih.gov/28825706/) Using scATAC-seq data, measures the gain / loss of chromatin accessibility within peaks sharing the same motif or annotation.

[CICERO - Pliner et al. Mol Cell 2018](https://pubmed.ncbi.nlm.nih.gov/30078726/) Concept of co-accessibility among peaks.


## Cell Annotation

[CellTypist - Conde et al. Science 2022](https://pubmed.ncbi.nlm.nih.gov/35549406/) Cell annotation using SGD + logistic regression. Applied on immune cell types. Supports both low and high resolution cell annotation, but may require manual curation of datasets.

[scNym - Kimmel et al. Genome Research 2021](https://pubmed.ncbi.nlm.nih.gov/33627475/) Cell annotation using domain adversarial neural network. Uses both training data (with labels) and target data (to learn the embeddings). Uses mixmatch scheme to permute the input data and labels and the domain adversarial network predicts the domain of origin (training / test data). Classifier is updated by the inverse of adversarial gradients.

[scANVI - Xu et al. Molecular System Biology 2021](https://pubmed.ncbi.nlm.nih.gov/33491336/) Cell annotation on top of scVI framework. Uses harmonization (similar to batch effect correction, but extended to supports datasets even from multiple technologies) and automatic cell annotation. Uses probabilistic cell annotation (generative model) in 2 steps: 1) First annotates a subset of cells with high confidence, 2) Then annotates the remaining cells using the annotations of the previous set of cells.


## Spatial Transcriptomics








## Single cell eQTL, ASE, variant annotation

[g-ChromVAR - Ulirsch et al. Nat Genet 2021](https://pubmed.ncbi.nlm.nih.gov/30858613/) presents g-chromVAR, a method to identify GWAS variant enrichment among closely related tissues/cell types, using scATAC-seq data. The objective is to measure the trait relevance of different cells or tissues, and here scATAC-seq data together with fine-mapped GWAS variants are used to measure such trait relevance scores. It computes the bias-corrected Z-scores to estimate the trait relevance for every single cell by integrating the probability of variant causality and quantitative strength of chromatin accessibility. The trait-peak matrix is a count matrix, which is used to compute the expected number of fragments per peak per sample, which is then multiplied with the fine-mapped variant posterior probabilities. Validated with respect to S-LDSC method.

[scAVENGE - Yu et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/35668323/) Uses network propagation on causal variants to identify their relevant cell types using single cell resolution. Using the g-chromVAR output, i.e. single cell based trait-relevance scores, we rank the cells and select the top cells as seed cells, which are used for network propagation algorithm. Using random walk algorithm, we reach the stationary state of network connectivity among these cells, and the final trait relevance scores (TRS) are computed for each cell.

[scLinker - Jagadeesh et al. Nat Genet 2022](https://pubmed.ncbi.nlm.nih.gov/36175791/) Integrates GWAS summary statistics, epigenomics, and scRNA-seq data from multiple tissue types, diseases, individuals and cells. The authors transform gene programs to SNP annotations using tissue-specific enhancer–gene links, standard gene window-based linking strategies such as MAGMA, RSS-E and linkage disequilibrium score regression (LDSC)-specifically expressed genes. Then they link SNP annotations to diseases by applying stratified LDSC (S-LDSC) to the resulting SNP annotations. 

[scDRS - Zhang et al. Nat Genet 2022](https://pubmed.ncbi.nlm.nih.gov/36050550/) scDRS method. Enrichment of cell types with respect to GWAS trait. First identifies the putative gene sets for individual GWAS traits or diseases using MAGMA. Then identifies the cell type and single-cell level correlation between the gene set and cells, and computes the GWAS enrichment of a cell type.

[NumBat - Gao et al. Nat Biotech 2023](https://pubmed.ncbi.nlm.nih.gov/36163550/) Haplotype aware CNV inference from scRNA-seq data. CNVs are inferred both from expression (expecting AMP and DEL to be associated with up/downregulation - FP for expression changes unrelated to CNV) and allele-specific (deviations of BAF - less affected by sample-specific variation). This method uses haplotype phasing prior to detecting CNVs.

