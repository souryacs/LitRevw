# Computational approaches for Single Cell Data

Check [This GitHub page](https://github.com/OmicsML/awesome-deep-learning-single-cell-papers) for a comprehensive summary of deep learning and single-cell papers.

Check [This eBook from Fabian Theis group](https://www.sc-best-practices.org/preamble.html) about the best practices in single cell data analysis.


## Multimodal Data Integration/pipelines

[Spatial reconstruction of single-cell gene expression data - Satija et al. Nat Biotech 2015](https://pubmed.ncbi.nlm.nih.gov/25867923/) First paper on Seurat. Talks about utilizing spatial and scRNA-seq datasets.
  
[Integrating single-cell transcriptomic data across different conditions, technologies, and species - Butler et al. Nat Biotech 2018](https://pubmed.ncbi.nlm.nih.gov/29608179/) Second Seurat paper. Proposes CCA, specially Diagonal CCA to integrate multiple scRNA-seq datasets. The diagonal (or regularized/penalized) CCA is useful since the number of genes (integrating features) is much lower than the number of cells. Aligning two CCA vectors (also called metagenes) is done by dynamic time warping (DTW) algorithm. Partial SVD implementation is used to identify a set of user-defined CCA vectors. Compares CCA with PCA to show that CCA retrieves group of features shared between different datasets. Also compares the integration to the conventional batch correction methods Combat and Limma.
  
[Comprehensive Integration of Single-Cell Data - Stuart et al. Cell 2019](https://pubmed.ncbi.nlm.nih.gov/31178118/) Third Seurat paper. Proposes scTransform + VST + IntegrateAnchors and IntegrateFeatures, to integrate scRNA-seq, scATAC-seq, or CITE-seq datasets. The VST is used to first estimate the variance from means of individual gene expression, using linear regression, and then standardize the expression by mean and variance normalization. Implement diagonal CCA implementation to maximize the sharing of features among both datasets. MNN concept is used after diagonal CCA and such neighbors are termed anchors. An anchor scoring mechanism followd by anchor weighting using the nearest anchor cells in the query dataset is employed using the shared nearest neighbor (SNN) concept to finally use the highest scoring anchors as integration features (implemented in the function IntegrateData()).
  
[Integrative single-cell analysis - Stuart et al. Nat Revw Genet 2019](https://pubmed.ncbi.nlm.nih.gov/30696980/) Review paper on Seurat.
  
[Integrated analysis of multimodal single-cell data - Hao et al. Cell 2021](https://pubmed.ncbi.nlm.nih.gov/34062119/) Fourth Seurat paper. Proposes WNN for multimodal data integration. Applied on CITE-seq data, and integrated scRNA-seq + scATAC-seq multimodal data. 1) Constructs independent KNN graph on both modalities, 2) Perform within and across-modality prediction, 3) Cell-specific modality weights and similarity between the observed and the predicted RNA and protein profile, using exponential distribution (approach large margin nearest neighbors), 4) WNN graph construction. scRNA-seq data is processed by Seurat, protein data is normalized by centered log-ratio (CLR) transform (all proteins are used as features without any feature selection). scATAC-seq data is processed according to the Signac package, by applying TF-IDF + log transformation on the peak matrix, and then applying SVD, which returns the final LSI (latent semantic indexing) components. Within-modality prediction means predicting cell profile from the neighbors using the same modality, while cross-modality prediction indicates predicting cell profile from the neighbors using the other modality information.
  
[Dictionary learning for integrative, multimodal, and scalable single-cell analysis - Hao et al. bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.02.24.481684v1) Fifth Seurat paper. Dictionary learning for multimodal data integration. Bridge integration to integrate multiple modalities, like integrating scATAC-seq on the reference cell annotations defined by scRNA-seq data. Then discusses dictionary learning and atomic sketching, inspired by the geometric sketching method from image processing, to select a subset of features from the datasets, integrate and then project back the integrated results on the full set of features. The final alignment between different modalities is implemented by the mnnCorrect algorithm. The computational complexity for handling many cells is reduced by the Laplacian Eigenmaps mechanism (graph eigendecomposition) thereby reducing the number of dimensions from the number of cells to the number of eigenvectors.

[A multi-view latent variable model reveals cellular heterogeneity in complex tissues for paired multimodal single-cell data - VIMCCA - Wang et al. Bioinformatics 2023](https://pubmed.ncbi.nlm.nih.gov/36622018/) Variational inference method - generalizing CCA. Multi-view latent variable. CCA is modeled by multi-view latent variable and variational distribution. Maximizing log-likelihood is modeled as maximizing evidence lower bound (ELBO). It has 2 components - KL divergence, and reconstruction error. SGVB estimator using the monte carlo simulator is used to estimate the ELBO.

[Joint probabilistic modeling of single-cell multi-omic data with totalVI - Gayoso et al. Nat Meth 2021](https://pubmed.ncbi.nlm.nih.gov/33589839/) scVI on multi-omic setting (CITE-seq). Probabilistic latent variable model. Joint low dimensional representations and the parameters are inferred by VAE framework. Compared against factor analysis (FA), single cell hierarchical poisson factorization (scHPF) and scVI. To check the model fitting, they used posterior predictive check (PPC) by simulating replicated datasets, and comparing the statistical significance between the coefficients of variation (CV) per gene and protein. To benchmark the single cell integration, they propose 4 different metrics, and compare against Seurat, Harmony, Scanorama.

[UINMF performs mosaic integration of single-cell multi-omic datasets using nonnegative matrix factorization - Kriebel et al. Nat Comm 2022](https://pubmed.ncbi.nlm.nih.gov/35140223/) LIGER v2. UINMF using both shared and unshared features to integrate multiple multi-omic datasets. Can integrate datasets with neither the same number of features (genes / peaks / bins) nor the same number of cells.

[Review paper on data integration - Luecken et al. Nat Meth 2022](https://pubmed.ncbi.nlm.nih.gov/34949812/) compares various multimodal data integration approaches. Conclusions: 1) Scanorama and scVI perform well, particularly on complex integration tasks. If cell annotations are available, scGen and scANVI outperform most other methods across tasks, and Harmony and LIGER are effective for scATAC-seq data integration on window and peak feature spaces. 2) In more complex integration tasks, there is a tradeoff between batch effect removal and bio-conservation. While methods such as SAUCIE, LIGER, BBKNN, and Seurat v3 tend to favor the removal of batch effects over the conservation of biological variation, DESC, and Conos make the opposite choice, and Scanorama, scVI, and FastMNN (gene) balance these two objectives.

[Review paper - Argelaguet et al. Nat Biotech 2021](https://pubmed.ncbi.nlm.nih.gov/33941931/) discusses scRNA-seq data integration - horizontal (gene-based), vertical (cell-based), and diagonal.

[MARIO - Zhu et al. Nat Meth 2023](https://pubmed.ncbi.nlm.nih.gov/36624212/) Appliable for protein-centric multimodal datasets such as CITE-seq, CyTOF, etc. First applies SVD + CCA on individual datasets to find the cell-cell pairing (matching cells). CCA is performed using both shared and unshared features. Then a regularized K-means clustering is performed for the final integration.

[GLUE - Cao et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/35501393/) Integrating multiple omics datasets using graph variational autoencoders and also by using the regulatory interactions between the omics as a prior guided graph. For example, integration between scRNA-seq and scATAC-seq data requires prior edge formation using the peak-to-gene correlation.

## Single Cell RNA-seq

## Integration / modeling

[Single-Cell Multi-omic Integration Compares and Contrasts Features of Brain Cell Identity - LIGER - Welch et al. Cell 2019](https://pubmed.ncbi.nlm.nih.gov/31178122/) Performs integrative nonnegative matrix factorization (INMF) for single-cell RNA-seq data integration.

[Jointly defining cell types from multiple single-cell datasets using LIGER - Liu et al. Nat Protocol 2020](https://pubmed.ncbi.nlm.nih.gov/33046898/) LIGER paper - running protocol.

[MEFISTO - Velten et al. Nat Meth 2022](https://pubmed.ncbi.nlm.nih.gov/35027765/) Uses factor analysis and extends the method MOFA to account for spatiotemporal variation of scRNA-seq data.

[scVI - Lopez et al. Nat Meth 2018](https://pubmed.ncbi.nlm.nih.gov/30504886/) Models scRNA-seq by ZINB distribution, but uses NN to infer its parameters. Performs batch correction. 

[scFormer: A Universal Representation Learning Approach for Single-Cell Data Using Transformers - Cui et al. bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.11.20.517285v1) Transformer based modeling of scRNA-seq data, gene expression, optimizing cell and gene embeddings in unsupervised manner.

## Batch correction

[HARMONY - Korsunski et al. Nat Meth 2019](https://pubmed.ncbi.nlm.nih.gov/31740819/) Batch correction method. First performs modified K-means soft clustering to assign cells to potential candidate clusters (1 cell is assigned to multiple clusters). Then define batch-specific parameters are used to compute the penalty of cluster assignments. Finally, a weighted sum of these clustering assignments are performed to define the final clusters.

## Cell annotation

[scissor - Sun et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/34764492/) Uses a network regression model to identify the cell populations/clusters associated with a given phenotype. Input: scRNA-seq matrix, bulk RNA-seq matrix, and phenotype matrix/vector (can be binary, continuous, based on that the regression model would be defined). The correlation between single-cell expression and bulk RNA-seq gene expression data is computed to produce a correlation matrix which is then applied to a regression framework with respect to the given phenotype.

[scBERT - scBERT as a large-scale pre-trained deep language model for cell type annotation of single-cell RNA-seq data - Fan Yang et al. Nature Machine Intelligence 2022](https://www.nature.com/articles/s42256-022-00534-z) Applies BERT together with performer (a modified transformer encoder model with higher receptive field) to annotate scRNA-seq cells.

[scPoli - Population-level integration of single-cell datasets enables multi-scale analysis across samples - Donno et al. bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.11.28.517803v1) Multiple scRNA-seq data integration using generative AI, specifically a modification of CVAE method. It integrates multiple samples and simultaneously annotates the cells, similar to Seurat and scANVI. Implements this framework inside scArches. Performs both reference building and reference mapping.


## Single Cell ATAC-seq

[ArchR - Granja et al. Nature Genetics 2021](https://pubmed.ncbi.nlm.nih.gov/33633365/) scATAC-seq processing method, and integrates scRNA-seq data. Supports: 1) Doublet detection by first synthesizing artificial doublets and then using their nearest neighbors as estimated doublets (similar to Scrublet), 2) Optimized iterative LSI for dimension reduction by applying LSI on most variable features and a selective subset of cells and then projecting the results on the complete set of cells, 3) Gene scores using ATAC-seq and TSS information to predict dummy of gene expression. 4) Also implements both Slingshot and Monocle3 for trajectory inference.

[Signac - Stuart et al. Nature Methods 2021](https://pubmed.ncbi.nlm.nih.gov/34725479/) scATAC-seq processing method, and integrates scRNA-seq data. Supports: 1) Peak calling from individual samples and then merging (to retain cell type-specific peaks) and showing that it retains all cell ranger peaks. 2) Dimension reduction using LSI. The TF-IDF matrix is computed using the total counts of a cell, total counts for a peak in a cell, the total number of cells, and the total number of counts for a given peak across all cells. The TF-IDF matrix (after log transformation) is applied to SVD. 3) Integration with scRNA-seq data is done by the FindTransferAnchors function in Seurat. 4) Computes gene activity score and performs peak-to-gene linkage.

[chromVAR - Schep et al. Nat Meth 2017](https://pubmed.ncbi.nlm.nih.gov/28825706/) Using scATAC-seq data, measures the gain/loss of chromatin accessibility within peaks sharing the same motif or annotation. Models the expected number of fragments per peak containing a particular motif and for a particular cell. Thus, variation of chromatin accessibility across cells between highly similar k-mers can be computed.

[CICERO - Pliner et al. Mol Cell 2018](https://pubmed.ncbi.nlm.nih.gov/30078726/) Concept of co-accessibility among peaks.

[Episcanpy - Danese et al. Nat Comm 2021](https://pubmed.ncbi.nlm.nih.gov/34471111/) Episcanpy processes both scATAC-seq and sc DNA methylation data, and performs cell-level clustering. Based on the Scanpy framework.

## Cell Annotation

[CellTypist - Conde et al. Science 2022](https://pubmed.ncbi.nlm.nih.gov/35549406/) Cell annotation using SGD + logistic regression. Applied on immune cell types. Supports both low and high-resolution cell annotation, but may require manual curation of datasets.

[scNym - Kimmel et al. Genome Research 2021](https://pubmed.ncbi.nlm.nih.gov/33627475/) Cell annotation using domain adversarial neural network. Uses both training data (with labels) and target data (to learn the embeddings). Uses a mix-match scheme to permute the input data and labels and the domain adversarial network predicts the domain of origin (training/test data). Classifier is updated by the inverse of adversarial gradients.

[scANVI - Xu et al. Molecular System Biology 2021](https://pubmed.ncbi.nlm.nih.gov/33491336/) Cell annotation on top of scVI framework. Uses harmonization (similar to batch effect correction, but extended to support datasets even from multiple technologies) and automatic cell annotation. Uses probabilistic cell annotation (generative model) in 2 steps: 1) First annotates a subset of cells with high confidence, 2) Then annotates the remaining cells using the annotations of the previous set of cells.


## Spatial Transcriptomics

[Review paper on integration between ST and scRNA-seq - Li et al. Nat Meth 2022](https://pubmed.ncbi.nlm.nih.gov/35577954/): Considers performance metrics: Pearson correlation coefficients (PCC), structural similarity index (SSIM), RMSE, Jensen-Shannon divergence (JS), accuracy score (AS), robustness score (RS). 1) Tangram and gimVI outperformed the other integration methods on the basis of these metrics. 2) Considering sparse datasets, Tangram, gimVI, and SpaGE outperformed other integration methods in predicting the spatial distribution of transcripts for highly sparse datasets. 3) In predicting cell type composition of spots, Cell2location, SpatialDWLS, RCTD, and STRIDE outperformed the other integration methods. 4) In terms of computational efficiency, Tangram and Seurat are the top two most-efficient methods for processing cell-type deconvolution of spots.

[HMRF - Zhu et al. Nat Biotech 2018](https://pubmed.ncbi.nlm.nih.gov/30371680/) First paper using spatial information for ST data clustering. After HVG selection and feature reduction, uses HMRF for spatial data. KNN is used for initial clustering using gene expression. Then spatial prior for HMRF is initialized by Potts model. The multinomial distribution is employed to decide the membership of individual cells/spots in a cluster, and the cluster refinement is done by the EM algorithm. Implemented in the Giotto toolbox framework [Dries et al. Genome Biology 2021](https://pubmed.ncbi.nlm.nih.gov/33685491/).

[BayesSpace - Zhao et al. Nat Biotech 2021](https://pubmed.ncbi.nlm.nih.gov/34083791/) Implements BayesSpace to model ST data. Minor adjustments of HMRF by implementing MCMC instead of EM algorithm in the spatial refinement. Also, employs a fixed precision matrix (similar across individual clusters for less parameter estimation).




## Single cell eQTL, ASE, variant annotation

[g-ChromVAR - Ulirsch et al. Nat Genet 2021](https://pubmed.ncbi.nlm.nih.gov/30858613/) presents g-chromVAR, a method to identify GWAS variant enrichment among closely related tissues/cell types, using scATAC-seq data. The objective is to measure the trait relevance of different cells or tissues, and here scATAC-seq data together with fine-mapped GWAS variants are used to measure such trait relevance scores. It computes the bias-corrected Z-scores to estimate the trait relevance for every single cell by integrating the probability of variant causality and quantitative strength of chromatin accessibility. The trait-peak matrix is a count matrix, which is used to compute the expected number of fragments per peak per sample, which is then multiplied with the fine-mapped variant posterior probabilities. Validated with respect to the S-LDSC method.

[scAVENGE - Yu et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/35668323/) Uses network propagation on causal variants to identify their relevant cell types using single-cell resolution. Using the g-chromVAR output, i.e. single cell-based trait-relevance scores, we rank the cells and select the top cells as seed cells, which are used for the network propagation algorithm. Using the random walk algorithm, we reach the stationary state of network connectivity among these cells, and the final trait relevance scores (TRS) are computed for each cell.

[scLinker - Jagadeesh et al. Nat Genet 2022](https://pubmed.ncbi.nlm.nih.gov/36175791/) Integrates GWAS summary statistics, epigenomics, and scRNA-seq data from multiple tissue types, diseases, individuals, and cells. The authors transform gene programs to SNP annotations using tissue-specific enhancer–gene links, standard gene window-based linking strategies such as MAGMA, RSS-E, and linkage disequilibrium score regression (LDSC)-specifically expressed genes. Then they link SNP annotations to diseases by applying stratified LDSC (S-LDSC) to the resulting SNP annotations. 

[scDRS - Zhang et al. Nat Genet 2022](https://pubmed.ncbi.nlm.nih.gov/36050550/) scDRS method. Enrichment of cell types with respect to GWAS trait. First identifies the putative gene sets for individual GWAS traits or diseases using MAGMA. Then identifies the cell type and single-cell level correlation between the gene set and cells, and computes the GWAS enrichment of a cell type.

[NumBat - Gao et al. Nat Biotech 2023](https://pubmed.ncbi.nlm.nih.gov/36163550/) Haplotype aware CNV inference from scRNA-seq data. CNVs are inferred both from expression (expecting AMP and DEL to be associated with up/downregulation - FP for expression changes unrelated to CNV) and allele-specific (deviations of BAF - less affected by sample-specific variation). This method uses haplotype phasing prior to detecting CNVs.

[GASPACHO - Kumasaka et al. Nat Genet 2023](https://pubmed.ncbi.nlm.nih.gov/37308670/) Presents GASPACHO, a Gaussian Process + Latent Variable based model to infer the eQTLs associated with dynamic cell states obtained from immune response/stimuli. Test their method on Fibroblast scRNA-seq data and also colocalizes with COVID-19 GWAS data to identify colocalized sc-eQTLs associated with OAS1 and OAS3 genes.

[POPS - Weeks et al. Nat Genet 2023](https://www.nature.com/articles/s41588-023-01443-6) PoPs method for prioritizing gene sets from GWAS summary statistics and LD matrix. Uses MAGMA to first compute the gene-based scores. Also uses pathways, PPIs, etc. to prioritize groups of genes with similar effects/features and uses them to compute the gene-based enrichment statistics, using a multivariate normal (MVN) distribution-based regression strategy. 

## Gene regulatory network (GRN)

[SCENIC - Aibar et al. Nat Meth 2017](https://pubmed.ncbi.nlm.nih.gov/28991892/) GRN from scRNA-seq data. Predicts interactions between TFs and target genes. Coexpression is computed at a very limited distance (~20 Kb) between genes and TFs.

[SCENIC+ - Gonzlez-Blas et al. bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.08.19.504505v1) Extends SCENIC by using  scATAC-seq data to identify the enhancers associated with candidate TFs and identify their correlation with the candidate gene expression. The co-accessibility of peaks helps to identify and examine the TFs related to selective peaks up to 150 Kb distance from the gene.

[CellOracle - Kamimoto et al. Nature 2023](https://pubmed.ncbi.nlm.nih.gov/36755098/) Constructs GRN from multi-omic data and then simulates the effect of dynamic GRN following TF perturbation. Builds an ML model to predict the effect of TF on GRN. *(To read in detail)*

## Trajectory analysis /  RNA velocity

[Monocle - Trapnell et al. Nat Biotech 2014](https://pubmed.ncbi.nlm.nih.gov/24658644/) Trajectory analysis from scRNA-seq data. ICA is used for dimensionality reduction, after selecting a subset of genes with respect to the variance explained. Then, the minimum spanning tree (MST) is used to construct the lineage. A PQ-Tree-specific algorithm is used to deal with the branching noise.

[RNA Velocity - La Manno et al. Nature 2018](https://pubmed.ncbi.nlm.nih.gov/30089906/) Concept of RNA velocity using the spliced and unspliced RNA. Provides a toolkit Velocyto. Assumes that the transcriptional regulation parameters are the same for all genes, and each gene has a sufficient time frame to reach the steady state.

[scVelo - Bergen et al. Nat Biotech 2020](https://pubmed.ncbi.nlm.nih.gov/32747759/) scVelo method, extending the RNA velocity concept on scRNA-seq data by modeling the transcriptional regulation parameters in a probabilistic model. Also improves the running time.

[multiVelo - Li et al. Nat Biotech 2023](https://pubmed.ncbi.nlm.nih.gov/36229609/) multiVelo approach. Using both scRNA-seq and scATAC-seq data for velocity estimation. Based on the fact that epigenomic changes (like the transition from euchromatin to heterochromatin) have a role in transcriptional regulation and rates. Uses ODE with switch and rate parameters. Inputs: time-varying levels of chromatin accessibility, unspliced pre-RNA, and spliced mature RNA. Parameters: rates of chromatin opening and closing, RNA transcription, RNA splicing, and RNA degradation of nuclear export.

## Disease-specific

[Dohmen et al. Genome Biology 2022](https://pubmed.ncbi.nlm.nih.gov/35637521/) Presents ikarus, an ML framework to identify and annotate tumor cells from normal cells using single-cell data. Identifies a marker gene set signature to identify the set of cells.
