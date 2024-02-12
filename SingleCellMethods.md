# Computational approaches for Single Cell Data

Check [This GitHub page](https://github.com/OmicsML/awesome-deep-learning-single-cell-papers) for a comprehensive summary of deep learning and single-cell papers.

Check [This eBook from Fabian Theis group](https://www.sc-best-practices.org/preamble.html) about the best practices in single cell data analysis.

## Data Simulation

[scDesign3 generates realistic in silico data for multimodal single-cell and spatial omics - Song et al. Nat Biotech 2023](https://www.nature.com/articles/s41587-023-01772-1) Simulates scRNA-seq, scATAC-seq, multi-omic (CITE-seq) data, and spatial transcriptomic data. Benchmarked against data simulators from individual data categories. Extends their earlier works scDesign1 and scDesign2. Also uses their another work scReadSim for simulating single cell reads.



## Multimodal Data Integration/pipelines

[Spatial reconstruction of single-cell gene expression data - Satija et al. Nat Biotech 2015](https://pubmed.ncbi.nlm.nih.gov/25867923/) 
    
    First paper on Seurat. Talks about utilizing spatial and scRNA-seq datasets. 
    
[Integrating single-cell transcriptomic data across different conditions, technologies, and species - Butler et al. Nat Biotech 2018](https://pubmed.ncbi.nlm.nih.gov/29608179/) 
    
    - Second Seurat paper. 
    - Input: Multiple scRNA-seq datasets.
    - Output: Integrated scRNA-seq data + cluster.
    - Method: 
        - HVG selection by dispersion (variance to mean ratio) and selecting top 1000 genes with highest dispersion
        - CCA - projections of two data such that the correlation between these two projections get maximized.
        - As the number of genes are much smaller than the number of cells, to handle the sparsity, they treat the covariance matrix as diagonal (Diagonal CCA). 
        - Aligning two CCA vectors (also called metagenes) is done by dynamic time warping (DTW) algorithm.
        - Canonical correlation vectors are computed using partial SVD - left and right singular vectors with a user-defined number K, to get a subset of the canonical correlation vectors.
        - Compare CCA with PCA to show that CCA retrieves a group of features shared between different datasets.
        - Also compares the integration to the conventional batch correction methods Combat and Limma.
  
[Comprehensive Integration of Single-Cell Data - Stuart et al. Cell 2019](https://pubmed.ncbi.nlm.nih.gov/31178118/) 

    - Third Seurat paper. 
    - Input: multi-omic (CITE-seq or scRNA-seq + scATAC-seq data)
    - Output: Merged object + downstream dimensionality reduction + clusters
    - Method:
        - Proposes scTransform + VST + IntegrateAnchors and IntegrateFeatures, to integrate scRNA-seq, scATAC-seq, or CITE-seq datasets. 
        - VST is used to first estimate the variance from means of individual gene expression, using linear regression, and then standardize the expression by mean and variance normalization. 
        - Implements diagonal CCA to maximize the sharing of features among both datasets. 
        - MNN is used after diagonal CCA and such neighbors are termed anchors. 
        - An anchor scoring mechanism followd by anchor weighting using the nearest anchor cells in the query dataset is employed using the shared nearest neighbor (SNN) concept to finally use the highest scoring anchors as integration features (implemented in the function IntegrateData()).
  
[Integrative single-cell analysis - Stuart et al. Nat Revw Genet 2019](https://pubmed.ncbi.nlm.nih.gov/30696980/) Review paper on Seurat.
  
[Integrated analysis of multimodal single-cell data - Hao et al. Cell 2021](https://pubmed.ncbi.nlm.nih.gov/34062119/) 

    - Fourth Seurat paper. 
    - Input: multi-omic (CITE-seq or scRNA-seq + scATAC-seq data)
    - Output: Merged object + downstream dimensionality reduction + clusters
    - Method:
        - Proposes WNN for multimodal data integration. 
            - Constructs independent KNN graph on both modalities,
            - Perform within and across-modality prediction, 
            - Cell-specific modality weights and similarity between the observed and the predicted RNA and protein profile, using exponential distribution (approach large margin nearest neighbors), 
            - WNN graph construction. 
            - scRNA-seq data is processed by Seurat, protein data is normalized by centered log-ratio (CLR) transform (all proteins are used as features without any feature selection). scATAC-seq data is processed according to the Signac package, by applying TF-IDF + log transformation on the peak matrix, and then applying SVD, which returns the final LSI (latent semantic indexing) components. 
            - Within-modality prediction means predicting cell profile from the neighbors using the same modality, while cross-modality prediction indicates predicting cell profile from the neighbors using the other modality information.
  
[Dictionary learning for integrative, multimodal, and scalable single-cell analysis - Hao et al. bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.02.24.481684v1) 

    - Fifth Seurat paper. 
    - Dictionary learning for multimodal data integration. 
    - Bridge integration to integrate multiple modalities, like integrating scATAC-seq on the reference cell annotations defined by scRNA-seq data. 
    - Then discusses dictionary learning and atomic sketching, inspired by the geometric sketching method from image processing, to select a subset of features from the datasets, integrate and then project back the integrated results on the full set of features. 
    - The final alignment between different modalities is implemented by the mnnCorrect algorithm. 
    - The computational complexity for handling many cells is reduced by the Laplacian Eigenmaps mechanism (graph eigendecomposition) thereby reducing the number of dimensions from the number of cells to the number of eigenvectors.

[A multi-view latent variable model reveals cellular heterogeneity in complex tissues for paired multimodal single-cell data - VIMCCA - Wang et al. Bioinformatics 2023](https://pubmed.ncbi.nlm.nih.gov/36622018/) 

    - VIMCCA method to integrate paired multimodal single cell datasets. 
    - Variational inference method - generalizing CCA. 
    - Multi-view latent variable. 
        - Reasons: 
            - 1) although Seurat integrates multiple data, it does not mention a statistical model thus not account for the underlying sources of variation within each modality. 
            - 2) factor analysis models like MOFA are not scalable for large-scale data. 
        - Objective: 
            - In VIMCCA, CCA is modeled by multi-view latent variable and variational distribution. 
            - Observed data [X,Y] where X = scRNA-seq, Y = scATAC-seq (or ADT) is modeled by a latent factor Z. 
            - Mapping between Z to X and Y are modeled by two non-linear functions whose parameters are estimated by NN. 
            - This non-linear function replaces the conventional CCA. 
        - Implementation: 
            - These non-linear functions are approximated by variational inference (VI) for linearity. 
            - Maximizing log-likelihood is modeled as maximizing evidence lower bound (ELBO). 
                - It has 2 components - KL divergence, and reconstruction error. 
                - SGVB estimator using the monte carlo simulator is used to estimate the ELBO.

[Joint probabilistic modeling of single-cell multi-omic data with totalVI - Gayoso et al. Nat Meth 2021](https://pubmed.ncbi.nlm.nih.gov/33589839/) scVI on multi-omic setting (CITE-seq). Probabilistic latent variable model. Joint low dimensional representations and the parameters are inferred by VAE framework. Compared against factor analysis (FA), single cell hierarchical poisson factorization (scHPF) and scVI. To check the model fitting, they used posterior predictive check (PPC) by simulating replicated datasets, and comparing the statistical significance between the coefficients of variation (CV) per gene and protein. To benchmark the single cell integration, they propose 4 different metrics, and compare against Seurat, Harmony, Scanorama.

[UINMF performs mosaic integration of single-cell multi-omic datasets using nonnegative matrix factorization - Kriebel et al. Nat Comm 2022](https://pubmed.ncbi.nlm.nih.gov/35140223/) 

    - LIGER v2. 
    - UINMF using both shared and unshared features to integrate multiple multi-omic datasets. 
    - Can integrate datasets with neither the same number of features (genes / peaks / bins) nor the same number of cells.
    - Each dataset (Ei) is decomposed into shared metagenes (W), dataset specific metagenes constructed from shared features (Vi), unshared metagenes (Ui) and cell factor loadings (Hi).
    - Adjusts the ANLS (adjusted non-negative least square) method and uses coordinate block descent (CBD) algorithm for solving the UINMF optimization problem. 
        - CBD divides the parameters into blocks and then finds the optimal parameters for one block while fixing the others.

[Benchmarking atlas-level data integration in single-cell genomics - Review paper on data integration - Luecken et al. Nat Meth 2022](https://pubmed.ncbi.nlm.nih.gov/34949812/) 

    - Compares various multimodal data integration approaches. 
    - Conclusions: 
        - Scanorama and scVI perform well, particularly on complex integration tasks. 
        - If cell annotations are available, scGen and scANVI outperform most other methods across tasks, and Harmony and LIGER are effective for scATAC-seq data integration on window and peak feature spaces. 
        - In more complex integration tasks, there is a tradeoff between batch effect removal and bio-conservation. 
        - While methods such as SAUCIE, LIGER, BBKNN, and Seurat v3 tend to favor the removal of batch effects over the conservation of biological variation, DESC, and Conos make the opposite choice, and Scanorama, scVI, and FastMNN (gene) balance these two objectives.

[Computational principles and challenges in single-cell data integration - Review paper - Argelaguet et al. Nat Biotech 2021](https://pubmed.ncbi.nlm.nih.gov/33941931/) 

    - Discusses and categorizes various approaches for scRNA-seq data integration - horizontal (gene-based), vertical (cell-based), and diagonal.

[Robust single-cell matching and multimodal analysis using shared and distinct features - MARIO - Zhu et al. Nat Meth 2023](https://pubmed.ncbi.nlm.nih.gov/36624212/) 

    - **Input**: Single cell proteomic datasets (CITE-seq, CyTOF, etc). 
    - **Output**: Integrated proteomic datasets.
    - *Method*:
        - First get the shared features (genes). Using SVD on the shared features, cell-cell correlations are computed, which produces the initial cross-data distance matrix.
        - Initial clustering is performed using this distance matrix and a convex optimization.
        - Then shared and distinct features are used on these aligned datasets, to project in a common subspace using CCA (incorporating the hidden correlations between exclusive features)
        - Then a regularized K-means clustering is performed for the final integration.

[Multi-omics single-cell data integration and regulatory inference with graph-linked embedding - GLUE - Cao et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/35501393/) Integrating multiple omics datasets using graph variational autoencoders and also by using the regulatory interactions between the omics as a prior guided graph (knowledge graph - vertices: features of different omics layers, edges: regulatory interactions). For example, integration between scRNA-seq and scATAC-seq data requires prior edge formation using the peak-to-gene correlation. First creates low-dimensional cell embeddings for individual modalities using VAE (scVI).  

[CMOT- Cross-Modality Optimal Transport for multimodal inference - Alatkar et al. Genome Biology 2023](https://pubmed.ncbi.nlm.nih.gov/37434182/) Integration of multiple single or multi-omic datasets where individual datasets may not have the same set of cells. *Input*: Multiple multi-omic or single omic single cell datasets. All datasets may not have same set of cells, i.e. there may be partial coverage. *Output*: Integrated multi-omic data. *Method*: Optimal transport based integration. Alignment of multi-omic data to single cell data, for the missing cells.

[Cobolt: integrative analysis of multimodal single-cell sequencing data - Gong et al. Genome Biology 2021](https://pubmed.ncbi.nlm.nih.gov/34963480/) Integrating multi-omics data. Also supports cells with single modality (RNA-seq or ATAC-seq). Uses mutlinomial VAE method. Uses latent dirichlet allocation (LDA) for modeling single modalities instead of the ZINB method.

[MultiVI: deep generative model for the integration of multimodal data - Ashuach et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37386189/) MultiVI - from scVI tools. Integrating scRNA-seq and scATAC-seq data. Uses VAE. Similar principles as Cobolt method.

[SIMBA: single-cell embedding along with features - Chen et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37248389/) SIMBA method. Multi-omic single cell data embedding using graph. However, graph nodes can be both genes and cells, and their mutual relation is determined by: 1) Gene expression, 2) TF motif, 3) k-mer, 4) ATAC peaks etc. The objective is to learn the embedding using all features, and then generate the marker features independent of clustering. Question: How these marker features are computed before embedding and clustering? Need to check in detail.

[scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI - Cui et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.04.30.538439v2) scGPT: Generative pre-trained transformer model on many single cell datasets. The pre-trained model is then fne-tuned foe specific tasks: 1) Clustering, 2) Batch correction, 3) GRN inference, 4) Perturbation response modeling.


## Single Cell RNA-seq

### Clustering

[Significance analysis for clustering with single-cell RNA-sequencing data - Grabski et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37429993/) Analyzes various scRNA-seq clustering methods theoretically in terms of significance analysis. Whether overconfidence in discovering cell types. Model based hypothesis testing approach. Extends their earlier method on significance of hierarchical clustering.

### Integration / modeling

[Single-Cell Multi-omic Integration Compares and Contrasts Features of Brain Cell Identity - LIGER - Welch et al. Cell 2019](https://pubmed.ncbi.nlm.nih.gov/31178122/) Performs integrative nonnegative matrix factorization (INMF) for single-cell RNA-seq data integration.

[Jointly defining cell types from multiple single-cell datasets using LIGER - Liu et al. Nat Protocol 2020](https://pubmed.ncbi.nlm.nih.gov/33046898/) LIGER paper - running protocol.

[Identifying temporal and spatial patterns of variation from multimodal data using MEFISTO - Velten et al. Nat Meth 2022](https://pubmed.ncbi.nlm.nih.gov/35027765/) Uses factor analysis and extends the method MOFA to account for spatiotemporal variation of scRNA-seq data.

[Deep generative modeling for single-cell transcriptomics - scVI - Lopez et al. Nat Meth 2018](https://pubmed.ncbi.nlm.nih.gov/30504886/) Models scRNA-seq observed counts by ZINB distribution, conditioned on the batch and additional Gaussian parameters, but uses NN to infer its parameters. Performs batch correction. 

[scFormer: A Universal Representation Learning Approach for Single-Cell Data Using Transformers - Cui et al. bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.11.20.517285v1) Transformer based modeling of scRNA-seq data, gene expression, optimizing cell and gene embeddings in unsupervised manner. Simultaneously model cell-level and gene-level information. Gene expression is applied a value binning approach after conventional normalization (to create gene tokens and solve scaling issues from different batches / experiments). External tokens are used to store meta information. Complete gene representations are used to create cell representation.

[Decomposing Cell Identity for Transfer Learning across Cellular Measurements, Platforms, Tissues, and Species - scCoGAPS method - Stein O Brien et al. Cell System 2019](https://pubmed.ncbi.nlm.nih.gov/31121116/) Extension of NMF (which requires nonnegative entries in decomposed matrices) to introduce Bayesian NMF - the decomposed matrix elements are either 0 or follow gamma distributions with normal prior, and a global poisson prior. Extends their previously published COGAPS method. The gamma distribution is represented as a sum of exponentials for efficient Gibbs sampling. Suited for sparse scRNA-seq datasets. Also implements ProjectR projection, a transfer learning framework to project data in to latent spaces and transfer annotations.

[xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data - Gong et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.03.24.534055v1) Efficient representation of scRNA-seq large scale training data and gene expression matrix by masking, storing only nonzero entries into separate gene expression and gene level embeddings. Also proposes assymetric encoder decoder architecture based transformer model to represent scRNA-seq data, and compares with the method [scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data - Fang et al. Nature Machine Intelligence](https://www.nature.com/articles/s42256-022-00534-z).

[Biologically informed deep learning to query gene programs in single-cell atlases - Lotfollahi et al. Nature Cell Biology 2023](https://pubmed.ncbi.nlm.nih.gov/36732632/) expiMap method. Biologically interpretable DL method to find the gene programs, using reference single cell atlases. 1. From reference single cell atlase with gene expression and biological conditions, constructs interpretable DL framework to find the gene programs. 2. Then identifies the gene programs responsible / differential between conditions. 3. Constructing gene programs for a collection of data is performed by fine-tuning and architectural surgery, similar to the scArches method.

### Batch correction

[Fast, sensitive and accurate integration of single-cell data with Harmony - Korsunski et al. Nat Meth 2019](https://pubmed.ncbi.nlm.nih.gov/31740819/) Batch correction method. First performs modified K-means soft clustering to assign cells to potential candidate clusters (1 cell is assigned to multiple clusters). Then define batch-specific parameters are used to compute the penalty of cluster assignments. Finally, a weighted sum of these clustering assignments are performed to define the final clusters.

### Differential Analysis

[A new Bayesian factor analysis method improves detection of genes and biological processes affected by perturbations in single-cell CRISPR screening - Zhou et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37770710/) Guided Sparse Factor Analysis (GSFA). Input: 1) scRNA-seq expression matrix (Y), 2) perturbation matrix  (G) - gRNA perturbations per cell. Method: Decomposes Y into latent factors (matrix Z) and derives the weights of genes onto factors (W). Then computes the dependencies of factors onto perturbation matrix via multivariate linear regression. Also performs differential analysis between control and perturbation conditions with respect to these factors to identify DEGs. These DEGs are better powered compared to DE analysis on standard scRNA-seq data. Applied on ASD data.

[Isolating salient variations of interest in single-cell data with contrastiveVI - Weinberger et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37550579/) ContrastiveVI approach. Modeling scRNA-seq data using scVI type approach. The change is, they model shared and unique latent factors specific to control and treatment conditions. Uses control and treatment condition scRNA-seq data and model their differences, using the inferred latent factors.

[Learning single-cell perturbation responses using neural optimal transport - Bunne et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37770709/) CellOT method. *Input*: Separate sets of single cell observations in control and perturbed sets. *Method*: Predict perturbation responses by learning the control and perturbed cell states. Uses optimal transport method. *Schema*:
Models the transition between control population p_c and perturbation population p_k by means of perturbation treatment k by learning the map t_k.

[Causal identification of single-cell experimental perturbation effects with CINEMA-OT - Dong et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37919419/) Another optimal transport based single cell perturbation response prediction method.

### Cell annotation

[Cross-tissue immune cell analysis reveals tissue-specific features in humans - CellTypist - Conde et al. Science 2022](https://pubmed.ncbi.nlm.nih.gov/35549406/) Cell annotation using SGD + logistic regression. Applied on immune cell types. Supports both low and high-resolution cell annotation, but may require manual curation of datasets.

[Semisupervised adversarial neural networks for single-cell classification - scNym - Kimmel et al. Genome Research 2021](https://pubmed.ncbi.nlm.nih.gov/33627475/) Cell annotation using domain adversarial neural network. Uses both training data (with labels) and target data (to learn the embeddings). Uses a mix-match scheme to permute the input data and labels and the domain adversarial network predicts the domain of origin (training/test data). Classifier is updated by the inverse of adversarial gradients.

[scBERT - scBERT as a large-scale pre-trained deep language model for cell type annotation of single-cell RNA-seq data - Fan Yang et al. Nature Machine Intelligence 2022](https://www.nature.com/articles/s42256-022-00534-z) Applies BERT (a transformer model with bidirectional encoder architecture, pre-trained for NLP) together with performer (a matrix decomposition version of transformer model to capture high receptive fields with lower number of feature dimensions) to annotate scRNA-seq cells. scRNA-seq annotation is benchmarked with classic methods like Seurat and ML methods like scNym.

[scPoli - Population-level integration of single-cell datasets enables multi-scale analysis across samples - Donno et al. bioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.11.28.517803v1) Multiple scRNA-seq data integration using generative AI, specifically a modification of CVAE method. It integrates multiple samples and simultaneously annotates the cells, similar to Seurat and scANVI. Implements this framework inside scArches. Performs both reference building and reference mapping. Unlike scArches, the architectural surgery (transfer learning) is performed by first freezing the weights of the trained model and then include the new set of embeddings to accomodate in the query data conditions.

[Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models - scANVI - Xu et al. Molecular System Biology 2021](https://pubmed.ncbi.nlm.nih.gov/33491336/) Cell annotation on top of scVI framework. Uses harmonization (similar to batch effect correction, but extended to support datasets even from multiple technologies) and automatic cell annotation. Uses probabilistic cell annotation (generative model) in 2 steps: 1) First annotates a subset of cells with high confidence, 2) Then annotates the remaining cells using the annotations of the previous set of cells.

[Automatic cell-type harmonization and integration across Human Cell Atlas datasets - Xu et al. Cell 2023](https://pubmed.ncbi.nlm.nih.gov/38134877/) cellHint method. Integrates single cell data and annotation from multiple studies, resolving the cell type labels, using a guided label tree and by hiearchical tree of class labels. In the CellTypist method, the authors did it manually - here they put a automatic workflow.

## Single Cell ATAC-seq

[ArchR is a scalable software package for integrative single-cell chromatin accessibility analysis - Granja et al. Nature Genetics 2021](https://pubmed.ncbi.nlm.nih.gov/33633365/) scATAC-seq processing method, and integrates scRNA-seq data. Supports: 1) Doublet detection by first synthesizing artificial doublets and then using their nearest neighbors as estimated doublets (similar to Scrublet), 2) Optimized iterative LSI for dimension reduction by applying LSI on most variable features and a selective subset of cells and then projecting the results on the complete set of cells, 3) Gene scores using ATAC-seq and TSS information to predict dummy of gene expression. 4) Also implements both Slingshot and Monocle3 for trajectory inference.

[Single-cell chromatin state analysis with Signac - Stuart et al. Nature Methods 2021](https://pubmed.ncbi.nlm.nih.gov/34725479/) scATAC-seq processing method, and integrates scRNA-seq data. Supports: 1) Peak calling from individual samples and then merging (to retain cell type-specific peaks) and showing that it retains all cell ranger peaks. 2) Dimension reduction using LSI. The TF-IDF matrix is computed using the total counts of a cell, total counts for a peak in a cell, the total number of cells, and the total number of counts for a given peak across all cells. The TF-IDF matrix (after log transformation) is applied to SVD. 3) Integration with scRNA-seq data is done by the FindTransferAnchors function in Seurat. 4) Computes gene activity score and performs peak-to-gene linkage.

[chromVaR inferring transcription factor associated accessibility from single cell epigenomic data - Schep et al. Nat Meth 2017](https://pubmed.ncbi.nlm.nih.gov/28825706/) Using scATAC-seq data, measures the gain/loss of chromatin accessibility within peaks sharing the same motif or annotation. Models the expected number of fragments per peak containing a particular motif and for a particular cell. Thus, variation of chromatin accessibility across cells between highly similar k-mers can be computed.

[Cicero Predicts cis-Regulatory DNA Interactions from Single-Cell Chromatin Accessibility Data - Pliner et al. Mol Cell 2018](https://pubmed.ncbi.nlm.nih.gov/30078726/) Concept of co-accessibility among peaks.

[EpiScanpy- integrated single-cell epigenomic analysis - Danese et al. Nat Comm 2021](https://pubmed.ncbi.nlm.nih.gov/34471111/) Episcanpy processes both scATAC-seq and sc DNA methylation data, and performs cell-level clustering. Based on the Scanpy framework.

## Spatial Transcriptomics

### Reviews

[Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type deconvolution - Review paper on integration between ST and scRNA-seq - Li et al. Nat Meth 2022](https://pubmed.ncbi.nlm.nih.gov/35577954/): Considers performance metrics: Pearson correlation coefficients (PCC), structural similarity index (SSIM), RMSE, Jensen-Shannon divergence (JS), accuracy score (AS), robustness score (RS). 1) Tangram and gimVI outperformed the other integration methods on the basis of these metrics. 2) Considering sparse datasets, Tangram, gimVI, and SpaGE outperformed other integration methods in predicting the spatial distribution of transcripts for highly sparse datasets. 3) In predicting cell type composition of spots, Cell2location, SpatialDWLS, RCTD, and STRIDE outperformed the other integration methods. 4) In terms of computational efficiency, Tangram and Seurat are the top two most-efficient methods for processing cell-type deconvolution of spots.

### Alignment between ST and scRNA-seq (or multi-omic data like SHARE-seq)

[Deep learning and alignment of spatially resolved single-cell transcriptomes with Tangram - Biancalani et al. Nat Meth 2021](https://pubmed.ncbi.nlm.nih.gov/34711971/) TANGRAM method for alignment between ST and scRNA-seq / snRNA-seq / multi-omic data, collected from the same origin, and at least having shared marker genes. Identifies the gene expression patterns and the spatial coordinates at cell resolution. Supports various protocols like MERFISH, STARmap, smFISH, Visium, and images. Objective function is to mimic the spatial correlation between each gene in the sc/snRNA-seq data and the spatial data. Cell density is compared by KL divergence, gene expression is assessed by cosine similarity. Assumes that cell segmentation is already done, using tools like ilastik or nucleAIzer. A few hundred marker genes are recommended for alignment.

[Identification of spatial expression trends in single-cell gene expression data - Edsgard et al. Nat Meth 2018](https://pubmed.ncbi.nlm.nih.gov/29553578/) trendsceek method. Identifies genes whose expressions are significantly associated with spatial pattern, using marked point process based modeling. ** TO Do: These genes can be utilized in the above mentioned Tangram method to align the ST data with scRNA-seq datasets.

[DOT- A flexible multi-objective optimization framework for transferring features across single-cell and spatial omics - Rahimi et al. arXiv 2023](https://arxiv.org/abs/2301.01682) DOT method. Integration between scRNA-seq and ST data. Optimal transport based solution to annotate the ST clusters (and spots) with the gene expression values from scRNA-seq data. Proposes multi-objective optimization, including mattching gene expression, matching cell annotation, matching ST neighborhood spot composition, etc.

### Clustering using gene expression and spatial location

[Identification of spatially associated subpopulations by combining scRNAseq and sequential fluorescence in situ hybridization data - HMRF - Zhu et al. Nat Biotech 2018](https://pubmed.ncbi.nlm.nih.gov/30371680/) First paper using spatial information for ST data clustering. After HVG selection and feature reduction, uses HMRF for spatial data. KNN is used for initial clustering using gene expression. Then spatial prior for HMRF is initialized by Potts model. The multinomial distribution is employed to decide the membership of individual cells/spots in a cluster, and the cluster refinement is done by the EM algorithm. Implemented in the Giotto toolbox framework [Dries et al. Genome Biology 2021](https://pubmed.ncbi.nlm.nih.gov/33685491/).

[Spatial transcriptomics at subspot resolution with BayesSpace - Zhao et al. Nat Biotech 2021](https://pubmed.ncbi.nlm.nih.gov/34083791/) Implements BayesSpace to model ST data. Minor adjustments of HMRF by implementing MCMC instead of EM algorithm in the spatial refinement. Also, employs a fixed precision matrix (similar across individual clusters for less parameter estimation). ** To Do: Understand the generative model, compare with scVI, and also understand the metagene identification and compare with the trendsceek and other marker gene identifiers.

### Imaging based methods

[Super-resolved spatial transcriptomics by deep data fusion - Bergenstråhle et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/34845373) Integrates spatial gene expression data with histology image data to predict the gene expression at cellular levels. Implements a deep generative model. Also uses evidence lower bound (ELBO) to define max 40 marker genes (denoted at metagenes).

[Learning consistent subcellular landmarks to quantify changes in multiplexed protein maps - Spitzer et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37248388/) CAMPA method. Profiling ST data using conditional variational autoencoder (cVAE). Uses scanpy and squidpy frameworks. Using the ST data and molecular imaging, performs segmentation at subcellular levels.

[SCS: cell segmentation for high-resolution spatial transcriptomics - Chen et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37429992) SCS: ST data subcellular segmentation using both sequence and imaging data. Uses imaging data for initial segmentation, and then uses transformer framework to refine the foreground and background subcellular information.

### Cell-Cell Communication

[Screening cell-cell communication in spatial transcriptomics via collective optimal transport - Cang et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/36690742) COMMOT method. Cell-Cell communication (Ligand Receptor) using ST data and optimal transport. *Input*: ST matrix - specifically cell-cell (ligand receptor) interaction matrix and spatial information for multiple species. *Method*: Presents an optimal transport based optimization (collective optimal transport) to select the putative Ligand-Receptor pairs. *To Do*: Check the eq 1 and supplementary methods. How these kind of optimization can be used in 1) Fine mapping of SNP? Here distance matrix: LD score.  And SNP-Gene pair specific Summary statistics score. 

## Single cell eQTL, ASE, variant annotation, cell type enrichment

[Single-cell eQTL models reveal dynamic T cell state dependence of disease loci - Nathan et al. Nature 2022](https://pubmed.ncbi.nlm.nih.gov/35545678/) Single cell eQTL inference for T cells, using pseudobulk clustering. Also uses top 20 CCA variables as canonical variates (CVs) and models individual cells' contribution in these 20 CVs to model different cell states. Individual CVs are observed to be correlated with gene regulatory functions. Decomposes for each SNP, the complete genotype-dependent effect (across all cells) as a sum of individual CV-specific genotype effect. Uses poisson mixed effect (PME) model and compares with linear mixed effect (LME) regression to compute cell-state-specific eQTLs. Shows PME is better. Uses genotype and cell specific differential expression as the interaction terms in eQTL inference. *Check*: how the genotype effect is determined for individual cell types or clusters. Can we better model the cell-type or cluster-specific effects, using velocity or trajectory analysis? 

[Identifying phenotype-associated subpopulations by integrating bulk and single-cell sequencing data - scissor - Sun et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/34764492/) Uses a network regression model to identify the cell populations/clusters associated with a given phenotype. Input: scRNA-seq matrix, bulk RNA-seq matrix, and phenotype matrix/vector (can be binary, continuous, based on that the regression model would be defined). The correlation between single-cell expression and bulk RNA-seq gene expression data is computed to produce a correlation matrix which is then applied to a regression framework with respect to the given phenotype.

[Interrogation of human hematopoiesis at single-cell and single-variant resolution - g-ChromVAR - Ulirsch et al. Nat Genet 2019](https://pubmed.ncbi.nlm.nih.gov/30858613/) presents g-chromVAR, a method to identify GWAS variant enrichment among closely related tissues/cell types, using scATAC-seq data. The objective is to measure the trait relevance of different cells or tissues, and here scATAC-seq data together with fine-mapped GWAS variants are used to measure such trait relevance scores (and causal variants). It computes the bias-corrected Z-scores to estimate the trait relevance for every single cell by integrating the probability of variant causality and quantitative strength of chromatin accessibility. The trait-peak matrix is a count matrix, which is used to compute the expected number of fragments per peak per sample, which is then multiplied with the fine-mapped variant posterior probabilities. Validated with respect to the S-LDSC method.

[Variant to function mapping at single-cell resolution through network propagation - scAVENGE - Yu et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/35668323/) Uses network propagation on causal variants to identify their relevant cell types using single-cell resolution. Using the g-chromVAR output, i.e. single cell-based trait-relevance scores, we rank the cells and select the top cells as seed cells, which are used for the network propagation algorithm. Using the random walk algorithm, we reach the stationary state of network connectivity among these cells, and the final trait relevance scores (TRS) are computed for each cell.

[Identifying disease-critical cell types and cellular processes by integrating single-cell RNA-sequencing and human genetics - scLinker - Jagadeesh et al. Nat Genet 2022](https://pubmed.ncbi.nlm.nih.gov/36175791/) Integrates GWAS summary statistics, epigenomics, and scRNA-seq data from multiple tissue types, diseases, individuals, and cells. The authors transform gene programs to SNP annotations using tissue-specific enhancer–gene links, standard gene window-based linking strategies such as MAGMA, RSS-E, and linkage disequilibrium score regression (LDSC)-specifically expressed genes. Then they link SNP annotations to diseases by applying stratified LDSC (S-LDSC) to the resulting SNP annotations. Cell-type gene programs are computed by converting the p-values of the DE genes (between the target cell type to others) using a chi-sq distribution and applying o-1 range normalization. Similar work was done for disease-relevant gene programs. NMF was used to define cellular process gene programs. 

[Polygenic enrichment distinguishes disease associations of individual cells in single-cell RNA-seq data - scDRS - Zhang et al. Nat Genet 2022](https://pubmed.ncbi.nlm.nih.gov/36050550/) scDRS method. Enrichment of cell types with respect to GWAS trait. First identifies the putative gene sets for individual GWAS traits or diseases using MAGMA. Then identifies the cell type and single-cell level correlation between the gene set and cells, and computes the GWAS enrichment of a cell type.

[NumBat - Gao et al. Nat Biotech 2023](https://pubmed.ncbi.nlm.nih.gov/36163550/) Haplotype aware CNV inference from scRNA-seq data. CNVs are inferred both from expression (expecting AMP and DEL to be associated with up/downregulation - FP for expression changes unrelated to CNV) and allele-specific (deviations of BAF - less affected by sample-specific variation). This method uses haplotype phasing prior to detecting CNVs.

[Mapping interindividual dynamics of innate immune response at single-cell resolution - Kumasaka et al. Nat Genet 2023](https://pubmed.ncbi.nlm.nih.gov/37308670/) Presents GASPACHO, a Gaussian Process + Latent Variable based model to infer the single cell eQTLs associated with dynamic cell states obtained from immune response/stimuli. Test their method on Fibroblast scRNA-seq data and also colocalizes with COVID-19 GWAS data to identify colocalized sc-eQTLs associated with OAS1 and OAS3 genes. Models eQTLs without pseudobulk generation. Generates cell-state specific eQTLs by considering the technical batches and donor specific variations as covariates. Classifies eQTLs as static and dynamic eQTLs. eGenes are depleted in differential genes but enriched in highly expressed genes. sc-eQTLs are also applied fine mapping using epigenomic data. Colocalization reveals enrichment of autoimmune GWAS traits.

[POPS - Weeks et al. Nat Genet 2023](https://www.nature.com/articles/s41588-023-01443-6) PoPs method for prioritizing gene sets from GWAS summary statistics and LD matrix. Uses MAGMA to first compute the gene-based scores. Also uses pathways, PPIs, etc. to prioritize groups of genes with similar effects/features and uses them to compute the gene-based enrichment statistics, using a multivariate normal (MVN) distribution-based regression strategy. 

## Gene regulatory network (GRN)

[SCENIC: single-cell regulatory network inference and clustering - Aibar et al. Nat Meth 2017](https://pubmed.ncbi.nlm.nih.gov/28991892/) GRN from scRNA-seq data. Predicts interactions between TFs and target genes. Coexpression is computed at a very limited distance (~20 Kb) between genes and TFs. 1) Co-expressed TF and gene by GENIE3, 2) Putative TFs by RCisTarget (motif discovery), 3) AUCell algorithm for cell-specific regulons.

[SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks - Gonzlez-Blas et al. Nature Methods 2023](https://www.nature.com/articles/s41592-023-01938-4) Extends SCENIC by using  scATAC-seq data to identify the enhancers associated with candidate TFs and identify their correlation with the candidate gene expression. The co-accessibility of peaks helps to identify and examine the TFs related to selective peaks up to 150 Kb distance from the gene. Uses GRNBoost2 to quantify the importance of both TFs and enhancer candidates for target genes and it infers the direction of regulation (activating/repressing) using linear correlation. Benchmarks GRNs with those from reference methods, by checking coverage of biological cell states (PCA), recovery of differentially expressed TFs, overlap of TF ChIP-seq peaks, Hi-C contacts, comparing TF perturbation scores per genes (regression model). Also implements a ranking of TFs, regions and genes.

[Dissecting cell identity via network inference and in silico gene perturbation - CellOracle - Kamimoto et al. Nature 2023](https://pubmed.ncbi.nlm.nih.gov/36755098/) CellOracle method. Uses DNA sequence, co-accessibility and gene expression clusters. Overview: 1) Constructs GRN from multi-omic data and generates cell-state specific GRN models. 2) Simulates the effect of dynamic GRN (or cell states) following TF perturbation (basically effect of purturbation on GRN). 3) Applies to systematically purturb TFs across Zebrafish development. 4) Models the "shift" in gene expression rather than its absolute values. 5) Uses genomic sequences and TF binding motifs to infer the base GRN structure and dimensionality. Step 1: The base GRN contains unweighted directional edges between a TF and a gene (co-accessibility peaks with max 500 Kb distance computed by Cicero, and using HOMER CRE database). Without using sample specific scATAC-seq data, it uses base / average mouse scATAC-seq atlas. Step 2: Uses scRNA-seq data to identify active connections in the base GRN by a regularized linear ML model.

[Dictys: dynamic gene regulatory network dissects developmental continuum with single-cell multiomics - Wang et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37537351/) Dictys method for GRN inference. Input: scRNA-seq, scATAC-seq data, ChIP-seq / regulatory annotations to define context specific regulatory elements. Novelty: Context specific GRN using gene expression and regulatory activity for dfferent contexts. Also infers dynamic GRNs using the velocity or trajectory.

## Trajectory analysis /  RNA velocity

[Monocle - Trapnell et al. Nat Biotech 2014](https://pubmed.ncbi.nlm.nih.gov/24658644/) Trajectory analysis from scRNA-seq data. ICA is used for dimensionality reduction, after selecting a subset of genes with respect to the variance explained. Then, the minimum spanning tree (MST) is used to construct the lineage. A PQ-Tree-specific algorithm is used to deal with the branching noise.

[RNA Velocity - La Manno et al. Nature 2018](https://pubmed.ncbi.nlm.nih.gov/30089906/) Concept of RNA velocity using the spliced and unspliced RNA. Provides a toolkit Velocyto. Assumes that the transcriptional regulation parameters are the same for all genes, and each gene has a sufficient time frame to reach the steady state.

[scVelo - Bergen et al. Nat Biotech 2020](https://pubmed.ncbi.nlm.nih.gov/32747759/) scVelo method, extending the RNA velocity concept on scRNA-seq data by modeling the transcriptional regulation parameters in a probabilistic model. Also improves the running time.

[multiVelo - Li et al. Nat Biotech 2023](https://pubmed.ncbi.nlm.nih.gov/36229609/) multiVelo approach. Using both scRNA-seq and scATAC-seq data for velocity estimation. Based on the fact that epigenomic changes (like the transition from euchromatin to heterochromatin) have a role in transcriptional regulation and rates. Uses ODE with switch and rate parameters. Inputs: time-varying levels of chromatin accessibility, unspliced pre-RNA, and spliced mature RNA. Parameters: rates of chromatin opening and closing, RNA transcription, RNA splicing, and RNA degradation of nuclear export.

[https://www.nature.com/articles/s41592-021-01346-6 - Lange et al. Nature Methods 2022](https://www.nature.com/articles/s41592-021-01346-6) Combines RNA velocity and trajectory inferernce methods. Automatically defines the root and terminal cells. *Input*: gene expression matrix, and RNA velocity matrix (or any vector field with a direction measure). *Method*: Computes transition probabilities, using: 1) RNA velocity, 2) Gene expression similarity of cells. Applies a weighted mixing - correlation of expression to the neighboring cells, and to the cells implied by the RNA velocity vector. Formulates markov chain and transition matrix accordingly. Assigns cells to macrostates (group of cells) and computes direction according to the states. Computing macrostates is done by generalized perron cluster cluster analysis (GPCCA). Identification of terminal states: Stability Index (SI), initial state: coarse grained stationary distribution. *Output*: 1) Cluster membership matrix, 2) Transition matrix, 3) Fate matrix.

## Disease-specific

[Identifying tumor cells at the single-cell level using machine learning - Dohmen et al. Genome Biology 2022](https://pubmed.ncbi.nlm.nih.gov/35637521/) 

    - Presents ikarus, an ML framework to identify and annotate tumor cells from normal cells using single-cell data. 
    - Method:
        - Identifies a marker (signature) gene set to identify the tumor cell signature. Based on DEGs and intersection among cancer-specific datasets. Total 162 genes significantly enriched in cancer cells across multiple datasets.
        - Logistic regression classifier for discriminating tumor and normal cells. AUCell method is used for cell scoring using marker genes.
        - Network based propagation of cell labels using cell-cell network.
        - CNV scores are also used for improved classifcation of tumor cells.
        - Additionally, bulk RNA-seq specific marker genes were also tested and their gene signature scores were computed using ssGSEA. Published gene sets from mSigDB, CancerSEA were also used for validation.

[Epigenomic dissection of Alzheimer’s disease pinpoints causal variants and reveals epigenome erosion - Xiong et al. Cell 2023](https://www.cell.com/cell/pdf/S0092-8674(23)00974-1.pdf) Alzheimer's disease (control, early and late) snRNA-seq and snATAC-seq data. Contributions: 1) Iterative approach of data integration and updating peak to gene links (using ArchR based scores) by analyzing subset of cells and re-estimation of gene score matrices (using ArchR), 2) chromVAR (in ArchR) to identify TFs enriched for different group of cells, 3) Identifying peak modules using modified BB-kNN approach, 4) Differential accessible peaks, 5) AD GWAS heritability enrichment analysis using S-LDSC, 6) single cell ATAC QTLs using SVD + pseudo-bulk gene expression profile and multivariate regression, 7) Colocalization between ATAC-QTLs and GWAS (PP4 > 0.1 ?? ) 8) Cell-type sharing of ATAC-QTLs using directionality-dependent method, 8) Propeller method to understand the cell-type composition changes in single-cell data.

[Spatially resolved multiomics of human cardiac niches - Kanemaru et al. Nature 2023](https://pubmed.ncbi.nlm.nih.gov/37438528/) Single cell and spatial transcriptomics for cardiac data. Uses CellPhoneDB to model cell-cell interactions. Also develops a pipeline **drug2cell** which integrates drug–target interactions from the ChEMBL database with user-provided single-cell data to comprehensively evaluate drug-target expression in single cells.

[Chromatin and gene-regulatory dynamics of the developing human cerebral cortex at single-cell resolution - Trevino et al. Cell 2021](https://pubmed.ncbi.nlm.nih.gov/34390642/) scRNA-seq + scATAC-seq atlas on cortical development, for modeling neurodevelopmental disorders. CREs are linked with gene expression using co-accessibility, genes with predictive chromatin (GPC), gene expression - regulatory element linkage analysis, etc. GSEA and TF motif enrichment analysis are carried out, and trajectory analysis is used to link GPCs to cell fates and cell states, using a cell cycle signature (MSigDB) + projection of ATAC-seq pseudobulks and multi-omics (scRNA + scATAC) into fuzzy c-means clustering space. Finally, BPNET is used to prioritize cluster (cell type or cell state) specific variant enrichment and scoring analysis, using a deep learning framework. 

[Pathformer: a biological pathway informed Transformer integrating multi-omics data for disease diagnosis and prognosis - Liu et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.05.23.541554v6.full) 
    
    - Pathformer method. 
    - Integrates multi-modal single cell data to first produce gene level embedding (multi-modality representative vector) which are then transfomed to pathway embeddings.
    - These embeddings are then fed into a transformer framework with pathway crosstalk serving as attention, to interpret the disease outcome. 
    - Also uses Shap values to identify the genes relevant to pathways and disease. 
    Applied to Cancer and liquid biopsy for early cancer prediction.

