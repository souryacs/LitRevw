
[Collection of spatial transcriptomics and ML papers](https://github.com/SindiLab/Deep-Learning-in-Spatial-Transcriptomics-Analysis)


## Reviews

[Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type deconvolution - Review paper on integration between ST and scRNA-seq - Li et al. Nat Meth 2022](https://pubmed.ncbi.nlm.nih.gov/35577954/): Considers performance metrics: Pearson correlation coefficients (PCC), structural similarity index (SSIM), RMSE, Jensen-Shannon divergence (JS), accuracy score (AS), robustness score (RS). 1) Tangram and gimVI outperformed the other integration methods on the basis of these metrics. 2) Considering sparse datasets, Tangram, gimVI, and SpaGE outperformed other integration methods in predicting the spatial distribution of transcripts for highly sparse datasets. 3) In predicting cell type composition of spots, Cell2location, SpatialDWLS, RCTD, and STRIDE outperformed the other integration methods. 4) In terms of computational efficiency, Tangram and Seurat are the top two most-efficient methods for processing cell-type deconvolution of spots.


## Imaging based ST data - segmentation

[Cell segmentation in imaging-based spatial transcriptomics](https://pubmed.ncbi.nlm.nih.gov/34650268/) 

    - Baysor method (implemented in Julia). 
    - Objective: Cell segmentation from imaging based ST data.
        - Discusses the limitations of segmentation free approaches in imaging based ST datasets.
    - Segmentation free approaches rely on the cell transcriptional profiles 
        - Forms a neighborhood composition vector (NCV) using K spatially nearest neighbors, to define the patch like structure.
        - However, NCVs near cell boundaries may represent mixture of cell types, similar to doublets.
    - Segmentation / labeling approach:
        - Determining spatial clustering - MRF priors
        - Labels are considered as latent variables and inferred from observed data using EM algorithm.
        - Label probability model: multinomial distribution of observed scRNA-seq cell specific transcriptional profiles.
    - Baysor: MRF segmentation, using Bayesian Mixture models.
        - Uses either molecular positions, or auxiliary nuclear staining information.

## Alignment between ST and scRNA-seq (or multi-omic data like SHARE-seq)

[Gene expression cartography - Nitzan et al. Nature 2019](https://pubmed.ncbi.nlm.nih.gov/31748748/) 

    - novoSpaRc method - Optimal transport to map single cell transcriptomics to spatial locations.
    - Objective: infer the spot locations of the single cells from scRNA-seq data.
    - Use of reference ST atlas is optional.
    - Assumption: cells are physically close to share similar transcription profiles.
    - Weighted average of two measures for optimization modeling: 
        - D1 = embedding between expression space (scRNA-seq) and physical space (ST) using a quadratic loss function. 
        - D2 = discrepancy with respect to the reference atlas. 
    - Also performs entropy regularization.

[Inferring spatial and signaling relationships between cells from single cell transcriptomic data - Cang et al. Nature Comm 2020](https://pubmed.ncbi.nlm.nih.gov/32350282/) 

    - SpaOTsc method - to summarize in detail.

[SpaGE- Spatial Gene Enhancement using scRNA-seq - Abdelaal et al. Nucleic Acids Research 2020](https://pubmed.ncbi.nlm.nih.gov/32955565/)

    - SpaGE method.
    - Input: scRNA-seq / snRNA-seq data (reference) + ST data (query) from the same region / cell type
    - Output: predicting expression of the ST data specific unmeasured genes.
    - Method:
        - PRECISE is used to compute the principal vectors (PVs), using a combination of PCA and SVD of cosine similarity matrix.
        - These PVs are used to align the scRNA-seq and ST datasets.
        - Undetected transcripts are imputed by weighted NN, with only cells having positive cosine similarity with the current cell are considered.

[Deep learning and alignment of spatially resolved single-cell transcriptomes with Tangram - Biancalani et al. Nat Meth 2021](https://pubmed.ncbi.nlm.nih.gov/34711971/) 

    - Tangram method for alignment between ST and scRNA-seq / snRNA-seq / multi-omic data, collected from the same origin, and at least having shared marker genes. 
    - Objective: to learn map the cells of input scRNA-seq / snRNA-seq data to the ST data, using the reference ST data atlas.
        - Use sc/snRNA-seq data as puzzle pieces to align in space to match the shape of the spatial data.    
    - Objective function: Mimic the spatial correlation between each gene in the sc/snRNA-seq data and the spatial data.
    - Supports various protocols like MERFISH, STARmap, smFISH, Visium, and images. 
    - Downstream analysis from the mapping function:
        - *expand* from a measured subset of genes to genome-wide profiles 
        - *correct* low-quality spatial measurements
        - *map* the location of cells of different types
        - *deconvolve* low-resolution measurements to single cells
        - *resolve* spatial patterns of chromatin accessibility at single-cell resolution by aligning multimodal data.
    - Methodology:
        - nonconvex optimization, by minimizing KL divergence between ST and scRNA-seq, and maximizing cosine similarity of gene expression.
        - Also defines an entropy regularizer to minimize the entropy of the spatial distribution of each cell.
        - As the number of cells in spatial data is much lower than scRNA-seq, uses a filter to use only the common cells between the two.
    - Output:
        - Probabilistic mapping, namely, a matrix M (of dimension *n_{cells} X n_{voxels}*) 
            - denoting the probability of finding each cell from the sc/snRNA-seq data in each voxel of the spatial data.        
            - M^{T}S (S = input scRNA-seq matrix): spatial gene expression predicted by M
        - Using this mapping M, M^{T}A (where A = reference scRNA-seq annotations) is the annotation of the ST data.

[Identification of spatial expression trends in single-cell gene expression data - Edsgard et al. Nat Meth 2018](https://pubmed.ncbi.nlm.nih.gov/29553578/) trendsceek method. Identifies genes whose expressions are significantly associated with spatial pattern, using marked point process based modeling. ** TO Do: These genes can be utilized in the above mentioned Tangram method to align the ST data with scRNA-seq datasets.

[DOT- A flexible multi-objective optimization framework for transferring features across single-cell and spatial omics - Rahimi et al. arXiv 2023](https://arxiv.org/abs/2301.01682) DOT method. Integration between scRNA-seq and ST data. Optimal transport based solution to annotate the ST clusters (and spots) with the gene expression values from scRNA-seq data. Proposes multi-objective optimization, including mattching gene expression, matching cell annotation, matching ST neighborhood spot composition, etc.

## Spatial Clustering using gene expression and spatial location

[Identification of spatially associated subpopulations by combining scRNAseq and sequential fluorescence in situ hybridization data - HMRF - Zhu et al. Nat Biotech 2018](https://pubmed.ncbi.nlm.nih.gov/30371680/) 

    - First paper using spatial information for ST data clustering. 
    - After HVG selection and feature reduction, uses HMRF for spatial data. 
        - Markov property + Gibbs sampling 
    - KNN is used for initial clustering using gene expression. 
    - Then spatial prior for HMRF is initialized by Potts model. 
    - The multinomial distribution is employed to decide the membership of individual cells/spots in a cluster 
        - cluster refinement is done by the EM algorithm. 
    - Implemented in the Giotto toolbox framework [Dries et al. Genome Biology 2021](https://pubmed.ncbi.nlm.nih.gov/33685491/).
    - scRNA-seq clustering: Top n genes are selected. K means clustering (guide).
    - Spatial information is modeled by HMRF approach -  refine the clusters and generate the spatial domain.
        - For HMRF, spatial distance between the nodes is computed using weighted gene expression (based on selected top variable genes).

[Spatial transcriptomics at subspot resolution with BayesSpace - Zhao et al. Nat Biotech 2021](https://pubmed.ncbi.nlm.nih.gov/34083791/) 

    - Implements BayesSpace to model ST data. 
    - Minor adjustments of HMRF by implementing MCMC instead of EM algorithm in the spatial refinement. 
    - Also, employs a fixed precision matrix (similar across individual clusters for less parameter estimation).     
    - Inferred expression values are actually at PC level. So XGBoost prediction model is employed to convert PC values into raw gene expression.
    - Employs t-distributed error model to identify spatial clusters. Robust against outliers.
    - Performs resolution enhancement by segmenting each spot into subspots (9 or 6 depending on the platform).
        - Equal division of spots into subspots based on the diameter, without any quantitative model.
        - Visium spots contain ~20 cells so subspots contain ~3 cells.
    - DEGs are generated from the gene expression values obtained from enhanced resolution.

## Imaging based methods

[Super-resolved spatial transcriptomics by deep data fusion - Bergenstråhle et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/34845373) Integrates spatial gene expression data with histology image data to predict the gene expression at cellular levels. Implements a deep generative model. Also uses evidence lower bound (ELBO) to define max 40 marker genes (denoted at metagenes).

[Learning consistent subcellular landmarks to quantify changes in multiplexed protein maps - Spitzer et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37248388/) CAMPA method. Profiling ST data using conditional variational autoencoder (cVAE). Uses scanpy and squidpy frameworks. Using the ST data and molecular imaging, performs segmentation at subcellular levels.

[SCS: cell segmentation for high-resolution spatial transcriptomics - Chen et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/37429992) SCS: ST data subcellular segmentation using both sequence and imaging data. Uses imaging data for initial segmentation, and then uses transformer framework to refine the foreground and background subcellular information.

## Cell-Cell Communication

[Screening cell-cell communication in spatial transcriptomics via collective optimal transport - Cang et al. Nature Methods 2023](https://pubmed.ncbi.nlm.nih.gov/36690742) 

    - COMMOT method. 
    - Cell-Cell communication (Ligand Receptor) using ST data and optimal transport. 
    - *Input*: ST matrix - specifically cell-cell (ligand receptor) interaction matrix and spatial information for multiple species. 
    - *Method*: Presents an optimal transport based optimization (collective optimal transport) to select the putative Ligand-Receptor pairs. 
    - *To Do*: Check the eq 1 and supplementary methods. 
        - How these kind of optimization can be used in: 
            - 1) Fine mapping of SNP? Here distance matrix: LD score. 
            - 2) And SNP-Gene pair specific Summary statistics score. 

