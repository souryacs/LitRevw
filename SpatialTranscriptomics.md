
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

[Deep learning and alignment of spatially resolved single-cell transcriptomes with Tangram - Biancalani et al. Nat Meth 2021](https://pubmed.ncbi.nlm.nih.gov/34711971/) TANGRAM method for alignment between ST and scRNA-seq / snRNA-seq / multi-omic data, collected from the same origin, and at least having shared marker genes. Identifies the gene expression patterns and the spatial coordinates at cell resolution. Supports various protocols like MERFISH, STARmap, smFISH, Visium, and images. Objective function is to mimic the spatial correlation between each gene in the sc/snRNA-seq data and the spatial data. Cell density is compared by KL divergence, gene expression is assessed by cosine similarity. Assumes that cell segmentation is already done, using tools like ilastik or nucleAIzer. A few hundred marker genes are recommended for alignment.

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

[Spatial transcriptomics at subspot resolution with BayesSpace - Zhao et al. Nat Biotech 2021](https://pubmed.ncbi.nlm.nih.gov/34083791/) 

    - Implements BayesSpace to model ST data. 
    - Minor adjustments of HMRF by implementing MCMC instead of EM algorithm in the spatial refinement. 
    - Also, employs a fixed precision matrix (similar across individual clusters for less parameter estimation).     
    - Employs t-distributed error model to identify spatial clusters. Robust against outliers.
    - Performs resolution enhancement by segmenting each spot into subspots (9 or 6 depending on the platform).
        - Equal division of spots into subspots based on the diameter, without any quantitative model.
        - Visium spots contain ~20 cells so subspots contain ~3 cells.

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

