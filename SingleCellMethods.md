# Computational approaches for Single Cell Data

Check [This GitHub page](https://github.com/OmicsML/awesome-deep-learning-single-cell-papers) for a comprehensive summary of deep learning and single-cell papers.

Check [This eBook from Fabian Theis group](https://www.sc-best-practices.org/preamble.html) about the best practices in single cell data analysis.

Check [This workshop of Broad Institute](https://broadinstitute.github.io/2020_scWorkshop/index.html) for a detailed overview of single cell data processing.



## Trajectory analysis /  RNA velocity




[Multi-omic single-cell velocity models epigenome–transcriptome interactions and improves cell fate prediction - Li et al. Nat Biotech 2023](https://pubmed.ncbi.nlm.nih.gov/36229609/)

    - multiVelo approach. 
    - *Inputs*: time-varying levels of chromatin accessibility (scATAC-seq), unspliced pre-RNA, and spliced mature RNA (scRNA-seq). 
    - *Parameters*: rates of chromatin opening and closing, RNA transcription, RNA splicing, and RNA degradation of nuclear export.
    - *Method*:
        - ODE with switch and rate parameters. 
        - Uses the fact that transcription rate is proportional to the chromatin accessibility.
            - Epigenomic changes (like the transition from euchromatin to heterochromatin) have a role in transcriptional regulation and rates.
        - Two states are defined each for chromatin accessibility (c) and RNA (u,s) : 
            - chromatin opening, chromatin closing, transcriptional induction and transcriptional repression (priming and decoupling).
            - Each gene uses 3D phase portraits (c,u,s), and defines two different models M1 and M2 based on two potential orderings of chromatin and RNA state changes.

[CellRank for directed single-cell fate mapping - Lange et al. Nature Methods 2022](https://www.nature.com/articles/s41592-021-01346-6) 

    - Combines RNA velocity and trajectory inferernce methods. 
    - Automatically defines the root and terminal cells. 
    - *Input*: gene expression matrix, and RNA velocity matrix (or any vector field with a direction measure). 
    - *Method*: 
        - Computes transition probabilities, using: 1) RNA velocity, 2) Gene expression similarity of cells. 
        - Applies a weighted mixing - correlation of expression to the neighboring cells, and to the cells implied by the RNA velocity vector. 
        - Formulates markov chain and transition matrix accordingly. 
        - Assigns cells to macrostates (group of cells) and computes direction according to the states. 
        - Computing macrostates is done by generalized perron cluster cluster analysis (GPCCA). 
        - Identification of terminal states: Stability Index (SI), initial state: coarse grained stationary distribution. 
    - *Output*: 1) Cluster membership matrix, 2) Transition matrix, 3) Fate matrix.

## Disease-specific

[Identifying tumor cells at the single-cell level using machine learning - Dohmen et al. Genome Biology 2022](https://pubmed.ncbi.nlm.nih.gov/35637521/) 

    - Presents ikarus, an ML framework to identify and annotate tumor cells from normal cells using single-cell data. 
    - Method:
        - Identifies a marker (signature) gene set to identify the tumor cell signature. 
            - Based on DEGs and intersection among cancer-specific datasets. 
            - Total 162 genes significantly enriched in cancer cells across multiple datasets.
        - Logistic regression classifier for discriminating tumor and normal cells. 
            - AUCell method is used for cell scoring using marker genes.
        - Network based propagation of cell labels using cell-cell network.
        - CNV scores are also used for improved classifcation of tumor cells.
        - Additionally, bulk RNA-seq specific marker genes were also tested and their gene signature scores were computed using ssGSEA. 
            - Published gene sets from mSigDB, CancerSEA were also used for validation.

[Epigenomic dissection of Alzheimer’s disease pinpoints causal variants and reveals epigenome erosion - Xiong et al. Cell 2023](https://www.cell.com/cell/pdf/S0092-8674(23)00974-1.pdf) 

    - Alzheimer's disease (control, early and late) snRNA-seq and snATAC-seq data. 
    - *Contributions*: 
        - 1) Iterative approach of data integration and updating peak to gene links (using ArchR based scores) by analyzing subset of cells and re-estimation of gene score matrices (using ArchR), 
        - 2) chromVAR (in ArchR) to identify TFs enriched for different group of cells, 
        - 3) Identifying peak modules using modified BB-kNN approach, 
        - 4) Differential accessible peaks, 
        - 5) AD GWAS heritability enrichment analysis using S-LDSC, 
        - 6) single cell ATAC QTLs using SVD + pseudo-bulk gene expression profile and multivariate regression, 
        - 7) Colocalization between ATAC-QTLs and GWAS (PP4 > 0.1 ?? ) 
        - 8) Cell-type sharing of ATAC-QTLs using directionality-dependent method, 
        - 9) Propeller method to understand the cell-type composition changes in single-cell data.

[Spatially resolved multiomics of human cardiac niches - Kanemaru et al. Nature 2023](https://pubmed.ncbi.nlm.nih.gov/37438528/) 

    - Single cell and spatial transcriptomics for cardiac data. 
    - Uses CellPhoneDB to model cell-cell interactions. 
    - Also develops a pipeline **drug2cell**  
        - integrates drug–target interactions from the ChEMBL database with user-provided single-cell data
        - Comprehensively evaluates drug-target expression in single cells.

[Chromatin and gene-regulatory dynamics of the developing human cerebral cortex at single-cell resolution - Trevino et al. Cell 2021](https://pubmed.ncbi.nlm.nih.gov/34390642/) 

    - scRNA-seq + scATAC-seq atlas on cortical development, for modeling neurodevelopmental disorders. 
    - CREs are linked with gene expression using co-accessibility, genes with predictive chromatin (GPC), gene expression - regulatory element linkage analysis, etc. 
    - GSEA and TF motif enrichment analysis 
    - trajectory analysis is used to link GPCs to cell fates and cell states 
        - using a cell cycle signature (MSigDB) 
        - also performs projection of ATAC-seq pseudobulks and multi-omics (scRNA + scATAC) into fuzzy c-means clustering space. 
    - Finally, BPNET is used to prioritize cluster (cell type or cell state) specific variant enrichment and scoring analysis, using a deep learning framework. 

[Pathformer: a biological pathway informed Transformer integrating multi-omics data for disease diagnosis and prognosis - Liu et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.05.23.541554v6.full) 
    
    - Pathformer method. 
    - *Input*: Cancer tissue datasets from Cancer Genome Atlas (TCGA) 
        - includes gene-level RNA expression, fragment-level DNA methylation, and both fragment-level and gene-level DNA CNV. 
        - These are all concatenated as gene level embedding features.
        - Also inputs are the pathways from different databases, which are used for validation.
    - *Output*: Integrated multi-modal single cell gene level embeddings (multi-modality representative vector) which are then transfomed to pathway embeddings.
        - These embeddings are then fed into a transformer framework with pathway crosstalk serving as attention, to interpret the disease outcome. 
        - Applied to Cancer and liquid biopsy for early cancer prediction.
    - Method:
        - Employs a sparse neural network architecture to deal with multimodal vectors of high dimension.
        - Also uses SHAP values to identify the genes relevant to pathways and disease. 
        - Benchmarks with 18 other integration methods in various classification tasks using mutiple cancer tissue datasets from TCGA.
