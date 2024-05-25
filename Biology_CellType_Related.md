# Biology / Cell type (like T cell) Related epigenomics papers

#### (understanding the regulatory regions, important genes, and SNPs related to their functions and differentiation)

## Regulatory regions, genes

[Transcriptional Enhancers in the Regulation of T Cell Differentiation - Nguyen et al. Frontiers in Immunology 2015](https://pubmed.ncbi.nlm.nih.gov/26441967/) 

    - Discussed important regulatory regions and genes for T cell differentiation. 
      - 1) Th2 lineage: expression of Cytokine IL-4 (and also IL-5, IL-13), RAD50 genes, and differentiation by TF GATA3, STAT6. 
      - 2) Th1 specific genes: Ifng and Tbx21. 
      - In general, STAT family of TFs are responsible for T cell differentiation.

[Single-cell transcriptomics identifies an effectorness gradient shaping the response of CD4+ T cells to cytokines - Cano-Gamez et al. Nature Comm 2020](https://www.nature.com/articles/s41467-020-15543-y) 

    - Derives T cell differentiation lineages using single cell transcriptomics and prioritizes cytokines and genes important for T cell differentiation.
    - Cell type specific gene expression programs in CD4+ T cells.

[A CD4+ T cell reference map delineates subtype-specific adaptation during acute and chronic viral infections - Andreatta et al. eLife 2022](https://elifesciences.org/articles/76339) 

    - Used scRNA-seq and TCR to identify key marker genes for T cell differentiation (check figures 1-4 for a complete list of T cell lineage deriving marker genes).
    - Models changes of CD4+ T cells in acute and chronic viral infection.

[Progesterone Inhibits the Establishment of Activation-Associated Chromatin During TH1 Differentiation - Rundquist et al. Frontiers Immunology 2022](https://www.frontiersin.org/articles/10.3389/fimmu.2022.835625/full) 

    - Discussed marker genes for T cell differentiation to Th1 (Fig. 2) and also the progesteron hormone during pregnancy and its effect on MS and RA.
    - ATAC-seq and RNA-seq time series data on TH1 differentiation.

[Tracking DNA-based antigen-specific T cell receptors during progression to type 1 diabetes - Mitchell et al. Science Advances 2023](https://pubmed.ncbi.nlm.nih.gov/38064552/) 

    - Deep sequencing of TCR alpha and beta receptor sequences and identifying the differences of antigen specific TCR receptors between T1D cohorts and control. 
    - To Do: Can we design a ML framework which can identify such differences?

[Base-editing mutagenesis maps alleles to tune human T cell functions - Schmidt et al. Nature 2023](https://pubmed.ncbi.nlm.nih.gov/38093011/) 

    - Proposes multiplexed parallel CRISPR screen based base editing scheme (mutagenesis) for human B and T cells, to validate the functional genes and variants.
    - Identifies mutations effecting activation and cytokine production
      - These guides are predicted to have high changes in BLOSUM62 scores (protein structure) as predicted by Alphafold.

## SNPs

[Soskic et al. Nat Genet 2022](https://www.nature.com/articles/s41588-022-01066-3) derived single cell eQTLs and in particular, time-dependent dynamic eQTLs for T cell differentiation (Suppl. Table 8). They used both linear and nonlinear dynamic eQTL model and derived FDR for both. They highlighted IL7R, CD69, IL2RA, IRF1, TOP2A as important genes.

[Bossini-Castillo et al. Cell Genomics 2022](https://doi.org/10.1016/j.xgen.2022.100117) identified Treg specific eQTLs, chromatinQTLs, and also colocalized with various immune disease specific GWAS SNPs. Good repository for Treg specific marker genes and SNPs.

[Ohkura et al. Immunity 2020](https://doi.org/10.1016/j.immuni.2020.04.006) identified Treg specific genes and putative causal genes, specifically in CTLA4 and IL2RA loci. They also provided a concept of GWAS LD-SNP group, by considering all SNPs with LD (R2 > 0.8) and MAF > 0.1 with the lead GWAS fine-mapped SNP (PICS) within 50 Kb, as a LD group of SNPs.
