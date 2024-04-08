# Papers discussing / utilizing enhancer - promoter contacts / loops / interactions

[GitHub page collecting Hi-C specific paper list](https://github.com/mdozmorov/HiC_tools)



## Loop Callers / Assays / multi-way chromatin contacts

[Cell-type specialization is encoded by specific chromatin topologies - Winick-Ng et al. Nature 2021](https://pubmed.ncbi.nlm.nih.gov/34789882/) ImmunoGAM + scRNA-seq + scATAC-seq assay to capture multi-way chromatin contacts. Brain related cell-specific multi-way chromatin organization, based on the GAM assay. Also propose a "melting gene" identification pipeline - identifying long stretch genes (~200 Kb) having contact loss between conditions. Also discusses about TAD and compartment changes detected by the ImmunoGAM assay.













## Inferring 3D Structure from chromatin contacts

[Pastis-NB: Nelle Varoquaux et al. Bioinformatics 2023](https://pubmed.ncbi.nlm.nih.gov/36594573/) Modeling overdispersion of HIC data to predict 3D structures, by using NB model instead of Poisson model from their previously published Pastis-PM method.




## Structural Variation (CNV, SV) analysis






## CRISPR / experimental validation of E-P contacts, regulatory regions

[SuRE - High-throughput identification of human SNPs affecting regulatory element activity - Arensbergen et al. Nature Genetics 2019](https://www.nature.com/articles/s41588-019-0455-2) A high throughput MPRA assay to capture the regulatory and putative causal eQTLs / GWAS SNPs, and annotate them using SuRE score.

[CRISPRQTL - A Genome-wide Framework for Mapping Gene Regulation via Cellular Genetic Screens - Gasperini et al. Cell 2019](https://pubmed.ncbi.nlm.nih.gov/30612741/) validated E-P pairs by high throughput CRISPR perturbations in thousands of cells. Tested 1119 candidate enhancers in K562. Enhancers were intergenic DHS - H3K27ac, P300, GATA, and RNA PolII binding sites. 5611 of 12984 genes expressed in K562 fell within 1 Mb of the tested enhancers.

[Compatibility rules of human enhancer and promoter sequences - Bergman et al. Nature 2022](https://pubmed.ncbi.nlm.nih.gov/35594906/) proposed Exp-STARRseq - compatibility rule between enhancer and promoter sequences. Tested 1000 enhancers (H3K27aca and DNase-seq peaks) and 1000 genes. Proposes a multiplicative model of enhancer effects on target gene expression (confirming ABC score effects), explaining 82% variance in STARR-seq expression. This model fits well for E-P pairs within 100 Kb. Also proposes two different classes of genes, namely variable expressed genes (weaker H3K27ac, DHS) and ubiquitously expressed genes (stronger H3K27ac, DHS), and also two different classes of enhancers (stronger and weaker according to H3K27ac and DHS levels).

[Genome-wide enhancer maps link risk variants to disease genes - Nasser et al. Nature 2021](https://pubmed.ncbi.nlm.nih.gov/33828297/) applied the ABC score to prioritize regulatory enhancers for target genes, and compared it against lots of E-P linking strategies. Applied to 131 cell types, 72 diseases, and complex traits. Specifically focussed on IBD. Showed that the nearest gene principle achieves the highest recall for IBD specific variant-to-gene identification, while ABC-max achieves better precision in terms of variant-to-gene identification.

[An encyclopedia of enhancer-gene regulatory interactions in the human genome - Gschwind et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.11.09.563812v1) Extension of ABC score paper. EPI prediction and their benchmarking across multiple studies and methods. SNP to gene linking strategies

[Activity-by-contact model of enhancer-promoter regulation from thousands of CRISPR perturbations - Fulco et al. Nat Genet 2019](https://pubmed.ncbi.nlm.nih.gov/31784727/) 

    - ABC score to prioritize regulatory enhancers for target genes.
    - Multiplication of H3K27ac (and DHS) activity with HiC contacts.
    - Prioritizes nearest enhancers. 84% of genes have regulatory enhancers within 100 Kb.
    - Equivalent to inverse (power-law) distance weighted score (contact ~ Distance^-1).
    - Applies on K562 H3K27ac and DHS peaks and the HiC data.
    - Average Hi-C contact matrix performs similar as cell-type-specific contact matrix.
    - ABC predictions are highly cell-specific, as using K562 HiC data for predicting E-P pairs in Mouse drops AUPRC from 0.73 to 0.11.
    - *To Do* Using H3K27ac HiChIP with DHS signal shows better performance than ABC score (Extended data fig. 3).
      - But capture Hi-C does not perform reasonably.
    - Developed CRISPRi-FlowFISH to perturb noncoding enhancers.
    - Although enhancer prioritization works, promoters (DP-G model) are not well characterized. 

[Sequence determinants of human gene regulatory elements - Sahu et al. Nat Genet 2022](https://pubmed.ncbi.nlm.nih.gov/35190730/) measured transcriptional activity using MPRA and observed using ML models that TFs act in an additive manner and enhancers affect promoters without relying on specific TF-TF interactions. Few TFs are strongly active in a cell, and individual TFs may have multiple regulatory activities. Shows 3 types of enhancers - classical (open chromatin), chromatin-dependent, closed-chromatin. Shows that TSS position and motif position and orientation relative to TSS are highly indicative of the gene expression and presents a CNN model to predict the TSS position from gene expression and sequence information. Also found that E-P interactions are additive - their effects are integrated into total transcriptional activity, suggesting that strong promoters do not need an enhancer, and strong enhancers render weak and strong promoters equally active. A CNN classifier using only promoter sequences outperform that with enhancer sequences for predicting gene expression.

[Endogenous fine-mapping of functional regulatory elements in complex genetic loci - Zhao et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.05.06.539696v1) Single cell CRISPR inhibition and activation on the CREs linked to fine-mapped eQTLs from GTEX v8, Geuvadis, Blueprint databases to identify the fine-mapped causal variants. Tested only those loci which have multiple lead QTLs with the same p-value, and they are in perfect LD.

[Deciphering eukaryotic gene-regulatory logic with 100 million random promoters - de Boer et al. Nat Biotech 2020](https://pubmed.ncbi.nlm.nih.gov/31792407/) Predicts TF binding from sequence using ML. Aplies on Yeast. Develops GPRA (Gigantic parallel reporter assay). 

[The evolution, evolvability and engineering of gene regulatory DNA - Vaishnav et al. Nature 2022](https://pubmed.ncbi.nlm.nih.gov/31792407/) Work from the same goup. Predicts sequence to gene expression, using DL, and using promoter characteristics, fitness scores etc. Mostly biophysical model is used in a DL framework.

[Genome-wide analysis of CRISPR perturbations indicates that enhancers act multiplicatively and without epistatic-like interactions - Zhou et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.04.26.538501v1) Presents GLiMMIRS, a generalized linear model for measuring interactions between regulatory sequences, uses CRISPRi data from the CRISPRQTL work, and presents a multiplicative model of enhancer interactions.

[Discovery of target genes and pathways at GWAS loci by pooled single-cell CRISPR screens - Morris et al. Science 2023](https://pubmed.ncbi.nlm.nih.gov/37141313/) Presents base editing STING-seq to purturb CREs using CRISPRi based on target fine-mapped GWAS variants. Applies on single cells. Candidate CREs are derived by fine-mapping. In addition, for differential expression testing, the authors used a recent work [SCEPTRE - Barry et al. Genome Biology 2021](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02545-2) which connects CRISPR purturbations with changes in gene and protein expression. Most of the cis target genes were proximal to the variants. K562 cell type and HiChIP data was used for validation.

[Complementary Alu sequences mediate enhancer–promoter selectivity - Liang et al. Nature 2023](https://pubmed.ncbi.nlm.nih.gov/37438529/) Presents RIC-seq to understand the E-P RNA interactions (EPRI) between ALU and non-ALU complementary sequences. Using CRISPRi type validation, they showed that ALU RNA sequences are crucial for E-P interaction selectivity.

[Determinants of Base Editing Outcomes from Target Library Analysis and Machine Learning - Arbab et al. Cell 2020](https://pubmed.ncbi.nlm.nih.gov/32533916/) Be-Hive method. ML based technique to predict the regulatory effects of CRISPRI or other base editing approaches. Specifically it considers edits of A and C bases. 

[3D Enhancer-promoter networks provide predictive features for gene expression and coregulation in early embryonic lineages - Murphy et al. Nat Struct and Mol Biol 2023](https://pubmed.ncbi.nlm.nih.gov/38053013/) 3D EPI of early embryonic lineages, gene expression regulated by using EPI and Enhancer Hubs. Uses HiChIP. Also proposes 3D-HiChAT method, to predict gene expression (high vs low) and absolute expression levels (correlation) using 1D and 3D epigenomic features and random forest ML method. Also tests perturbation experimental results and show comparable performance with ABC score. Finally, presents experimentally validated loci in ESC cell line.


## Applications to disease / cell type based models

[Cell-type-specific 3D epigenomes in the developing human cortex - Song et al. Nature 2020](https://pubmed.ncbi.nlm.nih.gov/33057195/) Human Cortex and different cell-specific gene identification using H3K4me3 PLAC-seq loops. Concept of super interacting promoters (SIP) and human gained enhancers (HGE). SIPs are shown to be cell-specific and lineage determining. QTL and GWAS annotations are enriched in SIPs by S-LDSC and H-MAGMA. Developed CRISPRView to validate cell-specific promoters and regulatory elements.




