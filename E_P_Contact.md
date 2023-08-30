# Papers discussing / utilizing enhancer - promoter contacts / loops / interactions


## Loop Callers






## Inferring 3D Structure from chromatin contacts

[Pastis-NB: Nelle Varoquaux et al. Bioinformatics 2023](https://pubmed.ncbi.nlm.nih.gov/36594573/) Modeling overdispersion of HIC data to predict 3D structures, by using NB model instead of Poisson model from their previously published Pastis-PM method.




## Structural Variation (CNV, SV) analysis






## CRISPR / experimental validation of E-P contacts, regulatory regions

[SuRE - High-throughput identification of human SNPs affecting regulatory element activity - Arensbergen et al. Nature Genetics 2019](https://www.nature.com/articles/s41588-019-0455-2) A high throughput MPRA assay to capture the regulatory and putative causal eQTLs / GWAS SNPs, and annotate them using SuRE score.

[CRISPRQTL - A Genome-wide Framework for Mapping Gene Regulation via Cellular Genetic Screens - Gasperini et al. Cell 2019](https://pubmed.ncbi.nlm.nih.gov/30612741/) validated E-P pairs by high throughput CRISPR perturbations in thousands of cells. Tested 1119 candidate enhancers in K562. Enhancers were intergenic DHS - H3K27ac, P300, GATA, and RNA PolII binding sites. 5611 of 12984 genes expressed in K562 fell within 1 Mb of the tested enhancers.

[Compatibility rules of human enhancer and promoter sequences - Bergman et al. Nature 2022](https://pubmed.ncbi.nlm.nih.gov/35594906/) proposed Exp-STARRseq - compatibility rule between enhancer and promoter sequences. Tested 1000 enhancers (H3K27aca and DNase-seq peaks) and 1000 genes. Proposes a multiplicative model of enhancer effects on target gene expression (confirming ABC score effects), explaining 82% variance in STARR-seq expression. This model fits well for E-P pairs within 100 Kb. Also proposes two different classes of genes, namely variable expressed genes (weaker H3K27ac, DHS) and ubiquitously expressed genes (stronger H3K27ac, DHS), and also two different classes of enhancers (stronger and weaker according to H3K27ac and DHS levels).

[Genome-wide enhancer maps link risk variants to disease genes - Nasser et al. Nature 2021](https://pubmed.ncbi.nlm.nih.gov/33828297/) applied the ABC score to prioritize regulatory enhancers for target genes, and compared it against lots of E-P linking strategies. Applied to 131 cell types, 72 diseases, and complex traits.

[Activity-by-contact model of enhancer-promoter regulation from thousands of CRISPR perturbations - Fulco et al. Nat Genet 2019](https://pubmed.ncbi.nlm.nih.gov/31784727/) proposed an ABC score to prioritize regulatory enhancers for target genes. Applies on K562 H3K27ac and DHS peaks and the HiC data. Multiplicative model to prioritize the nearest enhancers.

[Sequence determinants of human gene regulatory elements - Sahu et al. Nat Genet 2022](https://pubmed.ncbi.nlm.nih.gov/35190730/) measured transcriptional activity using MPRA and observed using ML models that TFs act in an additive manner and enhancers affect promoters without relying on specific TF-TF interactions. Few TFs are strongly active in a cell, and individual TFs may have multiple regulatory activities. Shows 3 types of enhancers - classical (open chromatin), chromatin-dependent, closed-chromatin. Shows that TSS position and motif position and orientation relative to TSS are highly indicative of the gene expression and presents a CNN model to predict the TSS position from gene expression and sequence information. Also found that E-P interactions are additive - their effects are integrated into total transcriptional activity, suggesting that strong promoters do not need an enhancer, and strong enhancers render weak and strong promoters equally active. A CNN classifier using only promoter sequences outperform that with enhancer sequences for predicting gene expression.

