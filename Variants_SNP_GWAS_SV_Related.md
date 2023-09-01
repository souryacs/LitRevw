# SNP / QTL / GWAS and functional variants

## Regulatory regions, motifs

[IMPACT: Genomic Annotation of Cell-State-Specific Regulatory Elements Inferred from the Epigenome of Bound Transcription Factors - Amariuta et al. AJHG 2019](https://doi.org/10.1016/j.ajhg.2019.03.012) TF binding Motif based approach to identify the cell-specific regulatory elements, using regulatory (ChIP-seq) profile. Can be further augmented by DeepLift related apporoaches. Applied this work on another work [Amariuta et al. Nat Gen 2020](https://www.nature.com/articles/s41588-020-00740-8) to predict TF binding motifs across 245 cell types, particularly to prioritize TF binding motifs and regulatory variants across multi-ancestry settings.


## QTL / ASE / GWAS inference

[fastQTL - Ongen et al. Bioinformatics 2016](https://pubmed.ncbi.nlm.nih.gov/26708335/) QTL inference by population model and efficient permutation. Default model in GTEx. Also implemented in the TensorQTL framework (including conditional eQTL analysis).

[geoP - Zhabotynsky et al. PLOS Genetics 2022](https://pubmed.ncbi.nlm.nih.gov/35286297/) is an efficient method to compute permutation-based p-values for ASE, used in TreCASE and RASQUAL.

[MixQTL - Liang et al. Nat Comm 2021](https://pubmed.ncbi.nlm.nih.gov/33658504/) models ASE using allelic fold change. The total read count and haplotype-specific read counts are separately modeled using linear regression, to provide an approximate fast QTL inference. Validated using GTEx v8 data. Implemented in tensorQTL. Appropriate for large sample sizes, but for smaller sample sizes (~100), RASQUAL is better.

[GTEx v8 - Castel et al. Genome Biology 2020](https://pubmed.ncbi.nlm.nih.gov/32912332/) Repository of haplotype-specific expression for GTEx v8 tissues. Allelic log fold change count is used for comparing SNP and haplotype-level ASE data. WASP filtering is recommended for ASE inference. 

[PLASMA - Wang et al. AJHG 2020](https://pubmed.ncbi.nlm.nih.gov/32004450/) uses both QTL and AS statistics to infer QTLs. Total expression (y) is modeled by allelic dosage (x) while the allelic imbalance (w) is determined by phasing (v). Two association statistics for QTL and AS are computed. Also performs fine mapping by using a genotype-LD matrix, and returns a credible causal set using shotgun stochastic search (SSS). Compares with fine-mapping approaches CAVIAR, AS-Meta, and RASQUAL (by converting the chi-sq statistics to z-scores and putting them as input to fine-map approaches).

[ASEP - Fan et al. PLOS Genetics 2020](https://pubmed.ncbi.nlm.nih.gov/32392242/) uses a mixture model to estimate ASE across individuals, and also computes differential ASE between conditions among groups of individuals.

[HiC-QTL: Common DNA sequence variation influences 3-dimensional conformation of the human genome - Gorkin et al. Genome Biology 2019](https://pubmed.ncbi.nlm.nih.gov/31779666/) First interaction-QTL paper. Applies HiC data on LCL to derive interaction QTLs, but does not extensively compare with conventional eQTLs. Rather, it focuses on Hi-C-specific FIRE-QTLs, etc.

[HTAL - Haplotype associated loops - Subtle changes in chromatin loop contact propensity are associated with differential gene regulation and expression - Greenwald et al. Nature Comm 2019](https://www.nature.com/articles/s41467-019-08940-5) generated phased Hi-C data from induced pluripotent stem cells (iPSCs) and iPSC-derived cardiomyocytes of seven individuals to derive 114 haplotype-associated chromatin loops (HTALs) primarily driven by imprinting and/or CNVs but not for eQTLs. Although subtle changes of these HTALs were shown to impact gene expression and H3K27ac levels, the study did not identify specific regulatory variants or SNPs since these HTALs were too few and limited to imprinted and CNV regions.

[BaseQTL - Vigorito et al. Nat Comp Sc 2021](https://pubmed.ncbi.nlm.nih.gov/34993494/) derives eQTL without genotype information. Specifically, they infer haplotypes from the reference genome panel, by estimating phase using the TreCASE method, and corrects reference mapping bias by applying modified WASP.

[WASP - Geijn et al. Nat Meth 2015](https://pubmed.ncbi.nlm.nih.gov/26366987/) integrates ASE and total read count for QTL inference. Performs combined haplotype test (CHT) and eliminates reference bias by discarding mis-mapped reads.

[RASQUAL - Kumasaka et al. Nat Genet 2016](https://pubmed.ncbi.nlm.nih.gov/26656845/) QTL inference by NB distribution of total reads and beta-binomial distribution of ASE.

[Shared and distinct molecular effects of regulatory genetic variants provide insight into mechanisms of distal enhancer-promoter communication - Helen Ray-Jones et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.08.04.551251v2) Uses CHi-C data to identify the contact QTL. Uses two different approaches: 1) Modifies BaseQTL by adapting it to both ATAC-seq and CHi-C contacts, but finds only 14 contact QTLs. 2) Adapts another Bayesian method GUESS, to identify 614 trimodal QTLs - associated with both gene expression, ATAC-seq, and CHi-C contacts. Overall, these combined 627 contact QTLs are then overlapped with REMAP ChIP-seq database for their enrichment with TF binding, tested with the Enformer method for their putative TF binding, and are also benchmarked with reference GWAS studies.  

[Multi-omics analysis in primary T cells elucidates mechanisms behind disease-associated genetic loci - Chenfu Shi et al. medRxiv 2023](https://www.medrxiv.org/content/10.1101/2023.07.19.23292550v1) Discusses all sorts of QTLs like eQTLs, caQTLs (ATAC-seq QTLs), insulation score QTLs, loopQTLs (QTLs affecting chromatin contacts), allelic imbalance of caQTLs and allelic imbalance of Hi-C reads (ASLoopQTLs). The results and examples utilize all of them, specifically caQTLs more than IQTLs. The ASLoopQTLs mostly show effects in a similar direction as the loopQTLs. Only 5% of IQTLs are ASLoopQTLs, suggesting the filtering was done during Hi-C mapping (a combination of BWA-MEM, SNP Phasing, and allele-specific HiCPro mapping). Mentioned that sequencing depth is a big issue for not getting a high number of ASLoopQTLs.
Both examples (including ORMDL3 Asthma locus) prioritize caQTLs over loopQTLs. In fact, the asthma locus mentions the CTCF binding SNP rs12936231, which is a caQTL. The utility of loop QTL is not evident. Did not find a specific analysis focused on loopQTLs or any motif analysis.

[DeepWAS: Multivariate genotype-phenotype associations by directly integrating regulatory information using deep learning - Arloth et al. PLoS comp biol 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007616) Derives GWAS SNPs by using regulatory annotations a prior. Uses DEEPSEA to first define the regulatory annotation score and filter the SNPs with score exceeding a certain threshold. These filtered SNPs are then subjected to a LASSO regression to call the significant GWAS entries.

## Genotyping

[Meta Imputation - Yu et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35508176/) - Integrates multiple genotype imputation output. Uses weighted average of meta analysis.

## Colocalization / Fine-mapping

[FINEMAP: efficient variable selection using summary data from genome-wide association studies - Benner et al. Bioinformatics 2016](https://academic.oup.com/bioinformatics/article/32/10/1493/1743040) Presents a shotgun stochastic search approach to identify the causal variants much quicker than CAVIAR, CAVIARBF, and PAINTOR. Applies SSS on top of Bayesian algorithm to compute the PIPs and credible sets.

[SusieRSS - Fine-mapping from summary data with the “Sum of Single Effects” model Zou et al. PLoS Genetics 2022](https://pubmed.ncbi.nlm.nih.gov/35853082/) Implements the sum of single effects ([SUSiE](https://academic.oup.com/jrsssb/article/82/5/1273/7056114)) model for fine mapping, and describes the differences of fine-mapping techniques when individual level genotype data are available, and when only the summary statistics are available (more common - SusieRSS approach - regression with summary statistics). Defines the common framework of fine-mapping using summary statistics, as employed by CAVIAR, FINEMAP, and SUSIE.

[Duijvoden et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.01.26.525702v1) Integrates GWAS with chromatin annotations + PCHiC and also performs annotation based fine mapping to prioritize GWAS SNPs. Applies on hypertension and blood pressure GWAS SNPs.

[EMS - Leveraging supervised learning for functionally informed fine-mapping of cis-eQTLs identifies an additional 20,913 putative causal eQTLs - Wang et al. Nat Comm 2021](https://www.nature.com/articles/s41467-021-23134-8) Work from David Kelley, Hillary Finucane etc. Presents EMS (expression modifier score) to predict fine-mapped causal variants. Trains data using fine-mapped variants derived by SUSIE + FINEMAP, using 49 tissues GTEX v8 data. Then uses annotation features like TSS distance, tissue and non-tissue specific binary annotations, DL features (Basenji scores), and trains a random forest classifier. Feature importance scores mention that Basenji scores and TSS distance are informative features. Using these EMS scores as prior, then they define a functional annotation based fine-mapping (PIP) across 95 traits.  ** Note: check Enformer performance. See the detailed feature list. Use motif binding information. 

[INTERACT - Deep learning predicts DNA methylation regulatory variants in the human brain and elucidates the genetics of psychiatric disorders - Zhou et al. PNAS 2022](https://www.pnas.org/doi/10.1073/pnas.2206069119) Presents a transformer based learning model to predict the changes in DNA methylation level from variants (mQTLs). Trains the data on SUSIE-derived fine-mapped mQTLs.

[BEATRICE: Bayesian Fine-mapping from Summary Data using Deep Variational Inference - Ghoshal et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.03.24.534116v1) In contrast to the sampling methods (Bayesian approaches using MCMC etc.) here the posterior distribution of causal variants given the GWAS summary statistics is modeled by a concrete distribution, whose parameters are estimated by a deep neural network. Such NN formulation helps to use computatioally efficient gradient-based optimization to minimize the KL divergence between the proposal binary concrete distribution and its posterior distribution of the causal variants.

[A more accurate method for colocalisation analysis allowing for multiple causal variants - Wallace, PLoS Genetics 2021](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1009440) Presents coloc-susie, uses SUSIE to fine map and decompose the input summary statistics (of 2 traits) into multiple independent signals (and fine-mapped variants), runs coloc on these pairs of signals, to identify the colocalized variants. Free from the assumption of single colocalized variant per locus. When SUSIE does not return any credible set, recommends to use the single variant version of coloc.


## QTL / SNP enrichment analysis

[Torres et al. AJHG 2021](https://pubmed.ncbi.nlm.nih.gov/33186544/) Tissue-of-action (TOA) scores of T2D GWAS, using fine-mapped variants, independent fine-mapped GWAS loci, reference coding annotations. A weighted sum of annotations for all fine-mapped SNPs are used for tissue-specific enrichment computation.

[ieQTL and isQTL - Kim-Hellmuth et al. Science 2020](https://pubmed.ncbi.nlm.nih.gov/32913075/) As a part of GTEx v8, they devised cell type specific enriched eQTLs and sQTLs, and showed that they correspond to better tissue specificity and colocalization with GWAS. These cell-specific enriched eQTLs are obtained by linear regression between genotypes and cell-specific gene expression values.

[DECAF - Kalita et al. Genome Biology 2022](https://pubmed.ncbi.nlm.nih.gov/35804456/) estimates cell-fraction enriched eQTLs - variants whose effect on gene expression vary across bulk samples according to the cell fraction. Performs linear regression using cell fraction and gene expression.

[Dynamic eQTL - Strober et al. Science 2019](https://pubmed.ncbi.nlm.nih.gov/31249060/) presents dynamic eQTLs significant in different time points of the cell differentiation trajectory.

[eQTLEnrich - Gamazon et al. Nat Genet 2018](https://pubmed.ncbi.nlm.nih.gov/29955180/) Tests wheteher eQTLs for a given tissue are enriched for a specific trait. Uses best eQTL per gene concept, and permutation based GWAS-eQTL enrichment method.

[GoShifter - Trynka et al. AJHG 2015](https://pubmed.ncbi.nlm.nih.gov/26140449/) Enrichment of eQTLs for specific regulatory annotations using permutation based controls.

[MESC - Yao et al. Nat Genet 2021](https://pubmed.ncbi.nlm.nih.gov/32424349/) Quantifying heritability by the effect of cis gene expression. Uses regression.

[SNP-SELEX - Yan et al. Nature 2021](https://pubmed.ncbi.nlm.nih.gov/33505025/) Presents experimental method + enrichment of TF binding SNPs on a set of regulatory sequences. Tests the method on T2D GWAS SNPs.

[Deep learning predicts the impact of regulatory variants on cell-type-specific enhancers in the brain - Zheng et al. Bioinformatics Advances 2023](https://academic.oup.com/bioinformaticsadvances/article/3/1/vbad002/6986158) Predicts cell specific enhancers by first using DeepSea framework followed by fine-tuning using RESNET. Then uses GRAD-CAM framework to obtain nucleotide importance score profiles. Augments with 6-mer based motif scoring and also employs TFmotifDisco to get the nucleotide importance scores. These scores are used to characterize the cell specific enhancers, and then fine-mapped GWAS SNPs are overlapped with them to predict the putative functional and cell-specific variants.

[Redefining tissue specificity of genetic regulation of gene expression in the presence of allelic heterogeneity - Arvanitis et al. AJHG 2022](https://doi.org/10.1016/j.ajhg.2022.01.002) CAFEH method. Uses tissue-specificity and allelic heterogeneity to call eQTLs. Tissue specific colocalization is computed by COLOC and eCAVIAR (2 variants per locus) to identify the causal variants shared between two tissues. In addition, variants shared between GTEx and other datasets are also derived by colocalization - pairwise colocalization analysis for all genes. In addition, matrixQTL was used with the cell decomposition (CIBERSORT) and other covariates as input. Finally, eGenes without colocalization between tissues / datasets were prioritized.

[GARFIELD classifies disease-relevant genomic features through integration of functional annotations with association signals - Lotchkova et al. Nat Gen 2019](https://www.nature.com/articles/s41588-018-0322-6) Peforms GWAS enrichment analysis of regulatory regions. Uses p-values of SNPs, LD matrix, regulatory annotations, and SNP distance from TSS. First performs LD-based pruning of SNPs (r2 > 0.1), then does LD tagging (R2 > 0.8) annotation overlap (basically assigning a variant an annotation if the variant itself or any variant in LD with it overlaps with the annotation), and then fits a logistic regression model between the TSS distance (Y and the variants (and LD variants).

[Partitioning heritability by functional annotation using genome-wide association summary statistics - S-LDSC - Finucane et al. Nat Gen 2015](https://www.nature.com/articles/ng.3404) Proposes S-LDSC method, to identify the functional annotation (category) of SNPs enriched for disease heritability. The idea is that for polygenic traits, a heritability information (chi-sq statistic) of a SNP is also accumulated by its tagged (LD) SNPs.

[Leveraging molecular quantitative trait loci to understand the genetic architecture of diseases and complex traits - Hormozdiari et al. Nat Gen 2018](https://www.nature.com/articles/s41588-018-0148-2) Constructs a fine-mapped set of eQTLs, hQTLs and splicing QTLs and show their disease-specific heritability enrichment using s-LDSC.



## QTL - Polygenic Risk Scores (PRS)

[Smail et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35588732/) utilizes rare variants and outlier genes (having Z score expression > certain threshold) to characterize the phenotypic effect of these rare variants.

[TL-PRS: Zhao et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/36240765/) Constructing cross-population polgenic risk scores using transfer learning.

[VIPRS - Zabad et al. AJHG 2023](https://pubmed.ncbi.nlm.nih.gov/37030289/) Approximates Bayesian computation of PRS by replacing MCMC with variational inference (VI), a deterministic class of algorithms replacing the posterior inference by an optimization problem (applied to LMM, Fine-mapping, association, enrichment).

[REGLE - Unsupervised representation learning improves genomic discovery for lung function and respiratory disease prediction - Yun et al. medRxiv 2023](https://www.medrxiv.org/content/10.1101/2023.04.28.23289285v1) From Google Research. Proposes low dimensional representation learning of high-dimensional clinical data (HDCD) and utilizes these low-dimensional embeddings to compute PRS. Applies on lung and respiratory data.

## QTL - TWAS

[METRO - Li et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35334221/) TWAS using multi-ancestry (population) data - different allele frequencies and LD matrix.

[Probabilistic integration of transcriptome-wide association studies and colocalization analysis identifies key molecular pathways of complex
traits - INTACT - Okamoto et al. AJHG 2023](https://pubmed.ncbi.nlm.nih.gov/36608684/) - Integration of TWAS and colocalization to identify causal genes. Posterior probability of a gene being causal is approximated as : prior probability of colocalization * bayes factor (BF) from TWAS * prior of TWAS.

[Genetic determinants of chromatin reveal prostate cancer risk mediated by context-dependent gene regulation - Baca et al.Nat Gen 2022](https://www.nature.com/articles/s41588-022-01168-y) Presents CWAS, analogous to TWAS, where imputed ChIP-seq data together with the phasing haplotype information is used to infer the ChIP-seq peak - trait association.

[GCSC - Leveraging gene co-regulation to identify gene sets enriched for disease heritability - Siewert-Rocks et al. AJHG 2022](https://doi.org/10.1016/j.ajhg.2022.01.005) Uses TWAS to first get the gene score for the trait (TWAS chi-sq statistics) and then detemines the gene co-regulation score (caused by shared eQTLs or eQTLs in LD). GCSC defines a gene set with higher disease heritability if the genes with high co-regulation to the gene set have higher TWAS chi-sq statistics than the genes with low co-regulation to the gene set.


## Identifying disease-risk / causal variants (and) target genes

[3DFAACTS-SNP: Liu et al. Epigenomics and Chromatin 2022](https://epigeneticsandchromatin.biomedcentral.com/articles/10.1186/s13072-022-00456-5) proposed T1D causal variant identification pipeline: 1) Bayesian fine mapping, 2) Overlap with ATAC-seq peaks (open chromatin region), 3) HiC ineractions, 4) FOXP3 binding sites.

[MA-FOCUS - Lu et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35931050/) - Fine mapping and TWAS using multiple ancestry information. Integrates GWAS, eQTL and LD information, and assumes that causal genes are shared across ancestry.

[PALM: a powerful and adaptive latent model for prioritizing risk variants with functional annotations - Yu et al. Bioinformatics 2023](https://pubmed.ncbi.nlm.nih.gov/36744920/) Uses functional annotations to prioritize GWAS variants.
Supports multiple functional annotations. Computes gradient boosting and tree based likelihood to prioritize the GWAS SNPs.

[MOVE - Allesoe et al. Nat Biotech 2023](https://pubmed.ncbi.nlm.nih.gov/36593394/) Defines MOVE - Multi-omics variational autoencoder including data from multiple omics from 789 sample cohort (vertical integration) and applies VAE, and defines the association between T2D with the latent space features. Significance is computed by t-test, and by feature purturbation (0/1) technique.

[Hierarchical Poisson model - He et al. Transl. Psychiatry 2022](https://pubmed.ncbi.nlm.nih.gov/35436980/) derives allele-specific QTLs and applies on AD. It employs hierarchical poisson model by prioritizing the heterozygous SNPs. Does not consider the total counts.

[stratAS - Shetty et al. AJHG 2021](https://pubmed.ncbi.nlm.nih.gov/34699744/) Applies allele specific QTL inference from ChIP-seq (chromatinQTL) on prCa data. Uses haplotype-based beta binomial model for the allele specific read counts, and identify the causal variants. Tests individual SNPs for each peak (within 100 Kb of peak center). It was proposed in [Gusev et al. bioRxiv 2019](https://www.biorxiv.org/content/10.1101/631150v1) and is also used in [Grishin et al. Nat Genet 2022](https://pubmed.ncbi.nlm.nih.gov/35697866/). 

[Jeong et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.03.29.534582v1) employed colocalization between GWAS and transcription factor binding QTL (bQTL), and employed motif analysis to prioritize the causal variants.

[e-MAGMA: Gerring et al. Bioinformatics 2021](https://pubmed.ncbi.nlm.nih.gov/33624746/) assigns SNPs to genes by employing tissue specific eQTL statistics from GTEx data, to identify disease-risk genes. Extends MAGMA which assigns SNPs to genes by proximity and infers meta p-values per gene. Different from H-MAGMA (https://pubmed.ncbi.nlm.nih.gov/32152537/) which uses Hi-C interactions to assign SNPs to the looped genes.

[S2G - Dey et al. Cell Genomics 2022](https://pubmed.ncbi.nlm.nih.gov/35873673/) Reviews 11 SNP to gene (S2G) linking strategies and tests the corresponding annotations using S-LDSC on 11 autoimmune traits.

## Structural variants

[Paczkowska et al. Nat Comm 2020](https://pubmed.ncbi.nlm.nih.gov/32024846/) Activepathways: integrative pathway analysis from SNPs, WGS, CNVs, and SVs. Requires 2 lists: 1) p-values related to different experiments and studies (and information of those studies, like DEG, etc.) 2) Gene sets related to pathways (downloaded from GO or Reactome, etc.) Performs a combined p-value analysis to return the most significant genes. Developed by PCAWG consortium.




