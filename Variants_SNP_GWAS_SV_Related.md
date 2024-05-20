# SNP / QTL / GWAS and functional variants

[GWAS tutorial - Github](https://github.com/Cloufield/GWASTutorial)

[GWAS tutorial](https://pbreheny.github.io/adv-gwas-tutorial/quality_control.html)

## Regulatory regions, motifs

[IMPACT: Genomic Annotation of Cell-State-Specific Regulatory Elements Inferred from the Epigenome of Bound Transcription Factors - Amariuta et al. AJHG 2019](https://doi.org/10.1016/j.ajhg.2019.03.012) 

    - IMPACT (inference and modeling of phenotype related active transcription) method. 
    - Uses histone marks, promoter and enhancer marks, for different cell types.
    - Two step approach to infer cell-state specific regulatory elements.
        - 1. For a candidate TF, first scans the TF binding motifs.
        - 2. Then learns the TF binding motifs for specific cell types, by using the cell-type-specific regulatory elements as signatures.
    - Uses a elastic net logistic regression model on 503 cell-type specific epigenomic features (open chromatin, histone marks), and 12 sequence features 
        - to identify the cell-type-specific active regulatory elements with TF binding motifs.

[Improving the trans-ancestry portability of polygenic risk scores by prioritizing variants in predicted cell-type-specific regulatory elements - Amariuta et al. Nat Gen 2020](https://www.nature.com/articles/s41588-020-00740-8) 

    - Uses IMPACT to predict TF binding motifs across 245 cell types,
    - Particularly to prioritize TF binding motifs and regulatory variants across multi-ancestry settings.
    - **** To Do: Can be further augmented by DeepLIFT related approaches - prioritizing motif scores using BPNet etc. ** 


## QTL / ASE / GWAS inference

[WASP: allele-specific software for robust molecular quantitative trait locus discovery - Geijn et al. Nat Meth 2015](https://pubmed.ncbi.nlm.nih.gov/26366987/)

    - Integrates ASE and total read count for QTL inference. 
        - Eliminates reference mapping bias by discarding mis-mapped reads, such that both alleles do not map to the same region. 
    - Performs combined haplotype test (CHT) for cis-QTL doscovery
        - 2 components: 1) total read depth in the target region, 2) allelic imbalance at phased heterozygous SNP
    - CHT is a combined likelihood ratio test from two models
        - Read depth is modeled by NB; allelic imbalance is modeled by beta-binomial distribution
        - Also accounts for genotyoping errors, by approximating allelic imbalance as a mixture of 2 beta-binomials.
    - Compares between 3 apporoaches: 
        - 1) mapping to genome using N masked SNP, 
        - 2) mapping to a personalized genome, 
        - 3) WASP, where the read mapping for both the alleles of a SNP is checked to overlap the same position.

[RASQUAL - Kumasaka et al. Nat Genet 2016](https://pubmed.ncbi.nlm.nih.gov/26656845/) 

    - QTL inference by NB distribution of total reads and beta-binomial distribution of ASE.
    - Applicable for 1D QTL (ATAC-QTL and ChIP-QTL).

[Fast and efficient QTL mapper for thousands of molecular phenotypes - fastQTL - Ongen et al. Bioinformatics 2016](https://pubmed.ncbi.nlm.nih.gov/26708335/) 

    - QTL inference by population model and efficient permutation. 
    - Default model in GTEx. Also implemented in the TensorQTL framework (including conditional eQTL analysis). 
    - Uses permutation and an adaptive permutation scheme. 
    - Approximate p-values at any significance levels without requiring the full set of permutations is performed by an approximate beta distribution 
        - (because order statistics of iid random variables are beta distributed). 
    - The second step uses FDR not from BH correction but from ST method (Storey and Tibshirani) to report higher number of significant entries, 
        - corresponding to testing thousands of molecular phenotypes (here gene expression for thousands of genes) genome-wide.

[Common DNA sequence variation influences 3-dimensional conformation of the human genome - Gorkin et al. Genome Biology 2019](https://pubmed.ncbi.nlm.nih.gov/31779666/) 

    - First interaction-QTL paper. 
    - Applies HiC data on LCL to derive interaction QTLs, but does not extensively compare with conventional eQTLs. 
    - Rather, it focuses on Hi-C-specific FIRE-QTLs, etc.

[Subtle changes in chromatin loop contact propensity are associated with differential gene regulation and expression - Greenwald et al. Nature Comm 2019](https://www.nature.com/articles/s41467-019-08940-5) 

    - Concept of HTAL - Haplotype associated loops.
    - Generated phased Hi-C data from induced pluripotent stem cells (iPSCs) and iPSC-derived cardiomyocytes of seven individuals 
        - to derive 114 haplotype-associated chromatin loops (HTALs) primarily driven by imprinting and/or CNVs but not for eQTLs. 
    - Subtle changes of these HTALs were shown to impact gene expression and H3K27ac levels, 
    - Did not identify specific regulatory variants or SNPs since these HTALs were too few and limited to imprinted and CNV regions.

[The GTEx Consortium atlas of genetic regulatory effects across human tissues - GTEx v8 release - Science 2020](https://pubmed.ncbi.nlm.nih.gov/32913098/) 

    - GTEx v8 release. Specifically check the supplementary material for the details of QTL derivation.

[A vast resource of allelic expression data spanning human tissues - GTEx v8 - Castel et al. Genome Biology 2020](https://pubmed.ncbi.nlm.nih.gov/32912332/) 

    - Repository of haplotype-specific expression for GTEx v8 tissues. 
    - Allelic log fold change count is used for comparing SNP and haplotype-level ASE data. 
    - WASP filtering is recommended for ASE inference. 
    - Also developed an extension of their tool phASER, namely phASER-POP 
        - which models population-scale haplotype level ASE data and calculate effect size for regulatory variants.

[Allele-Specific QTL Fine-Mapping with PLASMA - Wang et al. AJHG 2020](https://pubmed.ncbi.nlm.nih.gov/32004450/) 

    - uses both genotype and AS statistics to infer the significant SNPs.
    - Total expression (y) is modeled by allelic dosage (x) while the allelic imbalance (w) is determined by phasing (v). 
        - Imbalance has both magnitude as well as sign (depending on the phase)
    - Also performs fine mapping by using a genotype-LD matrix, and returns a credible causal set using shotgun stochastic search (SSS). 
    - Compares with fine-mapping approaches CAVIAR, AS-Meta, and RASQUAL 
        - (by converting the chi-sq statistics of RASQUAL to z-scores and putting them as input to fine-map approaches).

[DeepWAS: Multivariate genotype-phenotype associations by directly integrating regulatory information using deep learning - Arloth et al. PLoS comp biol 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007616) 

    - Derives GWAS SNPs by using regulatory annotations a prior. 
    - Uses DEEPSEA to first define the regulatory annotation score and filter the SNPs with score exceeding a certain threshold. 
    - These filtered SNPs are then subjected to a LASSO regression to call the significant GWAS entries.

[ASEP: Gene-based detection of allele-specific expression across individuals in a population by RNA sequencing - Fan et al. PLOS Genetics 2020](https://pubmed.ncbi.nlm.nih.gov/32392242/) 

    - Uses a mixture model to estimate ASE across individuals 
    - Also computes differential ASE between conditions among groups of individuals.

[Detection of quantitative trait loci from RNA-seq data with or without genotypes using BaseQTL - Vigorito et al. Nat Computational Science 2021](https://pubmed.ncbi.nlm.nih.gov/34993494/) 

    - BaseQTL for ASE analysis, using Bayesian approach.
    - Initial approach: Uses TRecASE with observed genotypes and fixed phasing. 
        - Models RNA-seq counts using NB distribution.
        - ASE is modeled by using haplotype specific reads (phasing)
    - Extension: Phasing as latent variable.
        - Infer haplotypes from the reference genome panel (known genotype model)
    - Extension: unknown genotype model.
    - Corrects reference mapping bias by applying modified WASP.

[A scalable unified framework of total and allele-specific counts for cis-QTL, fine-mapping, and prediction - MixQTL - Liang et al. Nat Comm 2021](https://pubmed.ncbi.nlm.nih.gov/33658504/) 

    - Models ASE using allelic fold change. 
    - The total read count and haplotype-specific read counts are separately modeled using linear regression, to provide an approximate fast QTL inference. 
    - Validated using GTEx v8 data. Implemented in tensorQTL. 
    - Appropriate for large sample sizes, but for smaller sample sizes (~100), RASQUAL is better.

[eQTL mapping using allele-specific count data is computationally feasible, powerful, and provides individual-specific estimates of genetic effects - geoP - Zhabotynsky et al. PLOS Genetics 2022](https://pubmed.ncbi.nlm.nih.gov/35286297/) 

    - Efficient method to compute permutation-based p-values for ASE, used in TreCASE and RASQUAL. 
    - Detects 20%-100% more eGenes, but whether these are reliable is a question.
    
[Shared and distinct molecular effects of regulatory genetic variants provide insight into mechanisms of distal enhancer-promoter communication - Helen Ray-Jones et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.08.04.551251v2) 

    - Uses CHi-C data to identify the contact QTL. 
    - Uses two different approaches: 
        - 1) Modifies BaseQTL by adapting it to both ATAC-seq and CHi-C contacts, but finds only 14 contact QTLs. 
        - 2) Adapts another Bayesian method GUESS, to identify 614 trimodal QTLs 
            - associated with both gene expression, ATAC-seq, and CHi-C contacts.             
    - Overall, these combined 627 contact QTLs are then overlapped with REMAP ChIP-seq database for their enrichment with TF binding, 
    - tested with the Enformer method for their putative TF binding, and are also benchmarked with reference GWAS studies.  

[Multi-omics analysis in primary T cells elucidates mechanisms behind disease-associated genetic loci - Chenfu Shi et al. medRxiv 2023](https://www.medrxiv.org/content/10.1101/2023.07.19.23292550v1) 

    - Discusses all sorts of QTLs like eQTLs, caQTLs (ATAC-seq QTLs), 
        - Also, insulation score QTLs, loopQTLs (QTLs affecting chromatin contacts), allelic imbalance of caQTLs and allelic imbalance of Hi-C reads (ASLoopQTLs). 
    - The results and examples utilize all of them, specifically caQTLs more than IQTLs. 
    - The ASLoopQTLs mostly show effects in a similar direction as the loopQTLs. 
    - Only 5% of IQTLs are ASLoopQTLs, suggesting the filtering was done during Hi-C mapping 
        - (a combination of BWA-MEM, SNP Phasing, and allele-specific HiCPro mapping). 
    - Mentioned that sequencing depth is a big issue for not getting a high number of ASLoopQTLs.
    - Both examples (including ORMDL3 Asthma locus) prioritize caQTLs over loopQTLs. 
    - In fact, the asthma locus mentions the CTCF binding SNP rs12936231, which is a caQTL. 
    - The utility of loop QTL is not evident. 
    - Did not find a specific analysis focused on loopQTLs or any motif analysis.

## Genotyping

[Meta-imputation: An efficient method to combine genotype data after imputation with multiple reference panels - Yu et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35508176/) 

    - Integrates multiple genotype imputation output. Uses weighted average of meta analysis.

## Colocalization / Fine-mapping

[Identifying causal variants at loci with multiple signals of association - Hormozdiari et al. - Genetics 2014](https://pubmed.ncbi.nlm.nih.gov/25104515/)

    - CAVIAR (CAusal Variants Identification in Associated Regions)
    - Input: GWAS summary statistics (association p-values or effect sizes)
    - Likelihood computation: Summary Statistics + LD structure 
    - Model: CAVIAR assumes that the association statistics of SNPs follow a multivariate normal distribution, 
        - the mean vector and covariance matrix are determined by the effect sizes and LD structure of the GWAS variants.
        - Prior Specification: sparse prior - 
            - most SNPs are assumed to have null effects, and only a small subset of SNPs are considered to be causal.
        - Likelihood: evaluating the probability density function of the observed summary statistics under the assumed model. 
        - integrating over all possible configurations of causal variants and their effect sizes, weighted by their prior probabilities.
        - Posterior: Bayes' theorem
            - used to rank SNPs according to their likelihood of being causal.

[Bayesian test for colocalisation between pairs of genetic association studies using summary statistics - Giambartolomei et al. PLoS Genetics 2014](https://pubmed.ncbi.nlm.nih.gov/24830394/) 

    - Decribes the various forms of posteriors from h_1 to h_5 - no colocalization vs colocalization
    - Bayes factors to decide the posterior of colocalization over all other events 
        - Wakefield’s approximation to decide approximate bayes factor. 
    - Assumes single causal variant per locus. 
    - Note: although not mentioned, one can implement stepwise conditioning by 
        - either using reference LD matrix, 
        - or using external packages like CoJo (Yang et al. Nat Genet 2012)

[FINEMAP: efficient variable selection using summary data from genome-wide association studies - Benner et al. Bioinformatics 2016](https://academic.oup.com/bioinformatics/article/32/10/1493/1743040) 

    - Presents a shotgun stochastic search approach to identify the causal variants much quicker than CAVIAR, CAVIARBF, and PAINTOR. 
    - Applies SSS on top of Bayesian algorithm to compute the PIPs and credible sets.

[Colocalization of GWAS and eQTL Signals Detects Target Genes - Hormozdiari et al. AJHG 2016](https://pubmed.ncbi.nlm.nih.gov/27866706/) 

    - eCAVIAR method. For a given variant, multiply the fine mapping posterior for two traits to decide if the variant is colocalized.  
    - Supports multiple causal variants per locus. 
    - Drawbacks: 
        - assumes trait independence, 
        - traits are from the same population, 
        - use of reference LD matrix, 
        - and effect sizes are aligned to the effect matrix. 
    - Requires effect size and allele information of the SNPs 

[SUSiE - Wang et al. 2020 - Journal of Royal Statistics](https://academic.oup.com/jrsssb/article/82/5/1273/7056114)

    - SUSiE model for fine mapping. 
    - Bayesian variable selection in regression (BVSR)
    - New formulation: Sum of single effects (SUSIE)
    - Single effect regression (SER) model - in a multiple regression problem, exactly one of the variables have nonzero regression coefficients.
    - Iterative Bayesian stepwise selection (IBSS) - Bayesian analogue of traditional stepwise selection methods.
    - Simple model-fitting algorithm - iterates through the single-effect vectors l = 1, . . ., L, 
        - at each iteration fitting bl while keeping the other single-effect vectors fixed. 
        - By construction, each step thus involves fitting an SER (single effect regression). 
        - IBSS can be understood as computing an approximate posterior distribution p(b1, . . ., bL | X, y, σ2), 
            - and that the algorithm iteratively optimizes an objective function known as the “evidence lower bound” (ELBO).

[Eliciting priors and relaxing the single causal variant assumption in colocalisation analyses - Wallace. PLoS Genetics 2020](https://pubmed.ncbi.nlm.nih.gov/32310995/) 

    - Discusses about prior probabilities of coloc package - must check. 
    - Also supports multiple causal variants per locus by masking all SNPs in LD with the lead SNP.

[A more accurate method for colocalisation analysis allowing for multiple causal variants - Wallace, PLoS Genetics 2021](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1009440) 

    - Presents *coloc-susie*, 
    - uses SUSIE to fine map and decompose the input summary statistics (of 2 traits) into multiple independent signals (and fine-mapped variants), 
    - runs coloc on these pairs of signals, to identify the colocalized variants. 
    - Free from the assumption of single colocalized variant per locus. 
    - When SUSIE does not return any credible set, recommends to use the single variant version of coloc. 
    - Uses SUSIE defined fine-mapped variants (list of L1 and L2 number of variants for two traits) and performs colocalization L1 X L2 times 
        - to infer whether a pair of variants are colocalized. 
    - Question: Seems maximum 2 variants are supported for colocalization - what if > 2 causal variants? 

[SusieRSS - Fine-mapping from summary data with the “Sum of Single Effects” model - Zou et al. PLoS Genetics 2022](https://pubmed.ncbi.nlm.nih.gov/35853082/) 

    - Describes the differences of fine-mapping techniques when individual level genotype data are available, and when only the summary statistics are available (more common). 
    - SusieRSS approach - regression with summary statistics. 
    - Defines the common framework of fine-mapping using summary statistics, as employed by CAVIAR, FINEMAP, and SUSIE.

[BEATRICE: Bayesian Fine-mapping from Summary Data using Deep Variational Inference - Ghoshal et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.03.24.534116v1) 
    
    - Here the posterior distribution of causal variants given the GWAS summary statistics is modeled by a concrete distribution 
        - whose parameters are estimated by a deep neural network. 
        - Existing methods (sampling methods) use Bayesian approaches using MCMC etc. for parameter estimation.
    - Such NN formulation helps to use computatioally efficient gradient-based optimization 
        - to minimize the KL divergence between the proposal binary concrete distribution and its posterior distribution of the causal variants.

## QTL / SNP enrichment analysis

[Disentangling the Effects of Colocalizing Genomic Annotations to Functionally Prioritize Non-coding Variants within Complex-Trait Loci - GoShifter - Trynka et al. AJHG 2015](https://pubmed.ncbi.nlm.nih.gov/26140449/) 

    - Enrichment of eQTLs for specific regulatory annotations using permutation based controls. 
    - Now a days mostly replaced by S-LDSC method.

[Partitioning heritability by functional annotation using genome-wide association summary statistics - S-LDSC - Finucane et al. Nat Gen 2015](https://www.nature.com/articles/ng.3404) 

    - Proposes S-LDSC method, to identify the functional annotation (category) of SNPs enriched for disease heritability. 
    - The idea is that for polygenic traits, a heritability information (chi-sq statistic) of a SNP is also accumulated by its tagged (LD) SNPs.

[Using an atlas of gene regulation across 44 human tissues to inform complex disease- and trait-associated variation - eQTLEnrich - Gamazon et al. Nat Genet 2018](https://pubmed.ncbi.nlm.nih.gov/29955180/) 

    - Tests wheteher eQTLs for a given tissue are enriched for a specific trait. 
    - Uses best eQTL per gene concept, and permutation based GWAS-eQTL enrichment method.

[Leveraging molecular quantitative trait loci to understand the genetic architecture of diseases and complex traits - Hormozdiari et al. Nat Gen 2018](https://www.nature.com/articles/s41588-018-0148-2) 

    - Constructs a fine-mapped set of eQTLs, hQTLs and splicing QTLs and show their disease-specific heritability enrichment using s-LDSC.

[Dynamic genetic regulation of gene expression during cellular differentiation - Dynamic eQTL - Strober et al. Science 2019](https://pubmed.ncbi.nlm.nih.gov/31249060/) 

    - presents dynamic eQTLs significant in different time points of the cell differentiation trajectory.

[GARFIELD classifies disease-relevant genomic features through integration of functional annotations with association signals - Lotchkova et al. Nat Gen 2019](https://www.nature.com/articles/s41588-018-0322-6) 

    - Peforms GWAS enrichment analysis of regulatory regions. 
    - Uses p-values of SNPs, LD matrix, regulatory annotations, and SNP distance from TSS. 
    - First performs LD-based pruning of SNPs (r2 > 0.1), then does LD tagging (R2 > 0.8) 
    - then does annotation overlap (basically assigning a variant an annotation if the variant itself or any variant in LD with it overlaps with the annotation), 
    - then fits a logistic regression model between the TSS distance (Y and the variants (and LD variants).

[Cell type-specific genetic regulation of gene expression across human tissues - ieQTL and isQTL - Kim-Hellmuth et al. Science 2020](https://pubmed.ncbi.nlm.nih.gov/32913075/) 

    - As a part of GTEx v8, they devised cell type specific enriched eQTLs and sQTLs, 
    - showed that they correspond to better tissue specificity and colocalization with GWAS. 
    - These cell-specific enriched eQTLs are obtained by linear regression between genotypes and cell-specific gene expression values.

[Quantifying genetic effects on disease mediated by assayed gene expression levels - MESC - Yao et al. Nat Genet 2020](https://pubmed.ncbi.nlm.nih.gov/32424349/) 

    - Quantifying heritability by the effect of _cis_ gene expression. Uses regression.

[Functional dynamic genetic effects on gene regulation are specific to particular cell types and environmental conditions - Findley et al. eLife 2021](https://pubmed.ncbi.nlm.nih.gov/33988505/) 

    - From the same group. 
    - GXE specific (context specific) eQTLs according to different cell-specific conditions. 
    - Also, integrates with ASE.

[Systematic analysis of binding of transcription factors to noncoding variants - SNP-SELEX - Yan et al. Nature 2021](https://pubmed.ncbi.nlm.nih.gov/33505025/) 

    - Characterizing effects of SNPs on binding TF. 
    - Presents experimental method by extending their previous method HT-SELEX which uses random DNA sequences as input. 
    - Here they provide 40 bp DNA from human genome. 
    - Enrichment of TF binding SNPs on a set of regulatory sequences is tested, and applied on T2D GWAS SNPs. 
    - Defines oligonucelotide binding score (OBS) and preferential binding score (PBS) for the SNPs with respect to TF binding 
        - corresponding SNPs are denoted as preferential binding SNPs (pbSNP). 
    - Also evaluates the PWMs by predicting differential binding of TFs, 
        - finds the pbSNPs perform better in predicting differential TF binding. 
    - DeltaSVM franework using gkm-SVM is used to quantify the variant effect size. 
    - **** Note: Supplementary is useful 
        - shows phasing of haplotypes, motif finding, allelic fold change, and haplotype estimation from Hi-C data.

[DeCAF: a novel method to identify cell-type specific regulatory variants and their role in cancer risk - Kalita et al. Genome Biology 2022](https://pubmed.ncbi.nlm.nih.gov/35804456/) 

    - estimates cell-fraction enriched eQTLs - variants whose effect on gene expression vary across bulk samples according to the cell fraction. 
    - Performs linear regression using cell fraction and gene expression.

[Redefining tissue specificity of genetic regulation of gene expression in the presence of allelic heterogeneity - Arvanitis et al. AJHG 2022](https://doi.org/10.1016/j.ajhg.2022.01.002) 

    - CAFEH method. Uses tissue-specificity and allelic heterogeneity to call eQTLs. 
    - Tissue specific colocalization is computed by COLOC and eCAVIAR (2 variants per locus) to identify the causal variants shared between two tissues. 
    - Variants shared between GTEx and other datasets are also derived by colocalization - pairwise colocalization analysis for all genes. 
    - matrixQTL was used with the cell decomposition (CIBERSORT) and other covariates as input. 
    - Finally, eGenes without colocalization between tissues / datasets were prioritized.

## QTL - Polygenic Risk Scores (PRS)

[Integration of rare expression outlier-associated variants improves polygenic risk prediction - Smail et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35588732/) 

    - Utilizes rare variants and outlier genes (having Z score expression > certain threshold) to characterize the phenotypic effect of these rare variants. 
    - *** Note: the methods section, GTEx v8 data shows how to infer the rare variants and associated statisitcs from variants, and calculate PRS.

[The construction of cross-population polygenic risk scores using transfer learning - TL-PRS: Zhao et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/36240765/) 

    - Constructing cross-population polgenic risk scores using transfer learning.

[Fast and accurate Bayesian polygenic risk modeling with variational inference - VIPRS - Zabad et al. AJHG 2023](https://pubmed.ncbi.nlm.nih.gov/37030289/) 
    
    - Approximates Bayesian computation of PRS by replacing MCMC with variational inference (VI)
        - a deterministic class of algorithms replacing the posterior inference by an optimization problem 
        - (applied to LMM, Fine-mapping, association, enrichment).

[A new method for multiancestry polygenic prediction improves performance across diverse populations - Zhang et al. Nature Genetics 2023](https://pubmed.ncbi.nlm.nih.gov/37749244/) 

    - CT-SLEB method. Estmating PRS from EUR and non-EUR populations. 
    - Requires 3 datasets: GWAS, Tuning dataset (parameter optimization) and validation.
    - Three steps: 
        - 1) CT: clump and threshold - select SNPs by 2D p-values (EUR and target populations) 
            - SNP ranking by association and clumped using LD estimates - SNPs associated with at least one population - by PLINK.
            - LDPred2: SNP effect sizes by a shrinkage estimator, combining GWAS summary statistics with a prior on effect sizes.
        - 2) EB - empirical bayes method to estimate standardized effect sizes for SNPs considering covariance matrix from all populations and assuming normal prior. 
            - SNP coefficient estimation - based on the combination of multiple GWAS summary statistics mean and variance and assuming a normal distribution.
        - 3) SL - Superlearning: tested with CNN and regression methods to estimate PRS. Benchmarked with 23andme datasets, and LDPred2 method.
    - Benchmarks PRS prediction among different methods - CT, LDPred2, PolyPred-S+, XPASS, PRS-CSx, CT-SLEB, and their weighted combinations.

## QTL - TWAS

[Integration of summary data from GWAS and eQTL studies predicts complex trait gene targets - Zhu et al. Nature Genetics 2016](https://pubmed.ncbi.nlm.nih.gov/27019110/)

    - SMR method - summary statistics based MR.
    - A genetic variant (for example, a SNP) is used as an instrumental variable to test for the causative effect of an exposure (for example, gene expression) on an outcome (for example, phenotype).
    - Discusses about causality (Causal variant -> Transcription -> Phenotype), Pleiotropy (Causal variant is independently associated with transcription and phenotype), and linkage (two causal variants are independently associated with transcription and phenotype).
    - HEIDI method - distinguishes pleiotropy from linkage. Smaller HEIDI value - higher probability of linkage.
    - Testing linakge is equivalent to testing whether there is a difference between effect size estimated using the top cis-eQTL and using any other significant SNP.

[Integrative approaches for large-scale transcriptome-wide association studies - Gusev et al. Nature Genetics 2016](https://pubmed.ncbi.nlm.nih.gov/26854917/)

    - Concept of TWAS. Imputed gene expression from GWAS summary statistics
    - Association between gene and phenotype.

[Exploring the phenotypic consequences of tissue specific gene expression variation inferred from GWAS summary statistics - Barbeira et al. Nat Comm 2018](https://pubmed.ncbi.nlm.nih.gov/29739930/) 

    - S-PrediXcan method. PrediXcan using GWAS summary statistics.
    - High concordance between PrediXcan and S-PrediXcan, with much more applicability.
    - Compared with coloc, RTC, eCAVIAR, ENLOC, and S-TWAS (summary based TWAS).

[A multi-tissue transcriptome analysis of human metabolites guides the interpretability of associations based on multi-SNP models for gene expression - Ndungu et al. AJHG 2019](https://pubmed.ncbi.nlm.nih.gov/31978332/) 

    - Benchmarking single SNP (eQTL) vs multi-SNP (TWAS or LASSO regression) models in predicting gene expression. 
        - 1. Shows that LASSO regression, although includes more than one SNP for a single locus / gene, often confounded by LD effect and random selection. 
        - 2. TWAS shows higher sensitivity in predicting causal genes, but also can point to the nearby (bystander) genes having similar effects. 
        - 3. Shows that multi-SNP model outperforms single SNP model in predicting gene expression. 
    - *To Do*: 
        - 1) Improve S-PrediXcan by deep learning, in predicting gene expression. 
        - 2) How LASSO regression and TWAS analysis can be combined to predict the marker genes? 
        - 3) Whether LASSO regression based eQTL and fine-mapping is mentioned in literature.

[A unified framework for joint-tissue transcriptome-wide association and Mendelian randomization analysis - Zhou et al. Nature Genetics 2020](https://pubmed.ncbi.nlm.nih.gov/33020666/) 

    - MR-JTI method - TWAS. 
        - 1. TWAS approach, extending PrediXcan, using multi tissue summary statistics. 
        - 2. Employs tissue specific gene expression correlation and DHS (regulatory annotation) correlation statistic for the optimization problem.

[METRO: Multi-ancestry transcriptome-wide association studies for powerful gene-trait association detection - Li et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35334221/) 

    - TWAS using multi-ancestry (population) data - different allele frequencies and LD matrix.

[Genetic determinants of chromatin reveal prostate cancer risk mediated by context-dependent gene regulation - Baca et al.Nat Gen 2022](https://www.nature.com/articles/s41588-022-01168-y) 

    - Presents CWAS, analogous to TWAS, where imputed ChIP-seq data together with the phasing haplotype is used to infer the ChIP-seq peak - trait association.

[GCSC - Leveraging gene co-regulation to identify gene sets enriched for disease heritability - Siewert-Rocks et al. AJHG 2022](https://doi.org/10.1016/j.ajhg.2022.01.005) 

    - Uses TWAS to first get the gene score for the trait (TWAS chi-sq statistics) 
    - then detemines the gene co-regulation score (caused by shared eQTLs or eQTLs in LD). 
    - GCSC defines a gene set with higher disease heritability if the genes with high co-regulation to the gene set have higher TWAS chi-sq statistics than the genes with low co-regulation to the gene set.

[Integrating transcription factor occupancy with transcriptome-wide association analysis identifies susceptibility genes in human cancers - He et al. Nat Comm 2022](https://pubmed.ncbi.nlm.nih.gov/36402776/) 

    - Proposes sTF-TWAS. Selecting subset of variants for TWAS.
    - Uses prior knowledge of TF binding sites. 
    - Performs regression on the chi-sq statistics (from GWAS summary) of SNPs (variants) with their TF binding status 
        - selects the top-K variants and uses only these variants to impute gene expression and perform TWAS.

[Probabilistic integration of transcriptome-wide association studies and colocalization analysis identifies key molecular pathways of complex
traits - INTACT - Okamoto et al. AJHG 2023](https://pubmed.ncbi.nlm.nih.gov/36608684/) 

    - Integration of TWAS and colocalization to identify causal genes. 
    - Posterior probability of a gene being causal is approximated as : prior probability of colocalization * bayes factor (BF) from TWAS * prior of TWAS.

[Modeling tissue co-regulation estimates tissue-specific contributions to disease - Amruita et al. Nature Genetics 2023](https://pubmed.ncbi.nlm.nih.gov/37580597/)

    - TCSC method. Similar to GCSC - here genes are replaced by tissues.
    - Uses TWAS gene-trait association with tissue-co-regulation score to identify causal tissues, and tissue association with a given trait.

## Identifying disease-risk / causal variants (and) target genes

[A Multi-omic Integrative Scheme Characterizes Tissues of Action at Loci Associated with Type 2 Diabetes - Torres et al. AJHG 2021](https://pubmed.ncbi.nlm.nih.gov/33186544/) 

    - Tissue-of-action (TOA) scores of T2D GWAS, to understand the relevant tissues and cell types for a given disease (here T2D). 
    - Uses tissue-specific gene expression, epigenomic maps, fine-mapped variants (and Bayesian PIPs), independent fine-mapped GWAS loci, reference coding annotations. 
    - A weighted sum of annotations for all fine-mapped SNPs are used for tissue-specific enrichment computation. 
    - Tissue-specificity of TPM nomalized gene expression was measured by expression specificity scores (ESS). 
    - Finally, TOA was computed using fine-mapped variants and using regulatory annotation information.

[Integration of genetic fine-mapping and multi-omics data reveals candidate effector genes for hypertension - Duijvoden et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.01.26.525702v1) 

    - Integrates GWAS with chromatin annotations + PCHiC 
    - Also performs annotation based fine mapping to prioritize GWAS SNPs. 
    - Applies on hypertension and blood pressure (CVD) GWAS SNPs. 
    - Uses regulatory annotations, fine-mapped GWAS SNPs (99% PIP causal set), genomic enrichment (GREAT), colocalization with GTEx eQTLs,  
        - identified target genes via inetrating capture Hi-C loops. 
    - Also uses EpiMap analysis (from Kellis group) to associate the gene expression with CREs.

[3DFAACTS-SNP: using regulatory T cell-specific epigenomics data to uncover candidate mechanisms of type 1 diabetes (T1D) risk - Liu et al. Epigenomics and Chromatin 2022](https://epigeneticsandchromatin.biomedcentral.com/articles/10.1186/s13072-022-00456-5) 

    - proposed T1D causal variant identification pipeline: 
        - 1) Bayesian fine mapping, 
        - 2) Overlap with ATAC-seq peaks (open chromatin region), 
        - 3) HiC ineractions, 
        - 4) FOXP3 binding sites.

[Multi-ancestry fine-mapping improves precision to identify causal genes in transcriptome-wide association studies - MA-FOCUS - Lu et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/35931050/) 

    - Fine mapping and TWAS using multiple ancestry information. 
    - Integrates GWAS, eQTL and LD information, and assumes that causal genes are shared across ancestry.

[PALM: a powerful and adaptive latent model for prioritizing risk variants with functional annotations - Yu et al. Bioinformatics 2023](https://pubmed.ncbi.nlm.nih.gov/36744920/) 

    - Powerful and adaptive latent model (PALM). 
    - Uses functional annotations to prioritize GWAS variants.
    - Supports multiple functional annotations, and develops functional gradient based EM algorithm. 
    - Computes gradient boosting and tree based likelihood to prioritize the GWAS SNPs. 
    - Applies on 30 GWAS data with 127 functional annotations.

[Allele-specific analysis reveals exon- and cell-type-specific regulatory effects of Alzheimer's disease-associated genetic variants - Hierarchical Poisson model - He et al. Transl. Psychiatry 2022](https://pubmed.ncbi.nlm.nih.gov/35436980/) 

    - Derives allele-specific QTLs and applies on AD. 
    - It employs hierarchical poisson model by prioritizing the heterozygous SNPs. 
        - Does not consider the total counts.

[Allelic imbalance reveals widespread germline-somatic regulatory differences and prioritizes risk loci in Renal Cell Carcinoma - Gusev et al. bioRxiv 2019](https://www.biorxiv.org/content/10.1101/631150v1) 

    - stratAS method..
    - Converts allele-specific reads to first haplotype-specific reads and then to SNP genotype specific statistics.

[Allele-specific epigenetic activity in prostate cancer and normal prostate tissue implicates prostate cancer risk mechanisms - stratAS - Shetty et al. AJHG 2021](https://pubmed.ncbi.nlm.nih.gov/34699744/) 

    - Extension of stratAS method. 
    - Applies allele specific QTL inference from ChIP-seq (chromatinQTL) on prCa data. 
    - Uses haplotype-based beta binomial model for the allele specific read counts, and identify the causal variants. 
    - Tests individual SNPs for each peak (within 100 Kb of peak center).

[Allelic imbalance of chromatin accessibility in cancer identifies candidate causal risk variants and their mechanisms - Grishin et al. Nat Genet 2022](https://pubmed.ncbi.nlm.nih.gov/35697866/) 

    - RWAS method. Regulome-wide association studies - similar to TWAS but in enhancer level.
        - 1. Using stratAS, derives Allele specific accessibility QTL (asQTL) on cancer ATAC-seq data. 
        - 2. Shows high heritability, motif enrichment, regulatory effect (SURE risk) for asQTLs. 
        - 3. Using ATAC-seq peaks, proposes RWAS (similar to TWAS) for predictive model of ATAC-seq accessibility. 
        - 4. RWAS shows higher power of detecting association to known GWAS loci compared to TWAS. 
            - Also, it identifies many loci which are not significant by GWAS. 
        - 5. Combines RWAS with CWAS (cistrome/ChIP-seq specific WAS) to identify candidate causal loci.

[Blood cell traits’ GWAS loci colocalization with variation in PU.1 genomic occupancy prioritizes causal noncoding regulatory variants - Jeong et al. Cell Genomics 2023](https://doi.org/10.1016/j.xgen.2023.100327) 

    - employed colocalization between GWAS and transcription factor binding QTL (bQTL), 
    - employed motif analysis to prioritize the causal variants. 
    - Uses DeltaSVM using gkm-SVM framework to derive the motif scores. 

[MAGMA: Generalized Gene-Set Analysis of GWAS Data](https://doi.org/10.1371/journal.pcbi.1004219)

    - Assigns SNPs to genes by proximity and infers meta p-values per gene.

[A gene co-expression network-based analysis of multiple brain tissues reveals novel genes and molecular pathways underlying major depression](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1008245)

    - Application of MAGMA to identify causal genes using GWAS.

[E-MAGMA: an eQTL-informed method to identify risk genes using genome-wide association study summary statistics: Gerring et al. Bioinformatics 2021](https://pubmed.ncbi.nlm.nih.gov/33624746/) 

    - Extension of above two methods.
    - assigns SNPs to genes by employing tissue specific eQTL statistics from GTEx data, to identify disease-risk genes. 

[A computational tool (H-MAGMA) for improved prediction of brain-disorder risk genes by incorporating brain chromatin interaction profiles - H-MAGMA](https://pubmed.ncbi.nlm.nih.gov/32152537/) 

    - Different method - uses Hi-C interactions to assign SNPs to the looped genes.

[Identifying enhancer properties associated with genetic risk for complex traits using regulome-wide association studies - Casella et al. PLoS Comp Biol 2022](https://pubmed.ncbi.nlm.nih.gov/36070311/) 

    - Proposes RWAS - regulome wide association study - to identify enhancers associated with a given disease / trait. 
    - Extends MAGMA package to identify the enhancers (whereas MAGMA finds the genes).

[SNP-to-gene linking strategies reveal contributions of enhancer-related and candidate master-regulator genes to autoimmune disease - Dey et al. Cell Genomics 2022](https://pubmed.ncbi.nlm.nih.gov/35873673/) 

    - Reviews 11 SNP to gene (S2G) linking strategies and tests the corresponding annotations using S-LDSC on 11 autoimmune traits.

[A catalog of GWAS fine-mapping efforts in autoimmune disease - Caliskan et al. AJHG 2021](https://pubmed.ncbi.nlm.nih.gov/33798443/) 

    - Integrates fine-mapping studies for various immune diseases, and defines a weighted gene prioritization score to assign the candidate risk genes.

[Genetic Control of Expression and Splicing in Developing Human Brain Informs Disease Mechanisms - Walker et al. Cell 2019](https://pubmed.ncbi.nlm.nih.gov/31626773/) 

    - Integrates eQTL, splicing QTL, GWAS, TWAS, S-LDSC, WGCNA, ATAC-seq and Hi-C 
    - prioritize the gene modules and interacting variants, for Human Cortex prenatal development.

[Integrative analyses highlight functional regulatory variants associated with neuropsychiatric diseases - Guo et al. Nature Genetics 2023](https://pubmed.ncbi.nlm.nih.gov/37857935/) 

    - Integrates SNPs from 10 Brain disorders, MPRA validated variants, epigenomic annotations (ATAC-seq peaks and cell-specific annotations) 
    - S-LDSC enrichment, TF motif information, and eQTLs, to prioritize putative causal variants. 
    - Uses eQTL-eGene links to assign to their nearest genes. 
    - Integrates multiple other databases (GO, CMAP, UK BioBank, PsychENCODE, etc.)

[Cell type and condition specific functional annotation of schizophrenia associated non-coding genetic variants - Rummel et al.bioRxiv 2023](https://pubmed.ncbi.nlm.nih.gov/37425902/) 

    - Presents SCZ and brain disorder related causal variant (specifically emVAR - expression modulating variant) identification methods using MPRA assays 
    - Shows that they overlap with causal variants.

[Massively parallel functional dissection of schizophrenia-associated noncoding genetic variants - Rummel et al. Cell 2023](https://pubmed.ncbi.nlm.nih.gov/37852259/) 

    - MVAP - Massively parallel association based variant annotation. 
    - Identifies putative causal variant for SCZ. 
    - Uses eQTL, GWAS, regulatory annotations, Hi-C, colocalization, partitioned heritability. 
    - Performs MPRA and emVAR experiment (allele-specific expression change) to identify the causal SNPs. 
    - The pipeline is a mixture of bioinformatics and wet-lab validation (MPRA + emVAR).

[Gene prioritization in GWAS loci using multimodal evidence - Schipper et al. medRxiv 2024](https://www.medrxiv.org/content/10.1101/2023.12.23.23300360v2)

    - FLAMES method for prioritizing genes from GWAS studies - S2G.
        - 1. Uses fine mapping statistics to prioritize genes in a locus.
        - 2. Also uses PoPS method, to identify genes which are GWAS enriched across a given locus.
        - 3. Combines both evidences to report the most causal gene in a locus.

## Structural variants

[Integrative pathway enrichment analysis of multivariate omics data - Paczkowska et al. Nat Comm 2020](https://pubmed.ncbi.nlm.nih.gov/32024846/) 

    - Developed by PCAWG consortium.
    - Activepathways: integrative pathway analysis from SNPs, WGS, CNVs, and SVs. 
    - Requires 2 lists: 
        - 1) p-values related to different experiments and studies (and information of those studies, like DEG, etc.) 
        - 2) Gene sets related to pathways (downloaded from GO or Reactome, etc.) 
    - Performs a combined p-value analysis to return the most significant genes. 
    
