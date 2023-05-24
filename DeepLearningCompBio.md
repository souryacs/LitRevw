# Deep Learning in Computational Biology

## Variant Calling:

[DeepVariant - Poplin et al. Nat Biotech 2018](https://www.nature.com/articles/nbt.4235): Uses samtools read pileups + known genotype calls for training. Uses CNN + Stochastic gradient descent model.

[CancerVar: Li et al. Science Advances 2022](https://pubmed.ncbi.nlm.nih.gov/35544644/): AI based cancer somatic mutation caller.

## QTL / SNP / GWAS etc.

[TL-PRS: Zhao et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/36240765/) Constructing cross-population polgenic risk scores using transfer learning.

[MetaChrom - Lai et al. PLOS Comp Biol. 2022](https://pubmed.ncbi.nlm.nih.gov/35576194/) employs transfer learning to annotate the regulatory variants and then combine with GWAS data to predict the functional variants.

[DeepCOMBI - Mieth et al. NAR Genomics Bioinformatics](https://pubmed.ncbi.nlm.nih.gov/34296082/) adopts layer-wise relevance propagation (LRP) to attribute SNP relevance scores and selection of significant SNPs in GWAS studies. Replaces conventional p-value thresholding. Extends their earlier work COMBI which uses SVM based method, to a DNN setting.  

[REGLE - Yun et al. medRxiv 2023](https://www.medrxiv.org/content/10.1101/2023.04.28.23289285v1) Representation learning using low dimensional representation of high-dimensional clinical data (HDCD) to perform GWAS on representative traits, and estimate polygenic risk scores (PRS) on individual representative traits.

## Prediction of Regulatory region using DNA sequence

[Basset - Kelley et al. Genome Research 2016](https://genome.cshlp.org/content/26/7/990.long): Predicts regulatory DNA sequences and sequence activities (chomatin accessibility). Uses DNase-seq, DHS and peaks, and applies to deep CNN. CNNs learn relevant sequence motifs and regulatory logic. Assigns GWAS variants and cell-type-scores to predict chromatin accessibility difference between alleles, and predicts causal SNPs. It also identifies sequence variants, TF motifs.

[Basenji - Kelley et al. Genome Research 2018](https://genome.cshlp.org/content/28/5/739.long): Extends Basset, which only generates peak based chromatin profiles. Predicts epigenomic and transcriptional effects using the ChIP-seq, DNAse seq, ATAC-seq. Also identifies causal variants using GWAS loci. Predicts distal regulatory interactions and finer resolution chromatin profiles. It also identifies sequence variants, TF motifs.

 **Note** : The input sequencing dataset is processed by a custom pipeline to use the multi-mapping reads and to normalize for GC bias.

[Enformer - Avsec et al. Nature Methods 2021](https://pubmed.ncbi.nlm.nih.gov/34608324/) predicts gene expression from DNA sequences using transformer model and 198 bp receptive field around TSS. Best performing model so far. It also identifies sequence variants, TF motifs, and eQTLs.

[INTERACT - Zhou et al. PNAS 2022](https://pubmed.ncbi.nlm.nih.gov/35969790/) predicts DNA methylation levels from DNA sequences, tissue specific DNA methylation data, TF motifs (validated by TOMTOM motif analysis tool) and also predicts DNA methylation QTLs (sequence variants) which are further integrated to brain GWAS studies.

[CORE-ATAC - Thibodeau et al. Plos Comp Biol 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009670) Prediction of *cis*-CREs using ATAC-seq data. Uses CNN + max pooling. Can not predict chromHMM annotations but predicts top 3 functional annotations.

[BIONIC - Forster et al. Nat Meth 2022](https://www.nature.com/articles/s41592-022-01616-x): Biological network integration. Uses GAT.
  
  **Note** : Modifies GAT to consider a priori network edge weights (Methods, eqs. 1-2). Uses 3 GAT layers, 10 attention heads per GAT encoder, each with a hidden dimension of 68, as per their hyperparameter optimization results.
  
[BindSpace - Yuan et al. Nature Methods 2019](https://pubmed.ncbi.nlm.nih.gov/31406384/) Predicts TF binding motifs from DNA sequences, using StarSpace framework (a NLP based model).

[Review paper: Karollus et al. Genome Biology 2023](https://pubmed.ncbi.nlm.nih.gov/36973806/) reviews the sequence to expression prediction model, particularly Enfomer, using deep learning, and concludes that these prediction models mostly do not consider distal enhancers for gene regulation.


## Prediction of Regulatory region without using DNA sequence 

[ATACworks - Lal et al. Nature Comm 2021](https://www.nature.com/articles/s41467-021-21765-5) Denoising ATAC-seq data and peak calling. Does not use DNA sequence but rather employs coverage around individual base pairs (6 Kb region). Performs denoising and peak calling. Uses ResNet architecture.


## Drug target prediction

[NETTAG - Xu et al. 2022](https://pubmed.ncbi.nlm.nih.gov/36450252/) - DL model to predict Alzheimer's disease (AD) risk genes. Integrates mutli-omics information - PPIs, QTLs, TFs, ENCODE, GWAS, GTEx, GO. Utilizes PPI + GO to prioritize putative AD risk genes, and assigns scores based on their regulatory information (QTLs, ENCODE, etc).


