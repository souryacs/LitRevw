# Deep Learning in Computational Biology

## Variant Calling:

[DeepVariant - Poplin et al. Nat Biotech 2018](https://www.nature.com/articles/nbt.4235): Uses samtools read pileups + known genotype calls for training. Uses CNN + Stochastic gradient descent model.

## QTL / SNP / GWAS etc.

[TL-PRS: Zhao et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/36240765/) Constructing cross-population polgenic risk scores using transfer learning.


## Prediction of Regulatory region using DNA sequence

[Basset - Kelley et al. Genome Research 2016](https://genome.cshlp.org/content/26/7/990.long): Predicts regulatory DNA sequences and sequence activities (chomatin accessibility). Uses DNase-seq, DHS and peaks, and applies to deep CNN. CNNs learn relevant sequence motifs and regulatory logic. Assigns GWAS variants and cell-type-scores to predict chromatin accessibility difference between alleles, and predicts causal SNPs.

[Basenji - Kelley et al. Genome Research 2018](https://genome.cshlp.org/content/28/5/739.long): Extends Basset, which only generates peak based chromatin profiles. Predicts epigenomic and transcriptional effects using the ChIP-seq, DNAse seq, ATAC-seq. Also identifies causal variants using GWAS loci. Predicts distal regulatory interactions and finer resolution chromatin profiles.

 **Note** : The input sequencing dataset is processed by a custom pipeline to use the multi-mapping reads and to normalize for GC bias.

[CORE-ATAC - Thibodeau et al. Plos Comp Biol 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009670) Prediction of *cis*-CREs using ATAC-seq data. Uses CNN + max pooling. Can not predict chromHMM annotations but predicts top 3 functional annotations.

[BIONIC - Forster et al. Nat Meth 2022](https://www.nature.com/articles/s41592-022-01616-x): Biological network integration. Uses GAT.
  
  **Note** : Modifies GAT to consider a priori network edge weights (Methods, eqs. 1-2). Uses 3 GAT layers, 10 attention heads per GAT encoder, each with a hidden dimension of 68, as per their hyperparameter optimization results.
  
[BindSpace - Yuan et al. Nature Methods 2019](https://pubmed.ncbi.nlm.nih.gov/31406384/) Predicts TF binding motifs from DNA sequences, using StarSpace framework (a NLP based model).


## Prediction of Regulatory region without using DNA sequence 

[ATACworks - Lal et al. Nature Comm 2021](https://www.nature.com/articles/s41467-021-21765-5) Denoising ATAC-seq data and peak calling. Does not use DNA sequence but rather employs coverage around individual base pairs (6 Kb region). Performs denoising and peak calling. Uses ResNet architecture.



