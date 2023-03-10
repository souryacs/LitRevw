# Deep Learning in Computational Biology

### Important papers and brief discussions related to their methodologies

#### Variant Calling:

[DeepVariant - Poplin et al. Nat Biotech 2018](https://www.nature.com/articles/nbt.4235): Uses samtools read pileups + known genotype calls for training. Uses CNN + Stochastic gradient descent model.

#### Prediction of Regulatory region using DNA sequence 

[Basset - Kelley et al. Genome Research 2016](https://genome.cshlp.org/content/26/7/990.long): Predicts regulatory DNA sequences and sequence activities (chomatin accessibility). Uses DNase-seq, DHS and peaks, and applies to deep CNN. CNNs learn relevant sequence motifs and regulatory logic. Assigns GWAS variants and cell-type-scores to predict chromatin accessibility difference between alleles, and predicts causal SNPs.

[Basenji - Kelley et al. Genome Research 2018](https://genome.cshlp.org/content/28/5/739.long): Extends Basset, which only generates peak based chromatin profiles. Predicts epigenomic and transcriptional effects using the ChIP-seq, DNAse seq, ATAC-seq. Also identifies causal variants using GWAS loci. Predicts distal regulatory interactions and finer resolution chromatin profiles.
  * Note * : The input sequencing dataset is processed by a custom pipeline to use the multi-mapping reads and to normalize for GC bias.






