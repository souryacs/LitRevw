# Deep Learning in Computational Biology

[LLM Models - review](https://github.com/RUCAIBox/LLMSurvey)

[Single cell transformer papers - Theislab](https://github.com/theislab/single-cell-transformer-papers)

[Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

## Variant Calling:

[A universal SNP and small-indel variant caller using deep neural networks - DeepVariant - Poplin et al. Nat Biotech 2018](https://www.nature.com/articles/nbt.4235): Uses samtools read pileups + known genotype calls for training. Uses CNN + Stochastic gradient descent model.

[CancerVar- An artificial intelligence–empowered platform for clinical interpretation of somatic mutations in cancer - Li et al. Science Advances 2022](https://pubmed.ncbi.nlm.nih.gov/35544644/): AI based cancer somatic mutation caller.

[Accurate proteome-wide missense variant effect prediction with AlphaMissense - Cheng et al. Science 2023](https://pubmed.ncbi.nlm.nih.gov/37733863/) AlphaMissense approach. DL based protein missense variant identification. Uses Evoformer architecture, same as alphafold2. 

[DeepConsensus improves the accuracy of sequences with a gap-aware sequence transformer - Baid et al. Nature Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/36050551/) DeepConsensus - utilized for PACBIO long-read HiFi Sequencing read alignment and minimizing alignment errors, using GAP (INDEL) aware encoder-only transformer model. Utilizes DeepVariant for variant calling, and shows improved accuracy of variant calling.

[Genome-wide mapping of somatic mutation rates uncovers drivers of cancer - Sherman et al. Nature Biotechnology 2022](https://pubmed.ncbi.nlm.nih.gov/35726091/) Dig method. Deep learning based identification of cancer driver mutations among a set of putative mutations. Supports multiple types of SNV. Implements CNN based method to detect the variants and then a probabilistic model to detect the positive mutations.

[The landscape of tolerated genetic variation in humans and primates - Gao et al. Science 2023](https://pubmed.ncbi.nlm.nih.gov/37262156/) primateAI-3D model. Deep learning model based database of 4.3 million benign missense variants across the primate lineage.

[Rare penetrant mutations confer severe risk of common diseases - Fiziev et al. Science 2023](https://pubmed.ncbi.nlm.nih.gov/37262146/) Using their previusly published primateAI-3D model which identifies benign and pathogenic variants, including rare variants (MAF >= 0.1%), they identify the gene-phenotype associations and compare with the conventional GWAS studies which only include common variants.
They benchmark with another rare variant specific method Backman et al. Nature 2021, and finds more GWAS supported genes. Also derives one polygenic risk score (PRS) prediction method using the rare variants and compare with the conventional PRS prediction methods employing common variants.

## QTL / SNP / GWAS etc.

[The construction of cross-population polygenic risk scores using transfer learning - TL-PRS: Zhao et al. AJHG 2022](https://pubmed.ncbi.nlm.nih.gov/36240765/) Constructing cross-population polgenic risk scores using transfer learning.

[Annotating functional effects of non-coding variants in neuropsychiatric cell types by deep transfer learning - MetaChrom - Lai et al. PLOS Comp Biol. 2022](https://pubmed.ncbi.nlm.nih.gov/35576194/) - Although the title is about annotating functional variants, the approach is similar to the DeepSEA framework, it predicts the epigenomic tracks (histone and chromatin accessibility) from DNA sequences. Uses RESNET architecture and transfer learning to predict the epigenomic tracks.

[DeepCOMBI: explainable artificial intelligence for the analysis and discovery in genome-wide association studies - Mieth et al. NAR Genomics Bioinformatics](https://pubmed.ncbi.nlm.nih.gov/34296082/) adopts layer-wise relevance propagation (LRP) to attribute SNP relevance scores and selection of significant SNPs in GWAS studies. Replaces conventional p-value thresholding. Extends their earlier work COMBI which uses SVM based method, to a DNN setting.

[REGLE - Unsupervised representation learning improves genomic discovery for lung function and respiratory disease prediction - Yun et al. medRxiv 2023](https://www.medrxiv.org/content/10.1101/2023.04.28.23289285v1) From Google Research. Proposes low dimensional representation learning of high-dimensional clinical data (HDCD) and utilizes these low-dimensional embeddings to compute polygenic risk scores (PRS). Applies on COPD (chronic obstructive pulmonary disease) and spirograms, lung and respiratory data.

[Genetic association studies using disease liabilities from deep neural networks - Yang et al. medRxiv 2023](https://www.medrxiv.org/content/10.1101/2023.01.18.23284383v1) Also talks about HDCD. First, they introduce GWAX, GWA using proxy. Includes persons as cases whose one of the family members has the disease of interest. Modification of CC GWAS study. Shown to retrive new GWAS risk loci. Performs AI based modeling (their pre-trained DL framework POPDx) on the input HDCD GWAS trait, and the generated GWAS trait features are known as disease liability scores, which are used as the GWAS trait features. They show that these liability scores perform better in GWAS trait mapping.

[EMS - Leveraging supervised learning for functionally informed fine-mapping of cis-eQTLs identifies an additional 20,913 putative causal eQTLs - Wang et al. Nat Comm 2021](https://www.nature.com/articles/s41467-021-23134-8) Work from David Kelley, Hillary Finucane etc. Presents EMS (expression modifier score) to predict fine-mapped causal variants. Trains data using fine-mapped variants derived by SUSIE + FINEMAP, using 49 tissues GTEX v8 data. Then uses annotation features like TSS distance, tissue and non-tissue specific binary annotations, DL features (Basenji scores), and trains a random forest classifier. Feature importance scores mention that Basenji scores and TSS distance are informative features. Using these EMS scores as prior, then they define a functional annotation based fine-mapping (PIP) across 95 traits.  ** Note: check Enformer performance. See the detailed feature list. Use motif binding information. 

## Prediction of Regulatory region / gene expression using DNA sequence

[Predicting effects of noncoding variants with deep learning-based sequence model - Zhou et al. Nature Methods 2015](https://pubmed.ncbi.nlm.nih.gov/26301843/) DEEPSEA deep learning framework. *Input*: DNA Sequence. 200 bp length sequences with at least one TF binding events from 919 chromatin features. Each such 200 bp sequence is used as the center to generate 1000 bp sequence with (1X919) label vector (one label for each chromatin feature and TF binding event). *Output*: 1. Predict DNAse-seq and TF binding from sequence. Allele-specific TF binding and chromatin activity. 2. Implements in-silico mutagenesis approach for variant effect prediction, used in Enformer.
3. Validates functional SNPs using Human Gene Mutation Database, GRASP (Genome-Wide Repository of Associations between SNPs and Phenotypes) database and GWAS catalog. [Their follow up work](https://pubmed.ncbi.nlm.nih.gov/31133750/) is an approach for finding non-coding mutations relevant to ASD. *Input*: WGS data from Simons Simplex Collection. *Output*: 1. Categorized variants based on predicted DNA transcriptional impact or RNA binding protein impact 2. Putative functional noncoding mutations in ASD. 3. Underlying genes and pathways related to functional noncoding mutations by developing network neighborhood diferential enrichment analysis (NDEA).

[Basset - learning the regulatory code of the accessible genome with deep convolutional neural networks - Kelley et al. Genome Research 2016](https://genome.cshlp.org/content/26/7/990.long): Basset method. *Input*: DNA sequence. Uses DNase-seq, DHS and peaks. *Output*: 1. CNN based prediction of chromatin accessibility and TF motifs 2. Predicts important GWAS SNPs by calculating the allele-specific differential chromatin accessibility. 3. Predicts regulatory DNA sequences and sequence activities (chomatin accessibility). 4. In silico mutagenesis to find allele-specific changes. 5. Tests with PICS fine-mapped GWAS SNPs. *Method*: Deep CNNs learn relevant sequence motifs and regulatory logic. Assigns GWAS variants and cell-type-scores to predict chromatin accessibility difference between alleles, and predicts causal SNPs. 

[Sequential regulatory activity prediction across chromosomes with convolutional neural networks - Basenji - Kelley et al. Genome Research 2018](https://genome.cshlp.org/content/28/5/739.long): Extends Basset, which only generates peak based chromatin profiles. Predicts epigenomic and transcriptional effects using the ChIP-seq, DNAse seq, ATAC-seq. Also identifies causal variants using GWAS loci. Predicts distal regulatory interactions and finer resolution chromatin profiles. Predicts a signed profile of distal regulatory regions to indicate if those are enhancers or silencers. Similarly, for a given SNP, it predicts the SNP expression difference (SED) score to characterize if these are eQTLs. Also reports a disease-specific variant scores, and tests with input PICS fine-mapped variants. **Implementation details** : 1) The input sequencing dataset is processed by a custom pipeline to use the multi-mapping reads and to normalize for GC bias. 2) Weight values are initialized by Glorot initialization. 3) GPyOpt python package is used for Bayesian optimization and hyperparameter tuning. 4) Data augmentation is done by either using reverse complement DNA sequences in every alternate epoch, and minor sequence shifts.

[Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk - Expecto - Zhou et al. Nat Genet 2018](https://pubmed.ncbi.nlm.nih.gov/30013180/) Deep learning framework to predict gene expression from epigenomic tracks. Uses 2002 tracks across 218 cell types, 40 Kb (20 kb in each direction) sequence from TSS, and applies 200 bp sldiding window for constructing the epigenomic features. Then applies spatial operation (basically averaging) to generate 10 features per track (2002 X 10 feature matrix) to predict the tissue-specific gene expression using L2 regularized linear regression models fitted by a gradient boosting algorithm.

[Effective gene expression prediction from sequence by integrating long-range interactions - Enformer - Avsec et al. Nature Methods 2021](https://pubmed.ncbi.nlm.nih.gov/34608324/) predicts gene expression from DNA sequences using transformer model and 198 bp receptive field around TSS. Best performing model so far. Beats earlier models Expecto, Basenji2. It also predicts signed effect of variants (or sequences) to check whether corresponding segment is either enhancer or repressor. Predicts the allele-specific changes in gene expression, motifs and variants. Also predicts the effect of distal regulatory enhancers (saliency score). Future work: 1) use the enformer derived scores and functional validations to fine-map the GWAS variants (similar to EMS), 2) Benchmark / use the enhancer prioritization with respect to conventional Hi-C, HiChIP maps.

[Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation - Linder et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1) Borzoi method. Extension of Enformer. *Input*: RNA Sequence. *Output*: 1. Predicted RNA coverage 2. Gene expression (TSS pr exon specific) 3. Enhancer prioritization by saliency score 4. QTL variant effect *Method*: Transformer + CNN + U-Net architecture (to apply attention at 128bp but predict RNA-seq coverage at 32bp, by repeated upsampling technique employed in image processing). 

[Deep learning predicts DNA methylation regulatory variants in the human brain and elucidates the genetics of psychiatric disorders - INTERACT - Zhou et al. PNAS 2022](https://pubmed.ncbi.nlm.nih.gov/35969790/) predicts DNA methylation levels from DNA sequences, tissue specific DNA methylation data, TF motifs (validated by TOMTOM motif analysis tool) and also predicts DNA methylation QTLs (sequence variants) which are further integrated to brain GWAS studies. Presents a transformer based learning model to predict the changes in DNA methylation level from variants (mQTLs). Trains the data on SUSIE-derived fine-mapped mQTLs.

[CoRE-ATAC: A deep learning model for the functional classification of regulatory elements from single cell and bulk ATAC-seq data - Thibodeau et al. Plos Comp Biol 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009670) Prediction of *cis*-CREs using ATAC-seq data. Uses CNN + max pooling. Can not predict chromHMM annotations but predicts top 3 functional annotations.
  
[BindSpace decodes transcription factor binding signals by large-scale sequence embedding - Yuan et al. Nature Methods 2019](https://pubmed.ncbi.nlm.nih.gov/31406384/) Predicts TF binding motifs from DNA sequences, using StarSpace framework (a NLP based model). Uses HT-SELEX TF database as input, and learns embedding space where TFs with similar binding profiles are closer. Uses one-vs-all LASSO framework on the TF k-mers as used in StarSpace.

[Technical Note on Transcription Factor Motif Discovery from Importance Scores (TF-MoDISco) version 0.5.6.5 - Shrikumar et al. arXiv 2018](https://arxiv.org/abs/1811.00416) Using DeepLIFT framework, designs TF-MoDISCO, a motif discovery framework using the per-base importance score. 

[Base-resolution models of transcription-factor binding reveal soft motif syntax - Avsec et al. Nature Genetics 2021](https://pubmed.ncbi.nlm.nih.gov/33603233/) BPNet method - CNN based TF binding motif prediction from DNA sequence. DNA binding profiles are obtained from ChIP-Nexus profiles. Also tests how the distance between motif pairs affects TF cooperativity. Uses their earlier devleoped method TF-MoDisco and DeepLIFT to understand the base-level contributions to motif scores (or predicted TF binding outputs). Developed a new motif representation called contribution weight matrix (CWM) and also compared with the traditional position frequency matrix (PFM) representation. Validates the motifs by performing targeted point-mutations in mapped motifs and comparing the observed changes in ChIP-Nexus profiles to those predicted by BPNet.

[Short tandem repeats bind transcription factors to tune eukaryotic gene expression - Horton et al. Science 2023](https://pubmed.ncbi.nlm.nih.gov/37733848/) Application of BPNet, to get the scores of STRs (short tandem repeats - 1-6 bp) and predict their influence on TF binding.

[DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers - Almeida et al. Nature Genetics 2022](https://pubmed.ncbi.nlm.nih.gov/35551305/) DeepSTARR method. Predicts chromatin regulatory activity from DNA sequence (STARR-seq). Adapts Basset architecture (CNN). Applies on Drosophilla Genome and also human genome. Identifies the motif syntax specific rules to detect the cis-regulatory code by adapting DeepLIFT and TF-MoDisco.

[Current sequence-based models capture gene expression determinants in promoters but mostly ignore distal enhancers - Review paper: Karollus et al. Genome Biology 2023](https://pubmed.ncbi.nlm.nih.gov/36973806/) reviews the sequence to expression prediction model, particularly Enfomer, using deep learning, and concludes that these prediction models mostly do not consider distal enhancers for gene regulation.

[Benchmarking of deep neural networks for predicting personal gene expression from DNA sequence highlights shortcomings - Sasse et al. Nature Genetics 2023](https://pubmed.ncbi.nlm.nih.gov/38036778/) applies Enformer on the personalized reference genome constructed from WGS data and finds that the prediction accuracy of gene expression for Enformer is actually lower than PrediXcan which uses SNVs from individuals to reconstruct individual level gene expression. Benchmarks Enformer on personalized genome with different genotypes for predicting genotype-dependent changes in gene expression. Examines locus specific examples and finds that for most of the genes, Enformer predicted gene expression shows different patterns than the true genotype-dependent change of gene expression. Also tests prediXcan which shows better prediction of gene expression. Similar observation is in [Personal transcriptome variation is poorly explained by current genomic deep learning models - Huang et al. Nature Genetics 2023](https://pubmed.ncbi.nlm.nih.gov/38036790/) tests on GEUVADIS RNA-seq cohort and shows that prediXcan predicts much better individual-specific gene expression variation compared to the DL methods like Enformer, Basenji2, Expecto, etc.

[Deep learning suggests that gene expression is encoded in all parts of a co-evolving interacting gene regulatory structure - Zrimec et al. Nat Comm 2020](https://pubmed.ncbi.nlm.nih.gov/33262328/) mentions that gene expression can be predicted by using both coding and non-coding regions, and deep learning helps to identify candidate motif regions. Motif co-occurrence helps to decode the gene expression.

[CodonBERT: Large Language Models for mRNA design and optimization - Li et al. bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.09.09.556981v1) Deep learning (BERT) model to predict the mRNA expression from codon occurrence, and codon usage bias. Uses mRNA sequence, masked sequence representation.

[Chromatin interaction-aware gene regulatory modeling with graph attention networks - Karbalayghareh et al. Genome Research 2022](https://pubmed.ncbi.nlm.nih.gov/35396274/) GraphReg method. Developed from HiCDC+ group. Proposes 2 models to predict gene expression from epigenomic data: 1) Epi-GraphReg - uses both epigenomic 1D annotations + 3D interactions (HiC, HiChIP) to predict gene expression. Also optionally predicts 1D tracks (methylation, acetylation). 2) Seq-GraphReg: Uses DNA sequence. Uses GAT (Graph Attention Network) instead of GCN for modeling chromatin interactions.

[CREaTor: zero-shot cis-regulatory pattern modeling with attention mechanisms - Li et al. Genome Biology 2023](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03103-8) CREATOR method. Uses CREs from ENCODE for multiple human cell types (DNA sequences) and applies transformer-based model to predict gene expression. Uses K562 with test chromosomes 8, 9 for performance evaluation. Also uses the attention scores to predict the regulatory importance of CREs (up to 2 Mb from TSS). benchmarks with the reference CRISPR datasets as well (such as Gasperini, Fulco et al.) and also evaluates the importance of CREs up to 2 Mb by testing with different models where CREs are within 500 Kb, 1 Mb, 2 Mb etc. * They use RNA-seq data to model the gene expression - use sum of transcript TPM with log1p transformation as the input gene expression *.

[scBasset: sequence-based modeling of single-cell ATAC-seq using convolutional neural networks - Han Yuan et al. Nature Methods 2022](https://pubmed.ncbi.nlm.nih.gov/35941239/) Single cell ATAC-seq track inference and TF binding signature from DNA sequence, using CNN.


## Prediction of Regulatory region without using DNA sequence 

[ATACworks - Lal et al. Nature Comm 2021](https://www.nature.com/articles/s41467-021-21765-5) Denoising ATAC-seq data and peak calling. Does not use DNA sequence but rather employs coverage around individual base pairs (6 Kb region). Performs denoising and peak calling. Uses ResNet architecture.

## Regulatory Networks

[BIONIC: biological network integration using convolutions - Forster et al. Nat Meth 2022](https://www.nature.com/articles/s41592-022-01616-x): Biological network integration. Uses GAT. **Note** : Modifies GAT to consider a priori network edge weights (Methods, eqs. 1-2). Uses 3 GAT layers, 10 attention heads per GAT encoder, each with a hidden dimension of 68, as per their hyperparameter optimization results. Uses static attention. After Exp() function, multiplies the numerator with the edge weights, and then row normalizes. The network specific node features are then combined by a weighted stochastically masked attention.

[Transfer learning enables predictions in network biology - Theodoris et al. Nature 2023](https://pubmed.ncbi.nlm.nih.gov/37258680/) 

    - Presents Geneformer - a transfer learning based method for gene regulatory network inference and predicting network dynamics.
    - *Input*: The training is performed on a huge single cell transcriptomic dataset collection Genecorpus-30M.
    - *Output*: Pre-trained model predicting dosage-sensitive disease genes and their downstream target genes through a context-aware in-silico deletion approach.
    - *Method:*    
        - The gene expression is transformed into a ranking based measure (rank-value encodings)
        - Ranking genes by (expression within that cell / expression across entire Genecorpus-30M)
        - to prioritize the cell-specific expressed genes and deprioritize the ubiquitous expressed housekeeping genes.
        - The rank value encodings were tokenized and stored by HuggingFace transformer library.
    - Training: masked learning by masking 15% genes
        - Predicting which gene should be in each masked position using the context of the remaining unmasked genes.
    - Gene embedding: 256 dimensional space + context awareness. Cell embedding: embedding of genes expressed in that cell.
    - *Applications*:
        - Fine-tuning this model on a specific disease and related few datasets accurately predicts the disease-specific target genes / dosage sensitive genes.
        - Predicting bivalent promoters vs unmethylated promoters.
        - Fine-tuning to distinguish long vs short distance TFs regulating gene expression.
        - Geneformer's cell embeddings and clustering is also shown to remove batch effects, and performs cell type annotations.
        - Using the network architecture and in-silico treatment, we can reveal candidate therapeutic targets.

[Biologically informed deep learning to query gene programs in single-cell atlases - Lotfollahi et al. Nature Cell Biology 2023](https://pubmed.ncbi.nlm.nih.gov/36732632/) 

    - Explainable Programmable Mapper (expiMap) method. Biologically interpretable DL method for single cell reference mapping
        - (mapping query dataset to a set of reference single cell atlas)
    - Identifies gene programs (GP) to contextualize the query data and better integration with reference data. 
        - These GPs are responsible / differential between conditions. 
    - Single cell atlases (with gene expression and biological conditions). 
    
    - Constructing gene programs for a collection of data is performed by fine-tuning and architectural surgery, similar to the scArches method.


## Single cell embedding + downstream analysis

[scGPT: toward building a foundation model for single-cell multi-omics using generative AI - Cao et al. Nature Methods 2024](https://www.nature.com/articles/s41592-024-02201-0) 

    - scGPT method. GPT model for single cell embedding and downstream applications 
        - (fine tuned for clustering, batch correction, cell annotation, GRN inference, Perturbation response modeling).
    - *Input*: Single cell (33M cells)
    - *Methodology*:
        - Gene representation by tokens (gene name vocabulary + special characters)
        - Gene expression values are applied *value binning* technique (after normalization) to convert into relative values. Usually M HVGs are used for embedding.
            - (to create gene tokens and solve scaling issues from different batches / experiments)
            - Sequence of M embedding vectors in transformer architecture (context information) captures relationship between genes.
        - External tokens are used to store meta information.

[GenePT- A Simple But Effective Foundation Model for Genes and Cells Built From ChatGPT - Chen et al. bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2023.10.16.562533v2)

    - GenePT method for scRNA-seq embedding and downstream analysis.
    - Uses genes from Geneformer and scGPT.
    - ChatGPT text description of genes are used to create the gene embeddings (GPT-3.5).
    - Also tested gene summary embeddings from BioLinkBert, and gene expression derived embeddings such as Gene2vec, Geneformer.
    - Cell embeddings are created by weighted combination of gene embeddings
      - Weights are derived by scRNA-seq normalized gene expression
      - Normalized embeddings by unit L2 norm.
      - Another approach is to perform sentence embedding, by creating a sequence of gene names,
        - where the sequence is created by decreasing normalized expression levels, omitting genes with 0 counts.
      - This sentence representation is then applied to GPT 3.5 to create gene embeddings. 

[Cell2Sentence: Teaching Large Language Models the Language of Biology - Levine et al. bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2023.09.11.557287v3) 

    - Genes are ranked by log transformation of expression.
    - Ranked list of gene names are the embeddings per cell, fed into LLM.
    - The inference from LLM is also a ranked list of gene names.
      - They are converted to expression values using a pre-trained regression between rank and gene expression.
    - This list of gene expression is used for downstream analysis such as clustering.


## Drug target prediction / disease-specific analysis

[Interpretable deep learning translation of GWAS and multi-omics findings to identify pathobiology and drug repurposing in Alzheimer's disease - NETTAG - Xu et al. 2022](https://pubmed.ncbi.nlm.nih.gov/36450252/) - DL model to predict Alzheimer's disease (AD) risk genes. Integrates mutli-omics information - PPIs, QTLs, TFs, ENCODE, GWAS, GTEx, GO. Utilizes PPI + GO to prioritize putative AD risk genes, and assigns scores based on their regulatory information (QTLs, ENCODE, etc).

[Identifying common transcriptome signatures of cancer by interpreting deep learning models - Jha et al. Genome Biology 2022](https://pubmed.ncbi.nlm.nih.gov/35581644/) The authors observed that DEGs between normal and cancer patients do not overlap between multiple datasets. So they used the expressions of protein-coding genes, lncRNA, and splicing junctions in an interpretable deep-learning model.

[Biologically informed deep neural network for prostate cancer discovery - Elmarakeby et al. Nature 2021](https://pubmed.ncbi.nlm.nih.gov/34552244/) Proposed an interpretable DL model (named P-NET) using DEEPLIFT framework to understand the molecular mechanisms of cancer. The input is genes, pathways, and biological processes (molecular profile). Their relationships are prior known and downloaded from Reactome pathway datasets (https://reactome.org/). These relationships are provided as the edges in the neural network architecture.

[Genome-wide mapping of somatic mutation rates uncovers drivers of cancer - Dig - Sherman et al. Nat Biotech 2022](https://pubmed.ncbi.nlm.nih.gov/35726091/) Developed by PCAWG consortium, this tool creates a list of genome-wide neutral somatic mutation maps using a DL model (CNN for dimensionality reduction and feature selection + Gaussian process modeling) and then develops a positive selection test to detect the potential cancer driver somatic mutations. *To read in detail*

[Discovery of drug-omics associations in type 2 diabetes with generative deep-learning models - MOVE - Allesoe et al. Nat Biotech 2023](https://pubmed.ncbi.nlm.nih.gov/36593394/) Defines MOVE - Multi-omics variational autoencoder including data from multiple omics from 789 sample cohort (vertical integration) and applies VAE, and defines the association between T2D with the latent space features. Significance is computed by t-test, and by feature purturbation (0/1) technique.


