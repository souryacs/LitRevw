## Gene expression / modules

[A genome-wide atlas of co-essential modules assigns function to uncharacterized genes - Wainberg et al. Nature Genetics 2021](https://pubmed.ncbi.nlm.nih.gov/33859415/) GLS (Generalized least square) based framework to detect co-essential gene modules specific to cell / tissue types which can be candidates for perturbation experiments. Differs from CRISPR in the sense that CRISPR background is a single gene while co-essentiality models genetic and phenotype characteristics of a given cell line. Uses CRISPR cell line from Achillis project for implementation, benchmarks with databases CORUM, STRING, hu.MAP, COXPRESdb, DoRothEA. Gene modules are derived by ClusterOne package. GLS regression is performed by Cholesky decomposition, and the input gene effects are obtained by CERES scores. Provides interactive webtool <http://coessentiality.net>.

[Collective effects of long-range DNA methylations predict gene expressions and estimate phenotypes in cancer - Kim et al. Scientic Reports 2020](https://pubmed.ncbi.nlm.nih.gov/32127627/) GeneExplore method - Prediction of gene expression using DNA Methylation regions, utilizing cis and trans chromatin ineractions upto 10 Mb. Applied on Cancer data.