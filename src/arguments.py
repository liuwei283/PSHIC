import argparse

def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description='Hi-C Phase Analysis Tool')

    # Parameters for optimization
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-6, 
                        help='Learning rate for the optimization process.')
    parser.add_argument('--momentum', type=float, default=0.99, 
                        help='Momentum factor for the optimizer.')
    parser.add_argument('--dimension', '-k', type=int, default=4, 
                        help='Dimension of gene expression features.')
    parser.add_argument('--task-name', '-t', type=str, required=True, 
                        help='Name of the task.')
    parser.add_argument('--hic-matrix', '-m', type=str, required=True, 
                        help='Path to the extracted Hi-C matrix.')
    parser.add_argument('--radius', '-r', type=int, default=3, 
                        help='Space constraint radius.')
    parser.add_argument('--noise', '-n', type=float, default=10.0, 
                        help='Noise level for matrix initialization; higher values increase differences between P and M.')
    parser.add_argument('--output-dir', '-o', type=str, default="result", 
                        help='Directory for output files.')
    parser.add_argument('--alpha', '-a', type=float, default=3.0, 
                        help='Alpha parameter for the transfer function.')
    parser.add_argument('--max-depth', '-d', type=int, default=1000, 
                        help='Maximum depth for sequencing coverage.')
    parser.add_argument('--dd-loss-weight', '-w', type=float, default=10.0, 
                        help='Weight for the dd loss function.')
    parser.add_argument('--outlier-threshold', type=int, default=2, 
                        help='Threshold for identifying outliers.')

    # Uncomment and modify the following lines if needed
    # parser.add_argument('--reference', '-r', type=str, help='Reference sequences, e.g., hg38.')
    # parser.add_argument('--chr', type=str, help='Chromosome ID, e.g., chr22.')

    args = parser.parse_args()
    return args

# Example data sources for reference
# ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/ALL.chr22.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz
# ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_raw_GT_with_annot/20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr22.recalibrated_variants.vcf.gz