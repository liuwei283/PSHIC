import numpy as np
from cmath import inf
import math
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import norm
from cmath import inf
import sys
import numpy as np
import math
# import torch
import random
import pandas as pd
import os
from Bio import SeqIO
import re
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Akima1DInterpolator

def generate_dataset(output_dir, args):

    global phasing_error
    phasing_error = args['phasing_error']

    pat_cod_100kb = np.loadtxt("./sim_30M_45M_pat.txt")
    mat_cod_100kb = np.loadtxt("./sim_30M_45M_mat.txt")

    pat_hic = calculate_contact_map(pat_cod_100kb)
    mat_hic = calculate_contact_map(mat_cod_100kb)
    chd_hic = pat_hic + mat_hic

    plot_hic_log(pat_hic, mat_hic, output_dir)

    np.savetxt(f"{output_dir}/pat.txt", pat_hic)
    np.savetxt(f"{output_dir}/mat.txt", mat_hic)
    np.savetxt(f"{output_dir}/chd.txt", chd_hic)


    # generate sequence of given SNP density
    reference_chr_path = "../reference_seq/hg19_chr22/chr22.fa"
    bin_snps, phasing_info = simulate_haplotypes(reference_chr_path)
    # pat_phasing_info = phasing_info
    # mat_phasing_info = [1 - x for x in phasing_info]
    # print(pat_phasing_info)
    snps_sum = [0]
    for bin_num in range(len(bin_snps.keys())):
        snps_sum.append(snps_sum[-1] + len(bin_snps[bin_num]))
    
    out_valid_pairs = open(f"{output_dir}/ashic.validPairs", "w")

    single_end, both_end, ambigous = simulate_reads(pat_hic, bin_snps, out_valid_pairs, "ref", "alt", phasing_info, snps_sum)
    print("Paternal:")
    print(single_end, both_end, ambigous)
    single_end, both_end, ambigous = simulate_reads(mat_hic, bin_snps, out_valid_pairs, "alt", "ref", phasing_info, snps_sum)
    print("Maternal:")
    print(single_end, both_end, ambigous)

    out_valid_pairs.close()

def simulate_reads(hic_matrix, bin_snps, out_valid_pairs, allele_tag, allele_tag_homo, phasing_info, snps_sum):
    single_end_total = 0
    both_end_total = 0
    ambigous_total = 0

    for i in range(hic_matrix.shape[0]):
        for j in range(i + 1, hic_matrix.shape[0]):
            hap_count = int(hic_matrix[i][j])
            if hap_count == 0:
                continue
            single_end, both_end, ambigous = write_valid_pairs(bin_snps, hap_count, resolution, i, j, out_valid_pairs, allele_tag, allele_tag_homo, phasing_info, snps_sum)
            # write_fastq(ligation_reads, i, j, out_fastq1, out_fastq2, hap_id)
            single_end_total += single_end
            both_end_total += both_end
            ambigous_total += ambigous
    return single_end_total, both_end_total, ambigous_total

def write_valid_pairs(bin_snps, hap_count, resolution, i, j, out_valid_pairs, allele_tag, allele_tag_homo, phasing_info, snps_sum):
    if i in bin_snps:
        i_snps = bin_snps[i]
    else:
        i_snps = []
    if j in bin_snps:
        j_snps = bin_snps[j]
    else:
        j_snps = []

    phasing_info_i = phasing_info[snps_sum[i]:snps_sum[i + 1]]
    phasing_info_j = phasing_info[snps_sum[j]:snps_sum[j + 1]]

    i_range = [i * resolution, (i + 1) * resolution - read_length]
    j_range = [j * resolution, (j + 1) * resolution - read_length]

    single_end_total = 0
    both_end_total = 0
    ambigous_total = 0

    for c in range(hap_count):
        i_start = random.randint(i_range[0], i_range[1])
        j_start = random.randint(j_range[0], j_range[1])
        i_covered, hap_i = check_snp_coverage(i_snps, i_start, phasing_info_i)
        j_covered, hap_j = check_snp_coverage(j_snps, j_start, phasing_info_j)
        if i_covered:
            if hap_i == 0:
                i_tag = allele_tag
            elif hap_i == 1:
                i_tag = allele_tag_homo
        else:
            i_tag = "both-ref"
        if j_covered:
            if hap_j == 0:
                j_tag = allele_tag
            elif hap_j == 1:
                j_tag = allele_tag_homo
        else:
            j_tag = "both-ref"
        if i_covered and j_covered:
            both_end_total += 1
        elif i_covered or j_covered:
            single_end_total += 1
        else:
            ambigous_total += 1
        line = f"chr22\t{i_start}\t{i_tag}\tchr22\t{j_start}\t{j_tag}\ttest\n"
        out_valid_pairs.write(line)
    return single_end_total, both_end_total, ambigous_total

def check_snp_coverage(bin_snp, start, phasing_info):
    is_covered = False
    count = 0
    snp_info = []
    for snp in bin_snp:
        if snp >= (start + read_length):
            break
        if snp >= start and snp < start + read_length:
            is_covered = True
            snp_info.append(phasing_info[count])
        count += 1
    if all(x == 0 for x in snp_info):
        hap_info = 0
    elif all(x == 1 for x in snp_info):
        hap_info = 1
    else:
        hap_info = -1
    is_covered = len(snp_info) > 0 and (hap_info == 0 or hap_info == 1) 
    
    return is_covered, hap_info

def simulate_haplotypes(reference_file, chr_name = "chr22"):
    haplotype1 = []
    haplotype2 = []
    bin_snps = {}
    phasing_info = []

    with open(reference_file, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            # Check if the record is for chromosome 22
            if record.id != chr_name:
                continue
            ref_seq = str(record.seq).upper()[30000000:45000000]
            print("Reference sequence length: ", len(ref_seq))
            pre_switch = False
            for bp_idx in range(len(ref_seq)):
                base = ref_seq[bp_idx]
                if base == 'N':
                    print("Error: N")
                    sys.exit()
                else:
                    mutated_base = simulate_mutation(base, snps_density)
                    mutated_base1 = base
                    mutated_base2 = mutated_base
                    if mutated_base != base:
                        if len(phasing_info) == 0:
                            phasing_info.append(0)
                        bin_idx = bp_idx // resolution
                        if bin_idx not in bin_snps:
                            bin_snps[bin_idx] = [bp_idx]
                        else:
                            bin_snps[bin_idx].append(bp_idx)
                        
                        if random.random() < (phasing_error / 2): # mismatch error
                            if pre_switch == True:
                                phasing_info.append(0)
                            else:
                                phasing_info.append(1)
                        elif random.random() < (phasing_error / 2): # switch error
                            if pre_switch == True:
                                pre_switch = False
                                phasing_info.append(0)
                            else:
                                pre_switch = True
                                phasing_info.append(1)
                        else:
                            if pre_switch == True:
                                phasing_info.append(1)
                            else:
                                phasing_info.append(0)
                    haplotype1.append(mutated_base1) # reference seq
                    haplotype2.append(mutated_base2) # alterate seq
    return bin_snps, phasing_info # ''.join(haplotype1), ''.join(haplotype2)
    
def simulate_mutation(base, mutation_rate):
    if random.random() < mutation_rate:
        bases = ['A', 'C', 'G', 'T']
        bases.remove(base)
        return random.choice(bases)
    return base
    
def calculate_contact_map(cod_matrix):
    # calculate the expected contact matrix
    expanded_matrix = cod_matrix[:, np.newaxis, :]
    diff_squared = (expanded_matrix - cod_matrix)**2
    pairwise_distances = np.sqrt(np.sum(diff_squared, axis=-1))
    np.fill_diagonal(pairwise_distances, 0.5)
    hic_matrix = np.power(pairwise_distances, alpha) * beta
    # poisson_samples = np.vectorize(np.random.poisson)(hic_matrix)
    np.fill_diagonal(hic_matrix, 0)
    # np.fill_diagonal(poisson_samples, 1)
    # hic_matrix = np.log(hic_matrix * 1000 + 1)
    return hic_matrix # poisson_samples

def transform_points(matrix, cube_radius=5):
    centroid = np.mean(matrix, axis=0)
    translated = matrix - centroid
    min_values = np.min(translated, axis=0)
    translated_to_first_quadrant = translated - min_values
    
    max_distance = np.max(np.abs(translated_to_first_quadrant), axis=0)
    scale_factors = cube_radius / max_distance
    
    scaled = translated_to_first_quadrant * scale_factors
    
    return scaled

def plot_hic_log(matrix1, matrix2, output_dir):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    im1 = axes[0].imshow(np.log1p(matrix1), aspect='auto', cmap='viridis')
    fig.colorbar(im1, ax=axes[0])
    axes[0].set_title('Paternal')
    im2 = axes[1].imshow(np.log1p(matrix2), aspect='auto', cmap='magma')
    fig.colorbar(im2, ax=axes[1])
    axes[1].set_title('Maternal')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gt.png")
    plt.close(fig)

global alpha, beta, snps_density, radius, resolution, read_length, phasing_error
alpha = -3
beta = 1000
snps_density = 0.001
radius = 10
resolution = 100000
read_length = 100