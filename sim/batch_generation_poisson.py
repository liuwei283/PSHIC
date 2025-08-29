import numpy as np
from datasets_simulation import generate_matrix, generate_validpairs
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Hi-C simulation data with Poisson distribution.")
    parser.add_argument("--rep-num", type=int, default=10, help="Number of repetitions.")
    parser.add_argument("--output-root-dir", type=str, default="D:\\hic\\sim_batch_poisson\\simulation", help="Root directory for output.")
    return parser.parse_args()

def generate_simulation_data(rep_num, output_root_dir):
    beta_values = [2000, 1500, 1000, 500]
    snp_densities = [0.01, 0.005, 0.002, 0.001]
    phasing_errors = [0, 0.0005, 0.001, 0.005, 0.01, 0.02]

    for beta_val in beta_values:
        beta_dir = os.path.join(output_root_dir, f"beta{beta_val}")
        os.makedirs(beta_dir, exist_ok=True)
        for rep_i in range(rep_num):
            sim_name = f"rep{rep_i}"
            sim_dir = os.path.join(beta_dir, sim_name)
            os.makedirs(sim_dir, exist_ok=True)

            p_hic, m_hic = generate_matrix(beta_val, sim_dir)

            if beta_val == 2000:
                for phasing_error_val in phasing_errors:
                    formatted_phasing_error_val = "{:.4f}".format(phasing_error_val)
                    error_dir = os.path.join(sim_dir, f"error{formatted_phasing_error_val}")
                    os.makedirs(error_dir, exist_ok=True)
                    if phasing_error_val == 0:
                        for snp_density_val in snp_densities:
                            formatted_snp_density_val = "{:.4f}".format(snp_density_val)
                            snp_dir = os.path.join(error_dir, f"snp{formatted_snp_density_val}")
                            os.makedirs(snp_dir, exist_ok=True)
                            generate_validpairs(p_hic, m_hic, snp_dir, snp_density_val, 0)    
                    else:
                        generate_validpairs(p_hic, m_hic, error_dir, 0.002, phasing_error_val)
            else:
                generate_validpairs(p_hic, m_hic, sim_dir, 0.002, 0)

def main():
    args = parse_arguments()
    generate_simulation_data(args.rep_num, args.output_root_dir)

if __name__ == "__main__":
    main()
