import numpy as np
import os
import csv
import json
import argparse

from fithic_misc import run_fithic, compare_fithic_output
from evaluation_matrices import eval_res

task_name = "k5"

def eval_init(output_dir, beta_val, rep_i, init_num, gt_p_hic, gt_m_hic, pat_fithic_gt, mat_fithic_gt, output_csv_file):
    for init_i in range(init_num):
        init_dir = os.path.join(output_dir, f"init{init_i}")
        eval_model("drhic", beta_val, init_dir, rep_i, init_i, gt_p_hic, gt_m_hic, pat_fithic_gt, mat_fithic_gt, output_csv_file)

def eval_model(model_name, beta_val, model_dir, rep_i, init_i, gt_p_hic, gt_m_hic, pat_fithic_gt, mat_fithic_gt, output_csv_file):
    print(f"Processing {model_dir}")
    json_path = os.path.join(model_dir, f"model_{task_name}.json")
    
    with open(json_path, 'r') as file:
        data = json.load(file)
        loss = data.get("loss", 0)

    res_p_hic = os.path.join(model_dir, f"p_hic_{task_name}.txt")
    res_m_hic = os.path.join(model_dir, f"m_hic_{task_name}.txt")
    res_p_struct = os.path.join(model_dir, f"p_struct_{task_name}.txt")
    res_m_struct = os.path.join(model_dir, f"m_struct_{task_name}.txt")

    log_recovery_rate, distance_error = eval_res(
        res_p_hic, res_m_hic, gt_p_hic, gt_m_hic, struct=True,
        res_p_struct_path=res_p_struct, res_m_struct_path=res_m_struct,
        gt_p_struct_path=gt_p_struct, gt_m_struct_path=gt_m_struct
    )

    res_p_hic_matched = os.path.join(model_dir, f"p_hic_{task_name}_matched.txt")
    res_m_hic_matched = os.path.join(model_dir, f"m_hic_{task_name}_matched.txt")

    run_fithic(res_p_hic_matched, model_dir, "pat")
    run_fithic(res_m_hic_matched, model_dir, "mat")

    pat_fithic_res = os.path.join(model_dir, "pat", "FitHiC.spline_pass1.res100000.significances.txt.gz")
    mat_fithic_res = os.path.join(model_dir, "mat", "FitHiC.spline_pass1.res100000.significances.txt.gz")

    p_recall, p_precision, p_f1_score = compare_fithic_output(pat_fithic_res, pat_fithic_gt)
    m_recall, m_precision, m_f1_score = compare_fithic_output(mat_fithic_res, mat_fithic_gt)

    recall = (p_recall + m_recall) / 2
    precision = (p_precision + m_precision) / 2
    f1_score = (p_f1_score + m_f1_score) / 2

    with open(output_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_name, beta_val, rep_i, init_i, log_recovery_rate, distance_error, recall, precision, f1_score, loss])

gt_p_struct = os.path.join("./sim_30M_45M_pat.txt")
gt_m_struct = os.path.join("./sim_30M_45M_mat.txt")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate DRHiC Poisson models.")
    parser.add_argument("--rep-num", type=int, default=10, help="Number of repetitions.")
    parser.add_argument("--init-num", type=int, default=10, help="Number of initializations.")
    parser.add_argument("--output-root-dir", type=str, default="test", help="Root directory for output.")
    parser.add_argument("--output-csv-file", type=str, default="./results_drhic_poisson_test.csv", help="Output CSV file path.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    beta_values = [2000, 1500, 1000, 500]
    gt_fithic_created = True

    with open(args.output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "beta", "rep", "init", "log_recovery_rate", "distance_error", "Recall", "Precision", "F1", "model_log"])

    for rep_i in range(args.rep_num):
        for beta_val in beta_values:
            beta_dir = os.path.join(args.output_root_dir, f"beta{beta_val}")
            fithic_input_dir = os.path.join(beta_dir, "fithic")

            if not gt_fithic_created:
                gt_p_hic = os.path.join(beta_dir, "pat.txt")
                gt_m_hic = os.path.join(beta_dir, "mat.txt")
                os.makedirs(fithic_input_dir, exist_ok=True)
                run_fithic(gt_p_hic, fithic_input_dir, "pat")
                run_fithic(gt_m_hic, fithic_input_dir, "mat")
            
            gt_p_hic = os.path.join(beta_dir, "pat.txt")
            gt_m_hic = os.path.join(beta_dir, "mat.txt")
            
            pat_fithic_gt = os.path.join(fithic_input_dir, "pat/FitHiC.spline_pass1.res100000.significances.txt.gz")
            mat_fithic_gt = os.path.join(fithic_input_dir, "mat/FitHiC.spline_pass1.res100000.significances.txt.gz")
    
            sim_name = f"rep{rep_i}"
            sim_dir = os.path.join(beta_dir, sim_name, "drhic")
            
            eval_init(sim_dir, beta_val, rep_i, args.init_num, gt_p_hic, gt_m_hic, pat_fithic_gt, mat_fithic_gt, args.output_csv_file)

if __name__ == "__main__":
    main()
