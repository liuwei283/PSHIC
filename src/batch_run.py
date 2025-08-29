import subprocess
import concurrent.futures
import time
import os
import argparse

def run_external_script(k, i, dname, hic_matrix, lr, momentum, alpha, radius, max_depth, dd_loss_weight, outlier_threshold, result_dir_postfix):
    """Run the external script with specified parameters and log output and errors."""
    base_filename = f"{dname}_output_{k}_{lr}_{momentum}_{alpha}_{radius}_{max_depth}_{dd_loss_weight}_{outlier_threshold}_{i}"
    output_file = os.path.join(f"output_{result_dir_postfix}", f"{base_filename}.txt")
    error_file = os.path.join(f"error_{result_dir_postfix}", f"{base_filename}.txt")
    
    command = [
        'python', 'imputation.py', '-k', str(k), '-o', f'result_{result_dir_postfix}',
        '--momentum', str(momentum), '--noise', "1", '-l', str(lr),
        '--task-name', f'{dname}_k{k}_{lr}_{momentum}_{alpha}_{radius}_{max_depth}_{dd_loss_weight}_{outlier_threshold}_{i}',
        '-m', hic_matrix, '--alpha', str(alpha), '--radius', str(radius),
        '--max_depth', str(max_depth), '--dd_loss_weight', str(dd_loss_weight),
        '--outlier_threshold', str(outlier_threshold)
    ]

    with open(output_file, 'w') as out, open(error_file, 'w') as err:
        result = subprocess.run(command, stdout=out, stderr=err, text=True)

    if result.returncode == 0:
        print(f"Success: Output written to {output_file}")
    else:
        print(f"Error occurred: Details in {error_file}")

def main(result_dir_postfix):
    params = {
        'lr': 0.00005,
        'k': 32,
        'momentum': 0.999,
        'alpha': 2,
        'radius': 3,
        'max_depth': 1000,
        'outlier_threshold': 3,
        'dd_loss_weight': 1,
        'hic_matrix': args.hic_matrix
    }

    # Create directories if they don't exist
    for dir_type in ['result', 'output', 'error']:
        os.makedirs(f"{dir_type}_{result_dir_postfix}", exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(run_external_script, params['k'], i, "1mb", params['hic_matrix'], params['lr'], params['momentum'], params['alpha'], params['radius'], params['max_depth'], params['dd_loss_weight'], params['outlier_threshold'], result_dir_postfix)
            for i in range(128)
        ]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch jobs with different result_dir_postfix.")
    parser.add_argument("--dataset-type", type=str, default=None, help="Type of dataset to process.")
    parser.add_argument("--hic-matrix", type=str, default=None, help="Path to the Hi-C matrix.")
    args = parser.parse_args()

    start_time = time.time()
    main(args.dataset_type)
    end_time = time.time()

    print(f"Total running time: {end_time - start_time:.2f} seconds")