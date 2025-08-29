# Procrustes Analysis

import numpy as np
import argparse
import gzip
import pandas as pd
import os
import subprocess


def run_fithic(hic_matrix, output_dir, prefix, resolution = 100000):
	hic_matrix = np.loadtxt(hic_matrix)

	# fragment file
	with gzip.open(os.path.join(output_dir, f"{prefix}_fragment.txt.gz"), 'wt', encoding='utf-8') as file:
		# file.write("chr\textraField\tfragmentMid\tmarginalizedContactCount\tmappable?(0\1)")
		for bin in range(hic_matrix.shape[0]):
			contact_count = int(np.sum(hic_matrix[bin, :]))
			if contact_count <= 0:
				continue
			frag_mid = int(bin * resolution + resolution / 2)
			line = f"1\t0\t{frag_mid}\t{contact_count}\t1\n"
			file.write(line)
	
	with gzip.open(os.path.join(output_dir, f"{prefix}_interaction.txt.gz"), 'wt', encoding='utf-8') as file:
		for i in range(hic_matrix.shape[0]):
			frag_mid_i = int(i * resolution + resolution / 2)
			for j in range(i + 1, hic_matrix.shape[0]):
				count = int(hic_matrix[i][j])
				if count <= 0:
					continue
				frag_mid_j = int(j * resolution + resolution / 2)
				line = f"1\t{frag_mid_i}\t1\t{frag_mid_j}\t{count}\n"
				file.write(line)

	command = [
		'fithic',
		'-f', os.path.join(output_dir, f"{prefix}_fragment.txt.gz"),
		'-i', os.path.join(output_dir, f"{prefix}_interaction.txt.gz"),
		'-o', os.path.join(output_dir, prefix),
		'-r', '100000',
		'--upperbound', '10000000',
		'--lowerbound', '200000'
	]

	try:
		result = subprocess.run(command, capture_output=True, text=True, check=True)
	except subprocess.CalledProcessError as e:
		print(f"Error occurred: {e}")
		print(f"Error output: {e.stderr}")
	except FileNotFoundError:
		print(f"Command not found. Make sure Conda is installed and in your PATH.")
	
	return 0

def compare_fithic_output(result, gt):

	# Load data
	data = pd.read_csv(result, sep='\t', header=0, compression='gzip')
	true_data = pd.read_csv(gt, sep='\t', header=0, compression='gzip')

	# Define significant interactions
	significant = data[data['q-value'] < 1e-6]

	# Assume 'true_data' is loaded similarly and filtered for significant interactions
	true_significant = true_data[true_data['q-value'] < 1e-6]

	# print(true_data)
	# Calculate intersections and differences
	TP = pd.merge(significant, true_significant, how='inner', on=['fragmentMid1', 'fragmentMid2'])
	# print(TP)
	FP = pd.concat([significant, TP]).drop_duplicates(keep=False)
	FN = pd.concat([true_significant, TP]).drop_duplicates(keep=False)

	# Calculate metrics
	# print(len(TP))
	# print(len(true_significant))
	recall = len(TP) / len(true_significant) if len(true_significant) > 0 else 0
	precision = len(TP) / len(significant) if len(significant) > 0 else 0
	f1_score = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

	# print(f"Recall: {recall}, Precision: {precision}, F1 Score: {f1_score}")
	return recall, precision, f1_score