# Procrustes Analysis

import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim
from cmath import inf
import sys
import math
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import scipy.spatial as sp
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Akima1DInterpolator
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

def rescale_structure(X):
	centroid = np.mean(X, axis=0)
	scaled_X = X - centroid
	scale = np.sqrt(np.mean(np.sum(scaled_X**2, axis=1)))
	return scaled_X / scale

def calculate_distances(X):
	from scipy.spatial import distance_matrix
	return distance_matrix(X, X)

def get_distance_error_rate(Xm, Xp, Xm_hat, Xp_hat):

	Xm_scaled = rescale_structure(Xm)
	Xp_scaled = rescale_structure(Xp)
	Xm_hat_scaled = rescale_structure(Xm_hat)
	Xp_hat_scaled = rescale_structure(Xp_hat)
	
	d_m = calculate_distances(Xm_scaled)
	d_p = calculate_distances(Xp_scaled)
	d_m_hat = calculate_distances(Xm_hat_scaled)
	d_p_hat = calculate_distances(Xp_hat_scaled)
	
	# der_m = np.sum(np.abs(d_m_hat - d_m)) / np.sum(d_m)
	# der_p = np.sum(np.abs(d_p_hat - d_p)) / np.sum(d_p)
	
	# return (der_m + der_p) / 2
	der = (np.sum(np.abs(d_m_hat - d_m)) + np.sum(np.abs(d_p_hat - d_p))) / (np.sum(d_p) + np.sum(d_m))
	return der

def get_recovery_rate(res_p_hic, res_m_hic, gt_p_hic, gt_m_hic):
	imputed_sum = np.sum(res_p_hic) + np.sum(res_m_hic)
	template_sum = np.sum(gt_p_hic) + np.sum(gt_m_hic)
	# print(imputed_sum)
	# print(template_sum)
	return imputed_sum / template_sum

def get_recovery_rate_inter(res_p_hic, res_m_hic, gt_p_hic, gt_m_hic, res_mp_hic):
	imputed_sum = np.sum(res_p_hic) + np.sum(res_m_hic) + np.sum(res_mp_hic)
	template_sum = np.sum(gt_p_hic) + np.sum(gt_m_hic)
	# print(imputed_sum)
	# print(template_sum)
	return imputed_sum / template_sum

def get_ssim_score(res_p_hic, res_m_hic, gt_p_hic, gt_m_hic):
	ssim_p = ssim(res_p_hic, gt_p_hic, data_range=gt_p_hic.max() - gt_p_hic.min())
	ssim_m = ssim(res_m_hic, gt_m_hic, data_range=gt_m_hic.max() - gt_m_hic.min())
	ssim_score = (ssim_p + ssim_m) / 2
	return ssim_score

def get_fnd(res_p_hic, res_m_hic, gt_p_hic, gt_m_hic):
	fnd_p = np.linalg.norm((res_p_hic - gt_p_hic), 'fro')
	fnd_m = np.linalg.norm((res_m_hic - gt_m_hic), 'fro')
	fnd = (fnd_m + fnd_p) / 2
	return fnd

def remove_outliers(matrix):
	flat_matrix = matrix.flatten()
	sorted_indices = np.argsort(flat_matrix)
	# smallest_indices = sorted_indices[:]
	largest_indices = sorted_indices[:num_outlier]
	# flat_matrix[smallest_indices] = 0
	flat_matrix[largest_indices] = 0
	clean_matrix = flat_matrix.reshape(matrix.shape)
	return clean_matrix

# def plot_hic_matrices(simulated_P, simulated_M, result_P, result_M):
# 	max_val = max(np.max(simulated_P), np.max(simulated_M), np.max(result_P), np.max(result_M))
# 	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
# 	im1 = axes[0, 0].imshow(simulated_P, aspect='auto', cmap='viridis', vmin=0, vmax=max_val)
# 	fig.colorbar(im1, ax=axes[0, 0])
# 	axes[0, 0].set_title('Simulated Paternal')
# 	im2 = axes[0, 1].imshow(simulated_M, aspect='auto', cmap='magma', vmin=0, vmax=max_val)
# 	fig.colorbar(im2, ax=axes[0, 1])
# 	axes[0, 1].set_title('Simulated Maternal')
# 	im3 = axes[1, 0].imshow(result_P, aspect='auto', cmap='viridis', vmin=0, vmax=max_val)
# 	fig.colorbar(im3, ax=axes[1, 0])
# 	axes[1, 0].set_title('Result Paternal')
# 	im4 = axes[1, 1].imshow(result_M, aspect='auto', cmap='magma', vmin=0, vmax=max_val)
# 	fig.colorbar(im4, ax=axes[1, 1])
# 	axes[1, 1].set_title('Result Maternal')
# 	plt.tight_layout()
# 	plt.show()

def process_matrix(matrix, nan_indices):
	# to do: remove diagnol for all methods
	matrix[nan_indices] = 0
	np.fill_diagonal(matrix, 0)
	np.fill_diagonal(matrix[1:], 0)
	np.fill_diagonal(matrix[:, 1:], 0)
	# matrix = remove_outliers(matrix)
	return matrix

def get_log(matrix):
	matrix[matrix < 1] = 1
	matrix = np.log(matrix)
	matrix[matrix < 0] = 0
	return matrix

global num_outlier
num_outlier = 0

def eval_res(res_p_hic_path, res_m_hic_path, gt_p_hic_path, gt_m_hic_path, struct = False, res_p_struct_path = 0, res_m_struct_path = 0, gt_p_struct_path = 0, gt_m_struct_path = 0, inter = False, res_mp_hic_path = 0):
	# To Do: comparing results of different methods and plot
	# To Do: design our own evalutation methods

	###### read data matrix
	res_parental_hic = [] 
	res_parental_hic.append(np.loadtxt(res_p_hic_path))
	res_parental_hic.append(np.loadtxt(res_m_hic_path))
	gt_p_hic = np.loadtxt(gt_p_hic_path)
	gt_m_hic = np.loadtxt(gt_m_hic_path)
	if inter:
		res_mp_hic = np.loadtxt(res_mp_hic_path)

	nan_indices = np.isnan(res_parental_hic[0])

	if struct == True:
		res_parental_struct = []
		res_parental_struct.append(np.loadtxt(res_p_struct_path))
		res_parental_struct.append(np.loadtxt(res_m_struct_path))
		gt_p_struct = np.loadtxt(gt_p_struct_path)
		gt_m_struct = np.loadtxt(gt_m_struct_path)

	res_parental_hic[0] = process_matrix(res_parental_hic[0], nan_indices)
	res_parental_hic[1] = process_matrix(res_parental_hic[1], nan_indices)
	gt_p_hic = process_matrix(gt_p_hic, nan_indices)
	gt_m_hic = process_matrix(gt_m_hic, nan_indices)

	if struct:
		res_parental_struct[0][np.isnan(res_parental_struct[0])] = 0
		res_parental_struct[1][np.isnan(res_parental_struct[1])] = 0

	if inter:
		res_mp_hic = process_matrix(res_mp_hic, nan_indices)


	# match the p and m templates and results
	hic_sim = []
	for res_hic in res_parental_hic:
		sim = ssim(res_hic, gt_p_hic, data_range=gt_p_hic.max() - gt_p_hic.min())
		hic_sim.append(sim)


	
	if hic_sim[0] >= hic_sim[1]:
		res_p_hic = res_parental_hic[0]
		res_m_hic = res_parental_hic[1]
		if struct:
			res_p_struct = res_parental_struct[0]
			res_m_struct = res_parental_struct[1]

	else:
		res_p_hic = res_parental_hic[1]
		res_m_hic = res_parental_hic[0]
		if struct:
			res_p_struct = res_parental_struct[1]
			res_m_struct = res_parental_struct[0]

	np.savetxt(res_p_hic_path.replace(".txt", "_matched.txt"), res_p_hic)
	np.savetxt(res_m_hic_path.replace(".txt", "_matched.txt"), res_m_hic)
	if struct:
		np.savetxt(res_p_struct_path.replace(".txt", "_matched.txt"), res_p_struct)
		np.savetxt(res_m_struct_path.replace(".txt", "_matched.txt"), res_m_struct)

	# ########## Matrix Similarity
	if inter:
		recovery_rate = get_recovery_rate_inter(res_p_hic, res_m_hic, gt_p_hic, gt_m_hic, res_mp_hic)
	else:
		recovery_rate = get_recovery_rate(res_p_hic, res_m_hic, gt_p_hic, gt_m_hic)
	# ssim_score = get_ssim_score(res_p_hic, res_m_hic, gt_p_hic, gt_m_hic)
	distance_error_rate = 0
	# disparity = 0

	if struct:
		distance_error_rate = get_distance_error_rate(res_p_struct, res_m_struct, gt_p_struct, gt_m_struct)
	
	return recovery_rate, distance_error_rate #, disparity