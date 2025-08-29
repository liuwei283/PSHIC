import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

def remove_outlier_bins(matrix, outlier_threshold):
    """Remove outlier bins from a matrix based on a threshold."""
    row_sums = np.sum(matrix, axis=1)
    threshold = np.mean(row_sums) - outlier_threshold * np.std(row_sums)
    outlier_indices = np.where(row_sums <= threshold)[0]
    filtered_matrix = np.delete(matrix, outlier_indices, axis=0)
    filtered_matrix = np.delete(filtered_matrix, outlier_indices, axis=1)
    return filtered_matrix, outlier_indices

def load_results(folder_path, num_results):
    """Load results and their structures from specified folder."""
    results = []
    results_struct = []
    for i in range(num_results):
        p_file = os.path.join(folder_path, f'M_1mb_k32_5e-05_0.999_2_3_1000_0_3_{i}_0.txt')
        m_file = os.path.join(folder_path, f'P_1mb_k32_5e-05_0.999_2_3_1000_0_3_{i}_0.txt')
        p_struct_file = os.path.join(folder_path, f'P_vector_1mb_k32_5e-05_0.999_2_3_1000_0_3_{i}_0.txt')
        m_struct_file = os.path.join(folder_path, f'M_vector_1mb_k32_5e-05_0.999_2_3_1000_0_3_{i}_0.txt')
        
        if all(os.path.exists(f) for f in [p_file, m_file, p_struct_file, m_struct_file]):
            p_res = np.loadtxt(p_file)
            m_res = np.loadtxt(m_file)
            p_struct = np.loadtxt(p_struct_file)
            m_struct = np.loadtxt(m_struct_file)
            results.append((p_res, m_res))
            results_struct.append((p_struct, m_struct))
    
    return results, results_struct

def match_matrices(results, results_struct):
    """Match matrices using spectral clustering."""
    data = [p_res.flatten() for p_res, m_res in results] + [m_res.flatten() for p_res, m_res in results]
    data = np.array(data)
    
    spectral = SpectralClustering(n_clusters=2, random_state=0, affinity='nearest_neighbors').fit(data)
    labels = spectral.labels_
    print(labels)
    
    cluster_0, cluster_1 = [], []
    cluster_struct_0, cluster_struct_1 = [], []
    
    for count, i in enumerate(range(0, len(labels), 2)):
        if labels[i] != labels[i + 1]:
            if labels[i] == 0:
                cluster_0.append(data[i])
                cluster_struct_0.append(results_struct[count][0])
                cluster_1.append(data[i + 1])
                cluster_struct_1.append(results_struct[count][1])
            else:
                cluster_0.append(data[i + 1])
                cluster_struct_0.append(results_struct[count][1])
                cluster_1.append(data[i])
                cluster_struct_1.append(results_struct[count][0])
        else:
            print("filtered...")
    
    return cluster_0, cluster_1, cluster_struct_0, cluster_struct_1

def plot_clusters_and_differential_map(mean_cluster_0, mean_cluster_1, differential_map, threshold=5000):
    """Plot mean clusters and differential map."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.where(mean_cluster_0 > threshold, threshold, mean_cluster_0), cmap='hot', interpolation='nearest')
    plt.title('Mean Cluster 0')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.where(mean_cluster_1 > threshold, threshold, mean_cluster_1), cmap='hot', interpolation='nearest')
    plt.title('Mean Cluster 1')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(differential_map, cmap='RdBu_r', interpolation='nearest', vmin=-100, vmax=100)
    plt.title('Differential Map (Cluster 0 - Cluster 1)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('mean_clusters_and_differential_xci_5.png')

def main():
    NUM_RESULTS = 128
    folder_path = './result_xci_1/'
    
    results, results_struct = load_results(folder_path, NUM_RESULTS)
    cluster_0, cluster_1, cluster_struct_0, cluster_struct_1 = match_matrices(results, results_struct)
    
    mean_cluster_0 = np.mean(cluster_0, axis=0).reshape(results[0][0].shape)
    mean_cluster_1 = np.mean(cluster_1, axis=0).reshape(results[0][0].shape)
    differential_map = mean_cluster_0 - mean_cluster_1
    
    plot_clusters_and_differential_map(mean_cluster_0, mean_cluster_1, differential_map)
    
    np.savetxt('mean_cluster_0.txt', mean_cluster_0)
    np.savetxt('mean_cluster_1.txt', mean_cluster_1)

if __name__ == "__main__":
    main()