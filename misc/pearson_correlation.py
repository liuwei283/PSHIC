import numpy as np
import matplotlib.pyplot as plt

def remove_outlier_bins(matrix):
    row_sums = np.sum(matrix, axis=1)
    threshold = np.mean(row_sums) - 2 * np.std(row_sums)
    outlier_indices = np.where(row_sums <= threshold)[0]
    filtered_matrix = np.delete(matrix, outlier_indices, axis=0)
    filtered_matrix = np.delete(filtered_matrix, outlier_indices, axis=1)
    print(filtered_matrix.shape, outlier_indices)
    return filtered_matrix, outlier_indices

def add_outlier_bins(matrix, outlier_indices):
    num_bins = matrix.shape[0] + len(outlier_indices)
    new_matrix = np.zeros((num_bins, num_bins))
    
    current_index = 0
    for i in range(num_bins):
        if i in outlier_indices:
            continue
        new_row = matrix[current_index]
        for index in sorted(outlier_indices):
            new_row = np.insert(new_row, index, 0)
        new_matrix[i] = new_row
        current_index += 1
   
    return new_matrix

def distance_normalize(matrix):
    """
    Normalize Hi-C matrix by expected value at each genomic distance.
    Return observed/expected matrix.
    """
    n = matrix.shape[0]
    expected = np.zeros(n)
    counts = np.zeros(n)

    for i in range(n):
        for j in range(n - i):
            if matrix[j, j + i] > 0:
                expected[i] += matrix[j, j + i]
                counts[i] += 1
    expected /= np.maximum(counts, 1)

    # Normalize matrix by expected
    norm_matrix = np.copy(matrix)
    for i in range(n):
        for j in range(n):
            distance = abs(i - j)
            if expected[distance] > 0:
                norm_matrix[i, j] = matrix[i, j] / expected[distance]
            else:
                norm_matrix[i, j] = 0
    return norm_matrix


def compute_pearson_correlation_matrix(matrix, output_file=None, log_transform=True, mask_diagonal=True):
    """
    Computes the Pearson correlation matrix from a Hi-C contact matrix.

    Parameters:
    - matrix_file: path to a .txt or .npy file containing a square contact matrix
    - output_file: optional, path to save the Pearson correlation matrix as .npy
    - log_transform: whether to apply log1p transform before computing correlation
    - mask_diagonal: whether to ignore diagonal elements during z-score normalization

    Returns:
    - pearson_corr: the resulting NxN Pearson correlation matrix
    """

    # Optional: log-transform to reduce skewness
    if log_transform:
        matrix = np.log1p(matrix)

    # Optional: mask the diagonal
    if mask_diagonal:
        np.fill_diagonal(matrix, np.nan)

    # Row-wise z-score normalization
    row_mean = np.nanmean(matrix, axis=1, keepdims=True)
    row_std = np.nanstd(matrix, axis=1, keepdims=True)
    matrix_zscore = (matrix - row_mean) / row_std
    matrix_zscore[np.isnan(matrix_zscore)] = 0  # replace nan with 0

    # Pearson correlation = dot product of z-scored rows
    pearson_corr = np.dot(matrix_zscore, matrix_zscore.T) / matrix.shape[1]

    return pearson_corr


if __name__ == "__main__":
    dataset_dir = "inv_hinge"
    maternal = np.loadtxt(f"{dataset_dir}/mean_cluster_0.txt")
    paternal = np.loadtxt(f"{dataset_dir}/mean_cluster_1.txt")

    paternal, outlier_paternal = remove_outlier_bins(paternal)
    maternal, outlier_maternal = remove_outlier_bins(maternal)

    # paternal = distance_normalize(paternal)
    # maternal = distance_normalize(maternal)

    maternal_pearson = compute_pearson_correlation_matrix(
        matrix=maternal,
        log_transform=True,
        mask_diagonal=True
    )

    paternal_pearson = compute_pearson_correlation_matrix(
        matrix=paternal,
        log_transform=True,
        mask_diagonal=True
    )

    paternal_pearson = add_outlier_bins(paternal_pearson, outlier_paternal)
    maternal_pearson = add_outlier_bins(maternal_pearson, outlier_maternal)

    # Visualize the Pearson correlation matrices
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot maternal Pearson correlation matrix
    im1 = ax1.imshow(maternal_pearson, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_title('Maternal Pearson Correlation Matrix')
    ax1.set_xlabel('Genomic Bins')
    ax1.set_ylabel('Genomic Bins')
    plt.colorbar(im1, ax=ax1, label='Pearson Correlation')
    
    # Plot paternal Pearson correlation matrix
    im2 = ax2.imshow(paternal_pearson, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_title('Paternal Pearson Correlation Matrix')
    ax2.set_xlabel('Genomic Bins')
    ax2.set_ylabel('Genomic Bins')
    plt.colorbar(im2, ax=ax2, label='Pearson Correlation')
    
    plt.tight_layout()
    # plt.savefig('pearson_correlation_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

    np.savetxt(f"{dataset_dir}/maternal_pearson.txt", maternal_pearson)
    np.savetxt(f"{dataset_dir}/paternal_pearson.txt", paternal_pearson)
