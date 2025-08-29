import warnings
import sys
import arguments
import time
from cmath import inf
import numpy as np
import math
import torch
import random
from scipy.optimize import curve_fit
from KR_norm_juicer import KR_norm
from sklearn.manifold import MDS

def eigenDecomposition(input_matrix, k):
    """Perform eigen decomposition on the input matrix and return the top k eigenvectors."""
    Lambda, U = np.linalg.eig(input_matrix)
    Lambda = Lambda[:k]
    Lambda = np.sqrt(Lambda)
    inv_U = np.linalg.inv(U)[:k, :]
    dig = np.diag(Lambda)
    half_matrix = np.dot(dig, inv_U)
    return half_matrix.T  # Return row vectors

def calculate_distance_matrix_numpy(matrix):
    """Calculate the pairwise Euclidean distance matrix using numpy."""
    expanded_matrix = matrix[:, np.newaxis, :]
    diff_squared = (expanded_matrix - matrix) ** 2
    pairwise_distances = np.sqrt(np.sum(diff_squared, axis=-1))
    return pairwise_distances

def calculate_distance_matrix_torch(matrix):
    """Calculate the pairwise Euclidean distance matrix using PyTorch."""
    expanded_matrix = matrix.unsqueeze(1)
    diff_squared = (expanded_matrix - matrix) ** 2
    distance_squared = torch.sum(diff_squared, axis=-1)
    eye = torch.eye(distance_squared.size(0), device=distance_squared.device, dtype=distance_squared.dtype)
    distance_squared = distance_squared * (1 - eye) + eye  # Set diagonal to 1
    pairwise_distances = torch.sqrt(distance_squared)
    return pairwise_distances

def power_law(x, b):
    """Define a power law function with exponent -3."""
    return b * (x ** -3)

def power_law_diploid(X, alpha, beta):
    """Define a diploid power law function with parameters alpha and beta."""
    x1, x2 = X
    return beta * (x1 ** -alpha) + beta * (x2 ** -alpha)

def convertToHiC_numpy(vectors, beta, alpha):
    """Convert vectors to a Hi-C matrix using numpy with given alpha and beta."""
    distance_matrix = calculate_distance_matrix_numpy(vectors)
    np.fill_diagonal(distance_matrix, 2)
    hic_matrix = np.power(distance_matrix, -alpha) * beta
    return hic_matrix

def convertToHiC_torch(vectors, beta, alpha):
    """Convert vectors to a Hi-C matrix using PyTorch with given alpha and beta."""
    distance_matrix = calculate_distance_matrix_torch(vectors)
    eye = torch.eye(distance_matrix.size(0), device=distance_matrix.device, dtype=distance_matrix.dtype)
    distance_matrix = distance_matrix * (1 - eye) + eye * 0.5  # Set diagonal to 0.5
    hic_matrix = torch.pow(distance_matrix, -alpha) * beta
    return hic_matrix, distance_matrix

def transform_points(points):
    """Center and scale points to fit within a sphere of radius RADIUS."""
    center = np.mean(points, axis=0)
    centered_points = points - center
    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
    scale_factor = RADIUS / max_distance
    scaled_points = centered_points * scale_factor
    return scaled_points

def initialMatrix(hic_matrix, interval, k, alpha, beta):
    """Initialize matrices for parental embeddings using MDS and noise addition."""
    parental_init = []
    for p in range(2):
        noise = np.full(hic_matrix.shape, interval / (2 + p))
        init_matrix = hic_matrix / 2 + noise
        init_matrix = (init_matrix + init_matrix.T) / 2
        distance_matrix = np.power(beta / (init_matrix + 1), 1 / alpha)
        mds = MDS(n_components=k, dissimilarity='precomputed', random_state=random.randint(0, 10000))
        embedding = transform_points(mds.fit_transform(distance_matrix))
        parental_init.append(embedding)
    return parental_init[0], parental_init[1]

def sphere_loss(matrix, k):
    """Calculate the loss for points outside a sphere of radius RADIUS."""
    norms = torch.norm(matrix, dim=1)
    penalties = torch.relu(norms - RADIUS)
    norm_loss = penalties.pow(2).sum()
    return norm_loss

def poisson_loss(y_true, lambda_pred):
    """Calculate the Poisson loss between true values and predicted lambda."""
    lambda_pred = torch.clamp(lambda_pred, min=1e-7)
    loss = lambda_pred - y_true * torch.log(lambda_pred)
    return loss

def diagonal_difference_loss(dist_m, dist_p, max_deviation=4):
    """Calculate the loss based on the difference between diagonals of two distance matrices."""
    loss = 0.0
    for r in range(1, max_deviation + 1):
        diag_m = torch.diagonal(dist_m, offset=r)
        diag_p = torch.diagonal(dist_p, offset=r)
        proportion_loss = torch.abs(diag_m - diag_p) / ((diag_m + diag_p) / 2)
        loss += torch.norm(proportion_loss) * (1 / (r ** 2))
    loss /= max_deviation
    return loss

def optimize(vP, vM, alpha, beta, hic_matrix, args_input):
    """Optimize the embeddings vP and vM using gradient descent."""
    stop = False
    iter_num = 0
    loss = inf
    optimizer = torch.optim.SGD([
        {'params': [vP, vM], 'lr': args_input.learning_rate},  # Learning rate for vP
        {'params': [beta], 'lr': args_input.learning_rate}  # Learning rate for beta
    ], momentum=args_input.momentum)

    loss_increase_count_max = 10000
    loss_threshold = 0
    loss_increase_count = 0
    prev_loss = inf

    while not stop:
        iter_num += 1
        p_hic, dist_p = convertToHiC_torch(vP, beta, alpha)
        m_hic, dist_m = convertToHiC_torch(vM, beta, alpha)
        hic_loss = hic_matrix - p_hic - m_hic
        size = hic_loss.size(0)
        device = hic_loss.device
        dtype = hic_loss.dtype
        eye = torch.eye(size, device=device, dtype=dtype)
        hic_loss = hic_loss * (1 - eye)
        hic_loss_abs = torch.square(hic_loss)
        hic_loss_log1p = torch.log1p(hic_loss_abs)
        hic_loss = torch.norm(hic_loss_log1p, p=2)

        if prev_loss - hic_loss < loss_threshold:
            loss_increase_count += 1
            if loss_increase_count > loss_increase_count_max:
                stop = True

        prev_loss = hic_loss
        space_loss = sphere_loss(vP, args_input.dimension) + sphere_loss(vM, args_input.dimension)
        dd_loss = args_input.dd_loss_weight * diagonal_difference_loss(dist_p, dist_m)
        loss = hic_loss + 10 * space_loss + dd_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(beta.item(), hic_loss.item(), space_loss.item(), dd_loss.item(), loss_increase_count)
        if iter_num > 50000:
            stop = True

    return vP, vM, loss, beta

def estimate_alpha(vP, vM, hic_matrix, alpha, beta):
    """Estimate the alpha and beta parameters using curve fitting."""
    distances_p = calculate_distance_matrix_numpy(vP)
    distances_m = calculate_distance_matrix_numpy(vM)
    num_bins = hic_matrix.shape[0]
    outlier_num = 4 * num_bins
    flat_indices = np.argpartition(hic_matrix.flatten(), -outlier_num)[-outlier_num:]
    mask = np.zeros(hic_matrix.shape, dtype=bool)
    row_indices, col_indices = np.unravel_index(flat_indices, hic_matrix.shape)
    mask[row_indices, col_indices] = True
    eye = np.eye(num_bins, dtype=bool)
    mask = ~(mask | eye)
    combined_mask = mask
    x1 = distances_p[combined_mask]
    x2 = distances_m[combined_mask]
    input_X = np.vstack((x1, x2))
    y = hic_matrix[combined_mask]
    init_guess = [alpha, beta]
    popt, pcov = curve_fit(power_law_diploid, input_X, y, p0=init_guess, maxfev=5000)
    alpha = float(format(popt[0], '.5f'))
    beta = float(format(popt[1], '.5f'))
    print(f"Estimated power law parameters: {alpha}, {beta}")
    assert beta > 0
    assert alpha > 0
    return alpha, beta

def estimate_beta(hic_matrix, k):
    """Estimate the beta parameter using curve fitting on the Hi-C matrix."""
    hic_matrix /= 2
    max_distance = 2 * math.sqrt(k) * RADIUS
    min_distance = max_distance / hic_matrix.shape[0]
    num_bins = hic_matrix.shape[0]
    contact_avg = [np.mean(np.diagonal(hic_matrix, offset=i)) for i in range(1, num_bins)]
    distances = np.linspace(min_distance, max_distance, num_bins - 1, endpoint=False)
    init_guess = 1
    popt, _ = curve_fit(power_law, distances, contact_avg, p0=init_guess, maxfev=5000)
    beta = float(format(popt[0], '.3f'))
    print(f"Estimated power law parameter beta: {beta}")
    assert beta > 0
    return beta

def remove_outlier_bins(matrix, args_input):
    """Remove outlier bins from the matrix based on a threshold."""
    row_sums = np.sum(matrix, axis=1)
    threshold = np.mean(row_sums) - args_input.outlier_threshold * np.std(row_sums)
    outlier_indices = np.where(row_sums <= threshold)[0]
    filtered_matrix = np.delete(matrix, outlier_indices, axis=0)
    filtered_matrix = np.delete(filtered_matrix, outlier_indices, axis=1)
    print(filtered_matrix.shape, outlier_indices)
    return filtered_matrix, outlier_indices

def add_outlier_bins(matrix, outlier_indices):
    """Add outlier bins back into the matrix at specified indices."""
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

def main(args_input):
    """Main function to execute the Hi-C phase analysis."""
    global RADIUS
    RADIUS = args_input.radius
    print("task name: ", args_input.task_name)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    print("Reading inputs...")
    hic_matrix_numpy = np.loadtxt(args_input.hic_matrix)
    down_scale_factor = np.max(hic_matrix_numpy) / args_input.max_depth
    hic_matrix_numpy /= down_scale_factor
    hic_matrix_numpy, outlier_indices = remove_outlier_bins(hic_matrix_numpy, args_input)
    hic_matrix_numpy = KR_norm(hic_matrix_numpy, perform_norm=False)
    alpha = args_input.alpha
    beta = 1.0
    vP, vM = initialMatrix(hic_matrix_numpy, args_input.noise, args_input.dimension, alpha, beta)
    hic_matrix = torch.tensor(hic_matrix_numpy)
    vP = torch.tensor(vP.astype(np.float64), requires_grad=True)
    vM = torch.tensor(vM.astype(np.float64), requires_grad=True)
    print("Starting optimization...")
    power_law_iteration_num = 5
    iteration_count = 0
    stop = False
    pre_alpha = alpha
    while not stop:
        beta = torch.tensor(beta, requires_grad=True)
        vP, vM, loss, beta = optimize(vP, vM, alpha, beta, hic_matrix, args_input)
        beta_item = beta.detach().numpy().item()
        alpha, beta = estimate_alpha(vP.detach().numpy(), vM.detach().numpy(), hic_matrix_numpy, alpha, beta_item)
        vP_np = vP.detach().numpy()
        vM_np = vM.detach().numpy()
        p_hic = add_outlier_bins(convertToHiC_numpy(vP_np, beta_item, alpha) * down_scale_factor, outlier_indices)
        m_hic = add_outlier_bins(convertToHiC_numpy(vM_np, beta_item, alpha) * down_scale_factor, outlier_indices)
        np.savetxt(f"./{args_input.output_dir}/P_{args_input.task_name}_{iteration_count}.txt", p_hic)
        np.savetxt(f"./{args_input.output_dir}/M_{args_input.task_name}_{iteration_count}.txt", m_hic)
        np.savetxt(f"./{args_input.output_dir}/P_vector_{args_input.task_name}_{iteration_count}.txt", vP_np)
        np.savetxt(f"./{args_input.output_dir}/M_vector_{args_input.task_name}_{iteration_count}.txt", vM_np)
        if abs(pre_alpha - alpha) < 0.1 or iteration_count >= power_law_iteration_num:
            stop = True
        pre_alpha = alpha
        iteration_count += 1
    print("Final loss:")
    print(loss.detach().numpy())

if __name__ == "__main__":
    start_time = time.time()
    args_input = arguments.get_args()
    main(args_input)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total running time: {total_time} seconds")