#! /usr/bin/env python
"""
knighRuizAlg is an implementation of the matrix balancing algorithm developed by Knight and Ruiz.
The goal is to take a matrix A and find a vector x such that, diag(x)*A*diag(x) returns a doubly stochastic matrix.
"""
import sys, argparse
import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt

def get_matrix(matrix, result):
    new_matrix = np.copy(matrix)
    length = len(matrix)
    for i in range(length):
        for j in range(i, length, 1):
            if not (np.isnan(result[i])) and not (np.isnan(result[j])):
                new_matrix[i, j] = matrix[i, j] / (result[i] * result[j])
                new_matrix[j, i] = new_matrix[i, j]
    return new_matrix

def get_matrix_asym(matrix, result):
    new_matrix = np.copy(matrix)
    length_r = matrix.shape[0]
    length_c = matrix.shape[1]
    result_r = result[:length_r]
    result_c = result[length_r:]
    for i in range(length_r):
        for j in range(length_c):
            if (not (np.isnan(result_r[i])) and not (np.isnan(result_c[j]))):
                new_matrix[i, j] = matrix[i, j] / (result_r[i]*result_c[j])
    return new_matrix

class Compute_KR():
    def __init__(self, matrix, asy=False, verbose=False, Onenorm=False):
        self.matrix = matrix
        self.length = len(self.matrix)
        self.asy = asy
        self.valid_thre = None
        self.verbose = verbose
        self.norm = Onenorm

    def run(self):
        norm = self.computeKR()
        if self.verbose:
            print("norm", norm)
        factor = self.getSumFactor(norm)
        if self.verbose:
            print("factor", factor)
        if self.norm:
            factor = 1
        return np.asarray(norm) * factor

    def computeKR(self):
        recalculate = True
        offset = self.getOffset(0.0)
        iteration = 1
        max_iter = 60
        while recalculate and iteration <= max_iter:
            sparsematrix = self.populateMatrix(self.matrix, offset)  # delete the rows/cols according to offset
            new_size = len(sparsematrix)
            if self.verbose:
                print("iteration = {}, total_bin = {}, left_bin = {}".format(iteration, self.length, new_size))
            x0 = [1.0] * new_size
            x0 = self.computeKRNormVector(sparsematrix, 1e-6, x0, 0.1)
            recalculate = False
            rowsTossed = 0
            if x0 is None or iteration == max_iter-1:
                recalculate = True
                if iteration < max_iter-1:    #x0 is None: not converse within 100 times
                    offset = self.getOffset(iteration)  # increase the threshold to delete more data
                else:   #iteration==max_iter-1
                    offset = self.getOffset(max_iter+5)
                    if self.verbose:
                        print(iteration, "Changed threshold to 25%.")
            else:   #x0 is valid
                kr = np.zeros(self.length)
                for i in range(self.length):
                    if (offset[i] == -1):
                        kr[i] = np.nan
                    else:
                        kr[i] = 1.0 / x0[offset[i]]  # transform the factor
                mySum = self.getSumFactor(kr)  # an overall factor to the whole matrix
                index = 0
                if self.valid_thre == None:
                    if self.asy:
                        test = kr * mySum
                        self.valid_thre = np.min(test[np.where(np.isnan(test)==False)])
                    else:
                        self.valid_thre = 0.01
                if self.verbose:
                    print(self.valid_thre, "valid_threshold")
                for i in range(self.length):
                    if kr[i] * mySum < self.valid_thre:  # factor is valid but too little
                        offset[i] = -1  # delete the row and calculate again
                        rowsTossed += 1
                        recalculate = True
                    elif (offset[i] != -1):  # valid
                        offset[i] = index
                        index += 1
            if self.verbose:
                print("Finish Iteration = {}, recalculate: {}".format(iteration, recalculate))
            iteration += 1
        if iteration > max_iter and recalculate:
            if self.verbose:
                print("Iteration > {}, output all nan.".format(max_iter))
            kr = [np.nan] * self.length
        return kr

    def getOffset(self, percent=0.0):
        """
        set rows/cols that cannot reach the threshold to -1 in offset list
        :param matrix: input matrix
        :return: offset list (all deleted rows are set to -1)
        """
        rowSums = np.zeros(self.length)
        for i in range(self.length):
            rowSums[i] = np.sum(self.matrix[i,])
        thresh = 0.0
        if (percent != 0.0):
            no_zeros = np.where(rowSums != 0)
            posRowSums = np.sort(rowSums[no_zeros])
            thresh = posRowSums[int(len(posRowSums) * percent / 100)]
        if self.verbose:
            print(thresh)
        offset = [0] * self.length
        index = 0
        for i in range(self.length):
            if (np.sum(self.matrix[i,]) <= thresh):
                offset[i] = -1
            else:
                offset[i] = index
                index += 1
        return offset

    def populateMatrix(self, matrix, offset):
        dele_row = np.where(np.asarray(offset) == -1)[0]
        matrix = np.delete(matrix, dele_row, axis=0)
        matrix = np.delete(matrix, dele_row, axis=1)
        return matrix

    def computeKRNormVector(self, matrix, tol, x0, delta=0.1, Delta=3):
        """
            A balancing algorithm for symmetric matrics, which attempts to find a vector x0 such that
            diag(x0)*A*diag(x0) is close to doubly stochastic.

            :param matrix: input matrix, must be symmetric and nonnegative.
            :param tol: error tolerance
            :param x0: initial guess, default: ones
            :param delta/Delta: how close/far balancing vectors can get to/from the edge of the positive cone.
            :return: balancing vector x0.
	    """
        n = len(x0)  # No. selected rows
        e = [1.0] * n   #initialization
        # Inner stopping criterion parameters
        g = 0.9
        etamax = 0.1
        eta = etamax
        rt = pow(tol, 2)  # default: 1e-12
        v = x0 * (matrix.dot(x0))  # shape = (509,)
        rk = 1.0 - v
        rho_km1 = rk.transpose().dot(rk)  # a scaler
        rout = rold = rho_km1
        MVP = 0  # We will count matrix vector products
        not_changing = 0  # Outer iteration count
        while (rout > rt and not_changing < 100):
            k = 0
            y = np.copy(e)
            rho_km2 = rho_km1
            innertol = max(pow(eta, 2) * rout, rt)
            while (rho_km1 > innertol):  # inner iteration by CG
                k += 1
                if (k == 1):
                    Z = rk / v
                    p = np.copy(Z)
                    rho_km1 = rk.transpose().dot(Z)
                else:
                    beta = rho_km1 / rho_km2
                    p = Z + beta * p
                if k > 10:
                    break
                # update search direction efficiently.
                tmp = matrix.dot(x0 * p)
                w = x0 * tmp + v * p
                alpha = rho_km1 / (p.transpose().dot(w))
                ynew = y + alpha * p
                # test distance to boundary of cone
                minynew = np.amin(ynew)
                if (minynew <= delta):
                    if (delta == 0):
                        break
                    gamma = np.inf
                    for i in range(len(ynew)):
                        if (alpha * p[i] < 0 and (delta - y[i]) / (alpha * p[i]) < gamma):
                            gamma = (delta - y[i]) / (alpha * p[i])
                    y += gamma * alpha * p
                    break
                maxynew = np.amax(ynew)
                if (maxynew >= Delta):
                    gamma = np.inf
                    for i in range(len(ynew)):
                        if (ynew[i] > Delta and (Delta - y[i]) / (alpha * p[i]) < gamma):
                            gamma = (Delta - y[i]) / (alpha * p[i])
                    y += gamma * alpha * p
                    break
                rho_km2 = rho_km1
                y = np.copy(ynew)
                rk -= alpha * w
                Z = rk / v
                rho_km1 = rk.transpose().dot(Z)
                if self.verbose:
                    print("-----------------")
            x0 = x0 * y
            v = x0 * matrix.dot(x0)
            rk = 1.0 - v
            rho_km1 = rk.transpose().dot(rk)
            # print(rho_km1, rout, abs(rho_km1 - rout))
            if (abs(rho_km1 - rout) < 1e-6 or np.isinf(rho_km1)):
                not_changing += 1
            rout = rho_km1
            MVP += k + 1
            # Update inner iteration stopping criterion
            rat = rout / rold
            rold = rout
            r_norm = np.sqrt(rout)
            eta_o = eta
            eta = g * rat
            if (g * pow(eta_o, 2) > 0.1):
                eta = max(eta, g * pow(eta_o, 2))
            eta = max(min(eta, etamax), 0.5 * tol / r_norm)
            if self.verbose:
                print("not_changing time = " + str(not_changing))
        if (not_changing >= 100):
            return None
        return x0

    def getSumFactor(self, norm_list):
        """
		Use the calculated bin factors to calculate the overall factor to the whole normalized matrix.
	    :param norm_list: the calculated factor list
	    :return: an overall factor to the whole matrix
	    """
        matrix_sum = 0  # raw pixel sum
        norm_sum = 0  # normalized pixel sum
        for i in range(self.length):
            for j in range(i, self.length, 1):
                if (not (np.isnan(norm_list[i])) and not (np.isnan(norm_list[j])) and
                        norm_list[i] > 1e-36 and norm_list[j] > 1e-36):  # only if the factor is larger than 0, valid
                    if i == j:  # pixel on the diagonal
                        norm_sum += self.matrix[i, j] / (norm_list[i] * norm_list[j])
                        matrix_sum += self.matrix[i, j]
                    else:
                        norm_sum += (2 * self.matrix[i, j]) / (norm_list[i] * norm_list[j])
                        matrix_sum += 2 * self.matrix[i, j]
        return np.sqrt(norm_sum / matrix_sum)

def KR_norm(contact_matrix, asym = False, perform_norm = False):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", dest="matrix", required=True, help="matrix")
    # parser.add_argument("-o", dest="output", default=None, help="output file path")
    # parser.add_argument("-a", dest="asym", action="store_true", help="if given, perform KR for asymmetric matrix")
    # parser.add_argument("-p", dest="plot", action="store_true", help="if given, plot the sum of row/col")
    # parser.add_argument("-v", dest="verbose", action="store_true", help="if given, print log")
    # parser.add_argument("-n", dest="norm", action="store_true", help="if given, perform one Norm on the factor.")
    # args = parser.parse_args()

    # if args.output == None:
    #     output = args.matrix.replace(".txt", "") + "_KR.txt"
    # else:
    #     output = args.output
    # test = pd.read_csv(args.matrix, delimiter='\t', header=0, index_col=0).values # np.loadtxt(args_input.hic_matrix) # 
    # valid_bin_start = 16
    # valid_bin_end = test.shape[0] - 1
    # test = test[valid_bin_start:valid_bin_end, valid_bin_start:valid_bin_end] 
    if not asym:   #symmetric mode
        result = Compute_KR(contact_matrix, verbose=False, Onenorm=perform_norm).run()
        # if plot:
        #     np.savetxt(args.matrix+".KRVec.txt", result)
        norm_matrix = get_matrix(contact_matrix, result)
    else:
        norm_matrix, result, new_contact_matrix = KRnorm_sym(contact_matrix, args.verbose)
        # if plot:
        #     plot_sum(get_matrix(new_test, result), output + "_sym")
        # np.savetxt(args.matrix + ".KRVec.txt", result)
        # print(result, len(result),len(np.where(result==0)[0]), len(np.where(result>0)[0]))
    return norm_matrix
    # if plot:
    #     plot_sum(norm_matrix, output)    #test

def KRnorm_sym(test, verbose=False):
    length_r = test.shape[0]
    length_c = test.shape[1]
    new_test = np.zeros((length_r + length_c, length_r + length_c))
    new_test[:length_r, length_r:] = test
    new_test[length_r:, :length_r] = test.T
    # add some noise
    noise_matrix = np.random.normal(loc=1e-5, scale=1e-6, size=(length_r + length_c, length_r + length_c))
    noise_matrix = np.triu(noise_matrix) + np.triu(noise_matrix, 1).T
    # print(noise_matrix, np.min(noise_matrix), np.max(noise_matrix))
    new_test += noise_matrix
    new_test[new_test < 0] = 0
    result = Compute_KR(new_test, True, verbose=verbose).run()
    norm_matrix = get_matrix_asym(test, result)
    return norm_matrix, result, new_test


# if __name__ == "__main__":
#     main()
