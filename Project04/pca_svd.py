'''pca_svd.py
Subclass of PCA_COV that performs PCA using the singular value decomposition (SVD)
Qingbo Liu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np

import pca_cov


class PCA_SVD(pca_cov.PCA_COV):
    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars` using SVD

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        TODO:
        - This method should mirror that in pca_cov.py (same instance variables variables need to
        be computed).
        - There should NOT be any covariance matrix calculation here!
        - You may use np.linalg.svd to perform the singular value decomposition.
        '''
        self.A = A = self.data[vars].values 
        self.vars = vars 

        if normalize:
            mins, maxes = np.min(A, axis=0), np.max(A, axis=0)
            A = (A - mins) / (maxes - mins)

        # center the data  
        A = A - np.mean(A, axis=0)
        
        u, s, v = np.linalg.svd(A)
        
        e_vecs = v.T

        N, M = A.shape

        e_vals = []
        if not M > N: 
            for i in range(M):
                e_vals.append((s[i] * s[i]) / (N-1))
        else: # M > N 
            for i in range(N):
                e_vals.append((s[i] * s[i]) / (N-1))

        e_vals = np.array(e_vals)


        # e = list(zip(e_vals, e_vecs))
        # e.sort(key=lambda x:x[0], reverse=True)
        # e_vals, e_vecs = np.array([val[0] for val in e]), np.array([val[1] for val in e])

        self.e_vals = e_vals 
        self.e_vecs = e_vecs
        self.prop_var = self.compute_prop_var(e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)


