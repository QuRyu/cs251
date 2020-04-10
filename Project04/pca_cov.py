'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Qingbo Liu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset.

        (No changes should be needed)
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of `data`

        NOTE: You should do this wihout any loops
        '''
        N = data.shape[0]
        mean = np.mean(data, axis=0) 
        A_centered = data - mean 
        cov = (A_centered.T @ A_centered) / (N-1)

        return cov 

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        self.e_vals = e_vals

        total = np.sum(e_vals)
        result = [] 
        for i in e_vals:
            result.append(i/total)
        return result 

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        v_sum = 0 
        result = [] 
        for i in prop_var:
            v_sum += i 
            result.append(v_sum)
        return result

    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''
        self.A = A = self.data[vars].values 
        self.vars = vars 

        if normalize:
            mins, maxes = np.min(A, axis=0), np.max(A, axis=0)
            A = (A - mins) / (maxes - mins)
        
        # center the data 
        A = A - np.mean(A, axis=0)

        cov = self.covariance_matrix(A)
        e_val, e_vec = np.linalg.eig(cov)

        
        e = list(zip(e_val, e_vec))
        e.sort(key=lambda x:x[0], reverse=True)
        e_val, e_vec = np.array([val[0] for val in e]), np.array([val[1] for val in e])
        self.prop_var = self.compute_prop_var(e_val)
        self.cum_var = self.compute_cum_var(self.prop_var)

        self.e_vals = e_val
        self.e_vecs = e_vec

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        N = num_pcs_to_keep if num_pcs_to_keep is not None else self.e_vals.shape[0]

        x = np.linspace(0, N, N+1)
        y = self.cum_var.copy()[:N]
        y.insert(0, 0)

        fig, ax = plt.subplots()
        # ax.plot(x, y, marker='o')
        ax.plot(x, y)
        ax.set_xlabel('PC')
        ax.set_ylabel('Cumulative Variance')
        ax.set_label('Elbow plot')
        ax.set_ylim((0, 1.05))

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        e_vecs = self.e_vecs[:, pcs_to_keep]
        self.A_proj = self.A @ e_vecs

        return self.A_proj

    def loading_plot(self):
        '''Create a loading plot of the top 2 PC eigenvectors

        TODO:
        - Plot a line joining the origin (0, 0) and corresponding components of the top 2 PC eigenvectors.
            Example: If e_1 = [0.1, 0.3] and e_2 = [1.0, 2.0], you would create two lines to join
            (0, 0) and (0.1, 1.0); (0, 0) and (0.3, 2.0).
            Number of lines = num_vars
        - Use plt.annotate to label each line by the variable that it corresponds to.
        - Reminder to create useful x and y axis labels.

        NOTE: Don't write plt.show() in this method
        '''
        vecs = self.e_vecs[:, [1, 2]]
        x_min, y_min = np.min(vecs, axis=0)
        x_max, y_max = np.max(vecs, axis=0)

        lim_f = lambda a, b: abs(a) if abs(a) > abs(b) else abs(b)
        y_lim = lim_f(y_min, y_max)
        x_lim = lim_f(x_min, x_max)

        fig, ax = plt.subplots()

        for i, (x, y) in enumerate(vecs):
            x_start, x_end = min(0, x), max(0, x)
            y_start, y_end = min(0, y), max(0, y)
            m = (y_end - y_start)/(x_end - x_start)

            line_x = np.linspace(x_start, x_end, 100)
            line_y = line_x * m 
            ax.plot(line_x, line_y)
            ax.annotate(self.vars[i], xy=(x, y))

        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        '''
        e_vecs = self.e_vecs[:, [i for i in range(top_k)]]

        A_proj = self.A @ e_vecs
        A_back = A_proj @ e_vecs.T

        return A_back 


