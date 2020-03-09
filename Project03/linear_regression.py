'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Qingbo Liu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def add_homogenous_coord(self, A):
        N = A.shape[0]
        if len(A.shape) == 1:
            A = A.copy().reshape(N, 1)
        return np.hstack([A, np.ones([N, 1])])

    def set_params(self): 
        predicted = self.predict(self.slope, self.intercept)
        self.residuals = self.compute_residuals(predicted)
        self.R2 = self.r_squared(predicted)

    def linear_regression(self, ind_vars, dep_var, method='scipy'):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization (linear algebra section only).

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression using the appropriate method.
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor), except
        for self.adj_R2.

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var

        x = self.data.select_data(ind_vars)
        y = self.data.select_data(dep_var)

        if len(ind_vars) == 1: 
            x = x.reshape([x.shape[0], 1])

        if method == 'scipy':
            self.linear_regression_scipy(x, y)
        elif method == 'normal':
            self.linear_regression_normal(x, y)
        elif method == 'qr':
            A = self.add_homogenous_coord(x)
            self.linear_regression_qr(A, y)
        else:
            raise ValueError(f'method {method} not supported')

    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''
        self.A = A 
        self.y = y 

        N, M = A.shape
        A = self.add_homogenous_coord(A)
        c, residuals, _, _ = scipy.linalg.lstsq(A, y)

        self.slope = np.array([c[:-1]]).reshape([M, 1])
        self.intercept = c[-1][0]

        self.set_params()

        return c

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
        Linear regression slope coefficients for each independent var AND the intercept term
        '''
        self.A = A 
        self.y = y 

        N, M = A.shape

        A = self.add_homogenous_coord(A)
        c = np.linalg.inv(A.T @ A) @ A.T @ y

        self.slope = np.array([c[:-1]]).reshape([M, 1])
        self.intercept = c[-1]

        self.set_params()

        return c


    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''
        M = A.shape[1] 
        self.A = np.delete(A, M-1, axis=1)
        self.y = y

        Q, R = self.qr_decomposition(A)
        c = scipy.linalg.solve_triangular(R, Q.T @ y)
        
        single_col = len(A.shape) == 1
        if single_col:
            self.slope = np.array([[c[0]]])
            self.intercept = c[-1] 
        else:
            self.slope = c[:-1] 
            self.intercept = c[-1] 
            
        self.set_params()

        return c

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        
        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''
        N, M = A.shape
        Q = A.copy()

        for i in range(M):
            for j in range(i):
                Q[:, i] -= (A[:, i].T @ Q[:, j]) * Q[:, j]
            Q[:, i] /= np.linalg.norm(Q[:, i])

        R = Q.T @ A 

        return Q, R

    def predict(self, slope, intercept, X=None):
        '''Use fitted linear regression model to predict the values of data matrix `X`.
        Generates the predictions y_pred = mD + b, where (m, b) are the model fit slope and intercept,
        D is the data matrix.

        Parameters:
        -----------
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.
        
        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        data = X if X is not None else self.A
        return data @ slope + intercept
            

    def r_squared(self, y_pred, original_y=None):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model
        original_y: ndarray. shape=(num_data_samps, 1).
            Original dependent variable values

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        y = original_y if original_y is not None else self.y
        residuals = self.compute_residuals(y_pred, y)
        mean = np.mean(y)
        smd = np.linalg.norm(y - mean) ** 2

        R2 = 1 - np.linalg.norm(residuals)**2/smd

        return R2 

    def compute_residuals(self, y_pred, original_y=None):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the 
            data samples
        '''
        y = original_y if original_y is not None else self.y
        return y - y_pred

    def mean_sse(self, X=None, poly=None):
        '''Computes the mean sum-of-squares error in the predicted y compared the actual y values.
        See notebook for equation.

        Parameters:
        -----------
        X: ndarray. shape=(anything, num_ind_vars)
            Data to get regression predictions on.
            If None, get predictions based on data used to fit model.
        poly: int. 
            If set to a number, indicates the degrees of polynomials. 
            

        Returns:
        -----------
        float. Mean sum-of-squares error

        Hint: Make use of self.compute_residuals
        '''
        data = X if X is not None else self.A 
        N = data.shape[0]

        if poly is not None: 
            data = data.reshape(N, 1)
            data = self.make_polynomial_matrix(X, poly)

        y_pred = self.predict(self.slope, self.intercept, data)
        residuals = self.compute_residuals(y_pred)
        msse = (np.linalg.norm(residuals) ** 2) / N

        return msse 


    def scatter(self, ind_var, dep_var, title, ind_var_index=0):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.
        
        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot
        ind_var_index: int. Index of the independent variable in self.slope
            (which regression slope is the right one for the selected independent variable
            being plotted?)
            By default, assuming it is at index 0.

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        x, y = super().scatter(ind_var, dep_var) 
        slope = self.slope[ind_var_index][0] 

        fitted_line_x = np.linspace(np.min(x), np.max(x), 100)
        fitted_line_y = fitted_line_x * slope + self.intercept

        plt.plot(fitted_line_x, fitted_line_y, 'r')
        plt.legend(['regression', 'data'])
        plt.title(f'{title}, R2 = {self.R2: .2f}')

    def scatter_poly(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes polynomial regression has been already run.
        
        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot
        '''
        p = self.A.shape[1] 
        x, y = super().scatter(ind_var, dep_var)
        N = y.shape[0]

        fitted_line_x = np.linspace(np.min(x), np.max(x), 100)
        ATp = self.make_polynomial_matrix(fitted_line_x.reshape([100, 1]), p)
        fitted_line_y = ATp @ self.slope + self.intercept

        Ap = self.make_polynomial_matrix(x.reshape([N, 1]), p)
        y_pred = self.predict(self.slope, self.intercept, Ap)

        r2 = self.r_squared(y_pred, y.reshape([N, 1]))

        plt.plot(fitted_line_x, fitted_line_y, 'r')
        plt.legend(['regression', 'data'])
        plt.title(f'{title}, R2 = {r2: .2f}')


    def pair_plot(self, data_vars, fig_sz=(12, 12)):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        fig, axes = super().pair_plot(data_vars, fig_sz)

        N = len(data_vars)
        data = self.data.select_data(data_vars)
        for i in range(N):
            for j in range(N):
                if not i == j:
                    A = data[:, j]
                    self.linear_regression(data_vars[j], data_vars[i])

                    fitted_x = np.linspace(np.min(A), np.max(A), 100)
                    fitted_y = fitted_x * self.slope[0] + self.intercept

                    axes[i, j].plot(fitted_x, fitted_y, 'r')
                    axes[i, j].set_title(f'R2: {self.R2: .2f}')
                else: # i == j, diagonal  
                    xlabel = axes[i, j].get_xlabel() 
                    ylabel = axes[i, j].get_ylabel() 
                    axes[i, j].clear()
                    axes[i, j].hist(data[:, j])
                    axes[i, j].set_xlabel(xlabel)
                    axes[i, j].set_ylabel(ylabel)

        fig.subplots_adjust(hspace=0.3)
        



    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.
        
        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        will take care of that.
        '''
        m = np.ones([A.shape[0], p])
        y = A[:, 0]

        for i in range(p):
            m[:, i] = y ** (i+1)

        return m 

    def poly_regression(self, ind_var, dep_var, p, method='normal'):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        
        (Week 2)
        
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10
            (The method that you call for the linear regression solver will take care of the intercept)
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization (linear algebra section only).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create the independent variable data matrix (self.A) with columns appropriate for
            polynomial regresssion. Do this with self.make_polynomial_matrix
            - You should programatically generate independent variable name strings based on the
            polynomial degree.
                Example: ['X_p1, X_p2, X_p3'] for a cubic polynomial model
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        for h in [ind_var, dep_var]:
            if h not in self.data.get_headers():
                raise ValueError(f'variable {h} not in headers: {self.data.get_headers()}')

        ind_data = self.data.select_data(ind_var)
        ind_data = ind_data.reshape([ind_data.shape[0], 1])

        y = self.data.select_data(dep_var)
        A = self.make_polynomial_matrix(ind_data, p)



        if method == 'scipy':
            self.linear_regression_scipy(A, y)
        elif method == 'normal':
            self.linear_regression_normal(A, y)
        elif method == 'qr':
            A = self.add_homogenous_coord(A)
            self.linear_regression_qr(A, y)
        else:
            raise ValueError(f'method {method} not supported')


