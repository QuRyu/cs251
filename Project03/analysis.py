'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Qingbo Liu
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        return np.min(self.data.select_data(headers, rows), axis=0)

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        return np.max(self.data.select_data(headers, rows), axis=0)

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        m = self.data.select_data(headers, rows)
        return (np.min(m, axis=0), np.max(m, axis=0))


    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: Loops are forbidden!
        '''
        data = self.data.select_data(headers, rows)
        return np.sum(data, axis=0) / (np.ones(data.shape[1])*data.shape[0])

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: Loops are forbidden!
        '''
        data = self.data.select_data(headers, rows)
        N = data.shape[0]
        mean = self.mean(headers, rows)
        diff = data - mean 
        return np.sum(diff*diff, axis=0)*(1/(N-1))
        

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: Loops are forbidden!
        '''
        return np.sqrt(self.var(headers, rows))

    def percentiles(self, headers, rows=[]):
        '''Computes 25th, median, and 75th percentiles for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        percentiles: ndarray. shape=(3, len(headers))
            25th, median, and 75th percentiles for each of the selected header variables
        '''
        data = self.data.select_data(headers, rows) 
        return np.percentile(data, [25, 50, 75], axis=0)

    def mode(self, headers, rows=[]):
        '''Computes mode (numbers that appear most often) for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        modes: ndarray. shape=(len(headers),)
            Mode for each of the selected header variables
        '''
        return stats.mode(self.data.select_data(headers, rows), axis=0)

    def skew(self, headers, rows=[]):
        '''Computes skewness  for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        skewness: ndarray. shape=(len(headers),)
            Mode for each of the selected header variables
        '''
        return stats.skew(self.data.select_data(headers, rows), axis=0)

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title='', marker='.', fig_sz=(6, 5), **kwargs):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot
        **kwargs:
            Optional keyword arguments passed to Axes

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        data = self.data.select_data([ind_var, dep_var])
        x = data[:, 0]
        y = data[:, 1]
        fig, plot = plt.subplots(figsize=fig_sz, subplot_kw=kwargs)
        fig.suptitle(title)
        plot.scatter(x, y, marker = marker)
        plot.set_xlabel(ind_var)
        plot.set_ylabel(dep_var)
        
        return (x, y)

    def pair_plot(self, data_vars, fig_sz=(12, 12), title='', marker='.', color='b'):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey optional parameters of plt.subplots
        '''

        M = len(data_vars)
        data = self.data.select_data(data_vars)
        fig, subplots = plt.subplots(nrows=M, ncols=M,
                sharex='col', sharey='row', figsize=fig_sz)

        fig.suptitle(title)

        for i in range(M):
            for j in range(M):
                subplots[i, j].scatter(data[:, j], data[:, i])

                if i == M-1: # last row 
                    subplots[i, j].set_xlabel(data_vars[j])
                if j == 0: # first column 
                    subplots[i, j].set_ylabel(data_vars[i])

        return (fig, subplots)

    def bar(self, var, title='', fig_sz=(12, 12), **kwargs):
        '''Creates a bar chart with "var" variable in the dataset `var`. `var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        var: str. 
            Name of variable that is plotted in the bar chart.
        title: str.
            Title of the bar chart.
        **kwargs:
            Optional keyword arguments passed to Axes

        Returns:
        -----------
        labels: ndarray. shape=(len(unique_enum_values),)
            List of enum values being plotted. 
        counts: ndarray. shape=(len(unique_enum_values),)
            Counts of each enum values.
        '''
        raw_data = self.data.select_data(var, data_type='enum')
        enum_mapping = self.data.get_enum_mappings(var)

        elements, counts = np.unique(raw_data, return_counts=True)
        labels = np.vectorize(lambda x: enum_mapping[x])(elements)

        fig, plot = plt.subplots(figsize=fig_sz, subplot_kw=kwargs)
        plot.set_title(title)

        plot.bar(range(len(counts)), counts, tick_label=labels)

        return (labels, counts) 

    def boxplot(self, data_vars, title='', **kwargs):
        '''Creates a box and whisker plot  with "var" variables in the dataset `vars`. `var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        vars: Python list of str.
            Names of variables that is plotted in histogram.
        title: str.
            Title of the histogram.
        **kwargs:
            Optional keyword arguments passed to Axes

        Returns:
        -----------
        result: dict. 
            A dictionary mapping each component of the boxplot to a list of 
            the matplotlib.lines.Line2D instances created. 
        '''

        data = self.data.select_data(data_vars)

        _, plot = plt.subplots(subplot_kw=kwargs)
        plot.set_title(title)

        result = plot.boxplot(data, labels=data_vars)
        return result

