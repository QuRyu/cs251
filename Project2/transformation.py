'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import analysis
import data
from palettable import colorbrewer 


class Transformation(analysis.Analysis):

    def __init__(self, data_orig, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        data_orig: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables
            — `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variables for `data_orig`.
        '''
        self.data_orig = data_orig
        super().__init__(data)

    def project(self, headers):
        '''Project the data on the list of data variables specified by `headers` — i.e. select a
        subset of the variables from the original dataset. In other words, populate the instance
        variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list dictates the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables are optional.

        HINT: Update self.data with a new Data object and fill in appropriate optional parameters
        (except for `filepath`)

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables).
        - Make sure that you create 'valid' values for all the `Data` constructor optional parameters
        (except you dont need `filepath` because it is not relevant).
        '''
        self.ndim = len(headers)
        enum_mappings = self.data_orig.get_mappings(data_type='enum')
        numeric_mappings = self.data_orig.get_mappings(data_type='numeric')

        project_data = [] 
        for h in headers:
            if h in enum_mappings: 
                project_data.append(self.data_orig.select_data(h, data_type='enum').copy())
            elif h in numeric_mappings:
                project_data.append(self.data_orig.select_data(h, data_type='numeric').copy())
            else:
                raise ValueError(f'header {h} not defined in dataset')
        project_data = np.hstack(project_data)

        header2col = {h: i for i, h in enumerate(headers)}
        types = ['numeric' for _ in range(len(headers))]

        self.data = data.Data(headers=headers, types=types, data=project_data, header2col=header2col)

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        return np.hstack((self.data.get_all_data(), np.ones((self.data.get_num_samples(), 1))))

    def drop_homogeneous_coord(self, m): 
        return np.delete(m, self.data.get_num_dims(), 1)

    def translation_matrix(self, headers, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be translated.
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        if not len(headers) == len(magnitudes):
            raise ValueError(f'length of headers {headers} should be equal to length of magnitudes {magnitudes}')

        data_headers = self.data.get_headers()

        N = self.data.get_num_dims()
        M = np.eye(N+1)
        header2col = self.data.get_mappings()
        for i, h in enumerate(headers): 
            if h not in header2col:
                raise ValueError(f'header {header} not defined in headers {self.data.get_headers()}')
            else:
                M[header2col[h], N] = magnitudes[i]

        return M 


    def scale_matrix(self, headers, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be scaled.
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        if not len(headers) == len(magnitudes):
            raise ValueError(f'length of headers {headers} should be equal to length of magnitudes {magnitudes}')

        data_headers = self.data.get_headers()

        N = self.data.get_num_dims()
        M = np.eye(N+1)
        header2col = self.data.get_mappings()
        for i, h in enumerate(headers): 
            if h not in header2col:
                raise ValueError(f'header {header} not defined in headers {self.data.get_headers()}')
            else:
                idx = header2col[h]
                M[idx, idx] = magnitudes[i]

        return M 

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data about the ONE
        axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''

        M = np.eye(4)
        header2col = self.data.get_mappings()

        angel = np.radians(degrees)

        if len(header2col) != 3:
            raise ValueError(f'three variables should be projected but actually {len(header2col)}')
        elif header not in header2col:
            raise ValueError(f'header {header} should be one of headers {self.data.get_headers()}')

        if header2col[header] == 0:
            M[1, 1] = np.cos(angel)
            M[1, 2] = -np.sin(angel)
            M[2, 1] = np.sin(angel)
            M[2, 2] = np.cos(angel)
        elif header2col[header] == 1: 
            M[0, 0] = np.cos(angel)
            M[0, 2] = np.sin(angel)
            M[2, 0] = -np.sin(angel)
            M[2, 2] = np.cos(angel)
        elif header2col[header] == 2: 
            M[0, 0] = np.cos(angel)
            M[0, 1] = -np.sin(angel)
            M[1, 0] = np.sin(angel)
            M[1, 1] = np.cos(angel)

        return M 

    def rotation_matrix_2d(self, degrees): 
        '''Make an 2-D homogeneous rotation matrix for rotating the projected data about the ONE
        axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(3, 3). The 2D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        M = np.eye(3)
        header2col = self.data.get_mappings()
        
        angel = np.radians(degrees)

        if len(header2col) != 2:
            raise ValueError(f'two variables should be projected but actually {len(header2col)}')
        
        M[0, 0] = np.cos(angel)
        M[0, 1] = -np.sin(angel)
        M[1, 0] = np.sin(angel)
        M[1, 1] = np.cos(angel)

        return M 


    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected dataset after it has been transformed by `C`
        '''
        return (C @ self.get_data_homogeneous().T).T
        
    def update_data(self, new_data): 
        return data.Data(data=new_data, headers=self.data.headers, header2col=self.data.header2col)

    def translate(self, headers, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be translated.
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        M = self.translation_matrix(headers, magnitudes)
        result = self.drop_homogeneous_coord(self.transform(M))
        self.data = self.update_data(result) 

        return result 

    def scale(self, headers, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be scaled.
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        M = self.scale_matrix(headers, magnitudes)
        result = self.drop_homogeneous_coord(self.transform(M))
        self.data = self.update_data(result) 

        return result 

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        M = self.rotation_matrix_3d(header, degrees)
        result = self.drop_homogeneous_coord(self.transform(M))
        self.data = self.update_data(result) 

        return result 

    def rotate_2d(self, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        '''
        M = self.rotation_matrix_2d(degrees)
        result = self.drop_homogeneous_coord(self.transform(M))
        self.data = self.update_data(result) 

        return result 

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        headers = self.data.get_headers()
        data = self.data.select_data(headers) 

        global_min = np.min(data)
        global_max = np.max(data)

        data = (data - global_min) / (global_max - global_min)
        self.data = self.update_data(data)

        return data 

    def normalize_together_zscore(self):
        '''Normalize all variables in the projected dataset together by translating the global mean
        (across all variables) to zero and scaling the global standard deviation (across all variables) to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        headers = self.data.get_headers()
        data = self.data.data 
        std = np.sqrt(np.var(data))
        mean = np.mean(data)

        data = (data - mean)/std
        self.data = self.update_data(data)

        return data  

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        headers = self.data.get_headers()
        mins, maxes = self.range(self.data.get_headers())
        data = self.data.select_data(headers)

        data = (data - mins) / (maxes - mins)
        self.data = self.update_data(data)
        
        return data 

    def normalize_separately_zscore(self):
        '''Normalize each variable separately by translating its local mean and scaling
        by one over its local standard deviation. 

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        headers = self.data.get_headers()
        data = self.data.data 
        stds = self.std(headers)
        means = self.mean(headers)

        data = (data - means)/stds 
        self.data = self.update_data(data)

        return data  

    def scatter_color(self, ind_var, dep_var, c_var, title=None, z_var=None, size_var=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Section B (Linear Algebra): Use a ColorBrewer color palette (e.g. from the
            `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        headers = self.data.get_headers()
        for h in [ind_var, dep_var, c_var]:
            if h not in headers:
                raise ValueError(f'argument header {h} not in headers {headers}')

        fig, ax = plt.subplots()

        if title is not None: 
            ax.set_title(title)
        ax.set_xlabel(ind_var)
        ax.set_ylabel(dep_var)
        
        ind_data = self.data.select_data(ind_var)
        dep_data = self.data.select_data(dep_var)
        c_data = self.data.select_data(c_var)
        
        pos = ax.scatter(ind_data, dep_data, c=c_data,
                cmap=colorbrewer.sequential.Greys_5.mpl_colormap)
        bar = fig.colorbar(pos, ax=ax)
        bar.set_label(c_var)

    def scatter_color_3D(self, ind_var, dep_var, z_var, c_var=None, size_var=None, title=None):
        '''Creates a 3D scatter plot with a color scale representing the 4th
        dimension and marker size representing the 5th dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        z_var: Header of the variable that will be plotted along the Z axis.
        c_var: Header of the variable that will be plotted along the color axis.
        size_var: Header of the variable that will be plotted to change marker size.
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        headers = self.data.get_headers()
        for h in [ind_var, dep_var, z_var, c_var, size_var]:
            if not h is None and h not in headers:
                raise ValueError(f'argument header {h} not in headers {headers}')

        fig= plt.figure()
        ax = Axes3D(fig)

        if title is not None: 
            ax.set_title(title)

        ind_data = self.data.select_data(ind_var)
        dep_data = self.data.select_data(dep_var)
        z_data = self.data.select_data(z_var)
        c_data = None if c_var is None else np.squeeze(self.data.select_data(c_var))
        size_data = None if size_var is None else self.data.select_data(size_var) ** 2 

        pos = ax.scatter(ind_data, dep_data, z_data, c=c_data, s=size_data,
                cmap=colorbrewer.sequential.Greys_5.mpl_colormap)
        if c_var is not None: 
            bar = fig.colorbar(pos, ax=ax)
            bar.set_label(c_var)

    def whiten(self):
        '''Normalize a group of observations on a per feature basis

        Returns
        -------
        ndarray. shape=(M, N)
            Contains the values in scaled by the 1/standard deviation
            of each column.'
        '''
        data = self.data.get_all_data()
        std_dev = np.std(data, axis=0)
        result = data/std_dev
        self.data = self.update_data(result)

        return result 
    
    def filter(self, header, condition):
        ''' Filter a variable based on a condition

        Parameters:
        -----------
        header: str. Header of the variable to be filtered.
        condition: function. condition to apply to the variable. 

        Returns
        -------
        ndarray. shape=(M, N)
            Original data before `fitler` is applied.
        '''
        old_data = self.data.data 
        header_idx = self.data.get_header_indices([header])[0]

        new_data = old_data[condition(old_data[:, header_idx])]
        self.data = self.update_data(new_data)

        return old_data 

    def heatmap(self, headers=None, title=None, cmap="gray"):
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = data.Data(headers=self.data.get_headers(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap)

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls )
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax
