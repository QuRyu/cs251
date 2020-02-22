'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import analysis
import data
import palettable 


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
        super(Transformation, self).__init__(data)

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
        if len(headers) < 2 or len(headers) > 3:
            raise ValueError(f"headers {headers} should be only of length 2 or 3")
        
        self.ndim = len(headers)
        project_data = [] 
        for h in headers:
            project_data.append(self.data_orig.select_data(h).copy())
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
        elif len(headers) < 2 or len(headers) > 3:
            raise ValueError(f'headers should be of length 2 or 3')

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
        elif len(headers) < 2 or len(headers) > 3:
            raise ValueError(f'headers should be of length 2 or 3')

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
        import math 

        M = np.eye(4)
        header2col = self.data.get_mappings()
        
        angel = math.radians(degrees)

        if header not in header2col:
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
        return self.drop_homogeneous_coord(self.transform(M))

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
        return self.drop_homogeneous_coord(self.transform(M))

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
        return self.drop_homogeneous_coord(self.transform(M))

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        headers = self.data.get_headers()
        mins = super().min(headers)
        maxs = super().max(headers)

        global_min = np.min(mins)
        global_max = np.max(maxs)
        range = global_max - global_min 

        translation_M = self.translation_matrix(headers, [-global_min for _ in range(len(headers))])
        scale_M = self.scale_matrix(headers, [1/range for _ in range(len(headers))])

        return self.drop_homogeneous_coord(self.transform(scale_M @ translation_M))

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        mins, maxs = super().range(self.data.get_headers())
        ranges = maxs - mins 

        translation_M = self.translation_matrix(headers, -mins)
        scale_M = self.scale_matrix(headers, 1/ranges)

        return self.drop_homogeneous_coord(self.transform(scale_M @ translation_M))

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
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
        headers = seld.data.get_headers()
        for h in [ind_var, dep_var, c_var]:
            if h not in headers:
                raise ValueError(f'argument header {h} not in headers {headers}')

        fig, ax = plt.subplots()

        if title is not None: 
            ax.set_title(title)
        
        ind_data = self.data.select_data(ind_var)
        dep_data = self.data.select_data(dep_var)
        c_data = self.data.select_data(c_var)
        
        pcm = ax.pcolormesh(ind_data, dep_data, c_data, cmap=palettable.Colorbrewer.PuOr_10)
        fig.colorbar(pcm, ax=ax, extend='both')

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
