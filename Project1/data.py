'''data.py
Reads CSV files, stores data, access/filter data by variable name
Qingbo Liu
CS 251 Data Analysis and Visualization
Spring 2020
'''

import csv 
import numpy as np
import os 
import sys
import time 

Types = ['numeric', 'string', 'enum', 'date']

# A given line of data does not have the right dimension specified by headers
class DimensionError(Exception):
    def __init(self, line, values, message = None):
        self.line = line
        self.values = self.values
        self.message = message

# Error of converting string to numeric values 
class ConversionError(Exception):
    def __init(self, row, col, value, error):
        self.line = row
        self.col = col 
        self.value = value
        self.error = error 

class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        '''Data object constructor
        TODO:
        - Declare/initialize the following instance variables:
            - filepath
            - headers
            - types: Python list of strings (initialized to None).
                Possible values: 'numeric', 'string', 'enum', 'date'
            - data
            - header2col
            - Any others you find helpful in your implementation
        - If `filepath` isn't None, call the `read` method.
        '''
        if filepath: 
            name, ext = os.path.splitext(filepath)
            if ext == '.csv':
                self.read(filepath)
        elif headers and data and header2col:
            self.headers = headers
            self.data = data
            self.header2col = header2col

    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data` at the end (think of this as 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if
            there should be nothing returned
        '''

        with open(filepath, mode='rU') as file:
            try: 
                self.filepath = filepath
                csv_reader = csv.reader(file)
                self._read_headers(next(csv_reader))
                self._read_types(next(csv_reader))
                self._read_content(csv_reader)

                # build header2col and remove non-numeric headers 

                headers = self.headers 
                types = self.types 
                self.headers = [] 
                self.types = []
                self.header2col = {} # dictionary mapping headers to corresponding cols in data 
                idx = 0  

                for (header, type) in zip(headers, types):
                    if type == 'numeric':
                        self.header2col[header] = idx
                        self.headers.append(header)
                        self.types.append(type)
                        idx += 1
            except: 
                e = sys.exc_info()[1]
                print(str(e))


    def __str__(self):
        '''toString method

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''

        str_list = []
        str_list.append('Headers\n')
        for header in self.headers:
            str_list.append(header)
            str_list.append(' ')
        str_list.append('\n')

        str_list.append('Types:\n')
        for type in self.types:
            str_list.append(type)
            str_list.append(' ')
        str_list.append('\n')

        str_list.append(str(self.data))

        return ''.join(str_list)

    def get_headers(self):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''
        return self.headers 


    def get_types(self):
        '''Get method for data types of variables

        Returns:
        -----------
        Python list of str.
        '''
        return self.types

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        return self.header2col

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        return len(self.headers)

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return self.data.shape[0]

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        return self.data[rowInd]


    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''
        indices = [] 
        for header in headers:
            indices.append(self.header2col[header])
        return indices

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''
        pass

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        pass

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        pass

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''
        pass

    def _read_headers(self, headers):
        headers = list(map(lambda s: s.strip(), headers))
        self.headers = headers

    def _read_types(self, types):
        types = list(map(lambda s: s.strip(), types))

        # check number of dimensions
        if len(types) != len(self.headers):
            raise ValueError("Number of types on line 2 not compatible with headers") 

        # check if type is allowed 
        for t in types:
          if t not in Types:
              raise ValueError("data type {} not supported: check row 2 of the data file\n".format(t))

        self.types = types 
        

    def _read_content(self, csv_reader):
        data = [] 
        N = self.get_num_dims()

        for line, values in enumerate(csv_reader):
            # check if there are enough values 
            if len(values) != N:
                raise DimensionError(line, values,
                        'Data on line {} does not have the right dimension'.format(line))

            temp = [] 
            for i, item in enumerate(values):
                if self.types[i] == 'numeric':
                    temp.append(self._parse_numeric(item, line, i))

            data.append(temp)
                    
        self.data = np.matrix(data)

    def _parse_numeric(self, num, line, col):
        # handle data omission 
        try:
            if int(num) == -9999:
                return 0
        except:
            pass 

        try: 
            return float(num)
        except (ValueError, OverflowError) as e: 
            raise ConversionError(line, col+1, num, e)

