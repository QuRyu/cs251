'''data.py
Reads CSV files, stores data, access/filter data by variable name
Qingbo Liu
CS 251 Data Analysis and Visualization
Spring 2020
'''

import csv 
import numpy as np
from enum import Enum
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

# Enum to differentiate between numeric and non-numeric data types 
class DataType(Enum):
    Numeric = 1 
    NonNumeric = 2 

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
            # try: 
                self.filepath = filepath
                csv_reader = csv.reader(file)
                self._read_headers(next(csv_reader))
                self._read_types(next(csv_reader))
                self._read_content(csv_reader)
            # except: 
                # e = sys.exc_info()[1]
                # print(str(e))


    def __str__(self):
        '''toString method
        For now support only numeric data 


        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''

        str_list = []
        for header in self.headers:
            if self.header2col[header][0] == DataType.Numeric: 
                str_list.append(header)
                str_list.append(' ')
        str_list.append('\n')

        for type in self.types:
            if not type == "string":
                str_list.append(type)
                str_list.append(' ')
        str_list.append('\n')

        # str_list.append(str(self.data))
        str_list.append(str(self.nmc_data))

        
        return ''.join(str_list)

    def get_headers(self, numeric_only = True):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''
        if numeric_only:
            list = [] 
            for (header, type) in zip(self.headers, self.types):
                if type == 'numeric':
                    list.append(header)
            return list 
        else:
            return self.headers 


    def get_types(self, numeric_only = True):
        '''Get method for data types of variables

        Returns:
        -----------
        Python list of str.
        '''
        if numeric_only:
            return list(filter(lambda x: x == 'numeric', self.types))
        else:
            return self.types

    def get_mappings(self, numeric_only = True):
        '''Get method for mapping between variable name and column index
        For now returns only mapping for numeric data 

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        if numeric_only:
            numeric_headers = self.get_headers(True)
            return {k: v[1] for k, v in self.header2col.items() if k in numeric_headers}
        else:
            return {k: v[1] for k, v in self.header2col.items()}

        

    def get_num_dims(self, numeric_only = True):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        return len(self.get_headers(numeric_only))

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return self.nmc_data.shape[0]

    def get_sample(self, rowInd, numeric = True):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)
        For now return only numeric data 

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        if numeric:
            return self.nmc_data[rowInd] 
        else:
            return self.nnmc_data[rowInd]


    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.
        For now support only numeric data 

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
            indices.append(self.header2col[header][1])
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
        # check if type is allowed 
        for t in types:
          if t not in Types:
              raise ValueError("data type {} not supported: check row 2 of the data file\n".format(t))
        self.types = types 

        # check number of dimensions
        if len(types) != len(self.headers):
            raise ValueError("Number of types on line 2 not compatible with headers") 
        
        self.header2col = {} # dictionary mapping headers to corresponding cols in data 
        nmc_idx = 0  # numeric data index  
        nnmc_idx = 0 # non-numeric data index 

        self.enum_dict = {} # dictionary mapping enum cols to enum dict 
                            # which maps enum strings to numeric data 
        for i, header in enumerate(self.headers):
            if not types[i] == 'string':
                self.header2col[header] = (DataType.Numeric, nmc_idx)
                nmc_idx += 1

                if types[i] == 'enum':
                    self.enum_dict[i] = {} 
            else:
                self.header2col[header] = (DataType.NonNumeric, nnmc_idx)
                nnmc_idx += 1 

    def _read_content(self, csv_reader):
        nmc_data = []  # numeric data 
        nnmc_data = [] # non-numeric data 
        N = self.get_num_dims(False)

        enum_count = {}
        # count #enums 
        for i, type in enumerate(self.types):
            if type == 'enum':
                enum_count[i] = 0

        for line, values in enumerate(csv_reader):
            # check if there are enough values 
            if len(values) != N:
                raise DimensionError(line, values,
                        'Data on line {} does not have the right dimension'.format(line))

            nmc_d = [] 
            nnmc_d = [] 
            for i, item in enumerate(values):
                if self.types[i] == 'numeric':
                    nmc_d.append(self._parse_numeric(item, line, i))
                elif self.types[i] == 'date':
                    nmc_d.append(self._parse_date(item, line, i))
                elif self.types[i] == 'enum':
                    if item not in self.enum_dict[i]:
                        self.enum_dict[i][item] = enum_count[i]
                        enum_count[i] += 1 
                    nmc_d.append(self.enum_dict[i][item])
                else:
                    nnmc_d.append(item)
                    
                    
            nmc_data.append(nmc_d) 
            nnmc_data.append(nnmc_d)

        self.nmc_data = np.matrix(nmc_data)
        self.data = self.nmc_data
        self.nnmc_data = np.matrix(nnmc_data)

    # parse date with several formats and returns the epoch time 
    def _parse_date(self, date, line, col):
        values = date.strip().split('/')
        try:
            if len(values) == 3:
                day = int(values[0])
                month = int(values[1]) 
                year = values[2]

                day = '0' + str(day) if day < 10 else values[0] 
                month = '0' + str(month) if month < 10 else values[1] 

                date = '{} {} {}'.format(day, month, year)
                t = time.mktime(time.strptime(date, "%d %m %y"))
                return t
            else:
                raise ValueError
        except:
            e = sys.exc_info()[1] 
            print(e)
            raise ValueError("ill-formatted date at line {}, col {}".format(line+1, col+1))
                    

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

