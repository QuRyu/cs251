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

class Data:
    def __init__(self, filepath=None, headers=None, types=None, data=None, header2col=None):
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
        elif headers and types and data and header2col:
            self.headers = self.headers_all = headers
            self.types = self.types_all = types
            self.data = data
            self.header2col = header2col

            self.all_data = {'numeric': self.data, 'enum': [], 
                    'date': [], 'string': []}
            self.all_header2col = {'numeric': self.header2col, 'enum': {}, 
                    'date': {}, 'string': {}}

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

                self.headers_all = headers # both numeric and non-numeric headers 
                self.types_all = types # both numeric and non-numeric types
                self.headers = []    # numeric headers 
                self.types = []      # numeric types 

                self.header2col = {} # mapping numeric headers to cols in (numeric) data
                self.header2col_enum = {} # mapping enum headers to cols in enum_data
                self.header2col_date = {} # mapping date headers to cols in date_data
                self.header2col_str  = {} # mapping str headers to cols in str_data

                idx = 0  # index for numeric data 
                idx_date = 0 
                idx_enum = 0 
                idx_str  = 0

                for (header, type) in zip(headers, types):
                    if type == 'numeric':
                        self.header2col[header] = idx
                        self.headers.append(header)
                        self.types.append(type)
                        idx += 1
                    elif type == 'enum':
                        self.header2col_enum[header] = idx_enum
                        idx_enum += 1 
                    elif type == 'date':
                        self.header2col_date[header] = idx_date 
                        idx_date += 1 
                    elif type == 'string':
                        self.header2col_str[header] = idx_str
                        idx_str += 1 

                # invert the enum_mapping to type Dict<col, Dict<int, str>>
                invert = lambda x: {v: k for k, v in x.items()}
                self.enum_mapping = {i: invert(v) for i, (_, v) in enumerate(self.enum_mapping.items())}

                self.all_data = {'numeric': self.data, 'enum': self.enum_data, 
                        'date': self.date_data, 'string': self.str_data}
                self.all_header2col = {'numeric': self.header2col, 'enum': self.header2col_enum, 
                        'date': self.header2col_date, 'string': self.header2col_str}
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

    def get_headers(self, return_all=False):
        '''Get method for headers

        Parameters:
        -----------
        return_all: if return all info; by default only return numeric

        Returns:
        -----------
        Python list of str.
        '''
        if return_all:
            return self.headers_all
        else:
            return self.headers 

    def get_types(self, return_all=False):
        '''Get method for data types of variables

        Parameters:
        -----------
        return_all: if return all info; by default only return numeric

        Returns:
        -----------
        Python list of str.
        '''
        if return_all:
            return self.types_all
        else:
            return self.types

    def get_mappings(self, data_type='numeric'):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        if data_type not in Types:
            raise ValueError(f"Data type {data_type} not supported: pass only types {Types}")
        else:
            return self.all_header2col[data_type]

    def get_enum_mappings(self, header):
        '''Get method for mapping between value and enum name 

        Returns:
        -----------
        Python dictionary. int -> str
        '''

        col = self.header2col_enum[header] 
        return self.enum_mapping[col]

    def get_num_dims(self, return_all=False):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        if return_all:
            return len(self.headers_all)
        else:
            return len(self.headers)

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return self.data.shape[0]

    def get_sample(self, rowInd, data_type='numeric'):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        if data_type not in Types:
            raise ValueError(f"Data type {data_type} not supported: pass only types {Types}")
        else:
            return self.all_data[data_type][rowInd]


    def get_header_indices(self, headers, data_type='numeric'):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''
        if data_type not in Types:
            raise ValueError(f"Data type {data_type} not supported: pass only types {Types}")

        if isinstance(headers, str):
            headers = [headers]
        if bool(headers) and isinstance(headers, list) and all(isinstance(elem, str) for elem in headers):
            indices = [] 
            for header in headers:
                indices.append(self.all_header2col[data_type][header])
            return indices
        else:
            raise ValueError(f'headers {headers} not supported')

    def get_all_data(self, data_type='numeric'):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''
        if data_type not in Types:
            raise ValueError(f"Data type {data_type} not supported: pass only types {Types}")
        else:
            return self.all_data[data_type].copy()

    def head(self, data_type='numeric'):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        if data_type not in Types:
            raise ValueError(f"Data type {data_type} not supported: pass only types {Types}")
        else:
            return self.all_data[data_type][:5, :]

    def tail(self, data_type='numeric'):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        if data_type not in Types:
            raise ValueError(f"Data type {data_type} not supported: pass only types {Types}")
        else:
            return self.all_data[data_type][-5:, :]

    def select_data(self, headers, rows=[], data_type='numeric'):
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
        if data_type not in Types:
            raise ValueError(f"Data type {data_type} not supported: pass only types {Types}")

        indices = self.get_header_indices(headers, data_type)
        if isinstance(rows, list) and not rows:
            rows = range(self.get_num_samples())
        return self.all_data[data_type][np.ix_(rows, indices)]
            
    def _read_headers(self, headers):
        headers = list(map(lambda s: s.strip(), headers))
        self.headers = headers

    def _read_types(self, types):
        types = list(map(lambda s: s.strip(), types))

        # check number of dimensions
        if len(types) != len(self.headers):
            raise ValueError(f"Number of types on line 2 not compatible with headers: should have {len(self.headers)} types, but has {len(types)}") 

        # check if type is allowed 
        for t in types:
          if t not in Types:
              raise ValueError(f"data type {t} not supported: check row 2 of the data file")

        self.types = types 
        

    def _read_content(self, csv_reader):
        data = [] # numeric data 
        enum_data = [] 
        str_data = [] 
        date_data = [] 

        enum_mapping = {} # of type Dict<col, Dict<str, int>>
        enum_idx = {} # of type Dict<col, int>
        for i, t in enumerate(self.types):
            if t == 'enum':
                enum_idx[i] = 0 
                enum_mapping[i] = {}

        N = len(self.headers)

        for line, values in enumerate(csv_reader):
            # check if there are enough values 
            if len(values) != N:
                raise ValueError(f'Record on line {line} does not have the right dimension: should have {N} values, but has {len(values)}.')

            temp = [] # numeric data temp
            enum_temp = [] 
            str_temp = []
            date_temp = [] 

            for i, item in enumerate(values):
                if self.types[i] == 'numeric':
                    temp.append(self._parse_numeric(item, line, i))
                elif self.types[i] == 'string':
                    str_temp.append(item)
                elif self.types[i] == 'enum':
                    if item not in enum_mapping[i]: 
                        idx = enum_idx[i] 
                        enum_mapping[i][item] = idx 
                        enum_idx[i] = idx+1
                    enum_temp.append(enum_mapping[i][item])
                elif self.types[i] == 'date':
                    date_temp.append(self._parse_date(item, line, i))

            data.append(temp)
            enum_data.append(enum_temp)
            date_data.append(date_temp)
            str_data.append(str_temp)
                    
        self.data = np.array(data)
        self.enum_data = np.array(enum_data)
        self.str_data = np.array(str_data)
        self.date_data = np.array(date_data)

        self.enum_mapping = enum_mapping

    def _parse_numeric(self, num, line, col):
        # handle data omission 
        if num.strip() == '':
            return 0 
        try:
            if int(num) == -9999:
                return 0
        except:
            pass 

        try: 
            return float(num)
        except (ValueError, OverflowError) as e: 
            raise ValueError(f"Fail to convert {num} to numeric values on line {line+3}, col {col+1}")

    # parse date of the format 01/01/20 (dd/mm/yy)
    def _parse_date(self, date, line, col):
        try: 
            dates = date.split('/')
            day = int(dates[0])
            month = int(dates[1])
            year = dates[2]

            if day > 0 and day <= 31 and month > 0 and month <= 12:
                day = '0'+str(day) if day < 10 else str(day)
                month = '0'+str(month) if month < 10 else str(month)

                date = f"{day} {month} {year}"
                return time.mktime(time.strptime(date, "%d %m %y"))
            else:
                raise ValueError()
        except: 
            raise ValueError(f"fail to parse date on line {line+3}, col {col+1}")



        

