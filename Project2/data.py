# Qingbo Liu 
# Spring 2020 

import csv  
import numpy as np
import string 
from enum import Enum 

Types = ['numeric', 'string', 'enum', 'date']

class TypeError(Exception):
    def __init__(self, type):
        self.type = type 

class DimensionError(Exception):
    def __init(self, line, values, message = None):
        self.line = line
        self.values = self.values
        self.message = message

class ConversionError(Exception):
    def __init(self, row, col, value, error):
        self.line = row
        self.col = col 
        self.value = value
        self.error = error 

class DataType(Enum):
    Numeric = 1 
    NonNumeric = 2 

# Parses csv files following rules specified at 
# http://cs.colby.edu/courses/S19/cs251-labs/labs/lab02/
class Data:

    # reads in a file 
    def __init__(self, filename = None):
        # declare and initialize fields 

        # if filename is not None 
        if filename:
            self.read(filename)

    def read(self, filename):
        with open(filename, mode='rU') as file:
            csv_reader = csv.reader(file)
            self._read_headers(next(csv_reader))
            self._read_types(next(csv_reader))
            self._read_content(csv_reader)

    # returns a list of all of the headers
    def get_headers(self):
        return self.headers

    # returns a list of all of the types
    def get_types(self):
        return self.types 

    # returns the number of columns
    def get_num_dimensions(self):
        return len(self.headers)

    # returns the number of points/rows in the data set
    def get_num_points(self):
        return self.nmc_data.shape[0]

    # returns the specified row as a NumPy matrix
    def get_row(self, rowIndex):
        return np.concatenate((self.nmc_data[rowIndex], self.nnmc_data[rowIndex]), axis=1)

    # returns cols specified by headers as a Numpy matrix 
    # optionally the user can specify range of rows to return 
    def get_cols(self, headers, rows = []): 
        # check headers are defined 
        for header in headers:
            if header not in self.header2col:
                raise ValueError("header {} is not defined".format(header)) 
        
        cols = [] 
        for header in headers: 
            (datatype, idx) = self.header2col[header]
            if datatype is DataType.Numeric:
                if not rows: 
                    col = self.nmc_data[:, idx]
                else:
                    col = self.nmc_data[rows, idx]
            else:
                if not rows: 
                    col = self.nnmc_data[:, idx]
                else:
                    col = self.nnmc_data[rows, idx]
            cols.append(col)
        
        return np.hstack(cols)


    # returns the specified value in the give column
    def get_value(self, header, rowIndex):
        (datatype, col) = self.header2col[header]
        if datatype is DataType.Numeric:
            return self.nmc_data[rowIndex, col]
        else:
            return self.nnmc_data[rowIndex, col]

    def _read_headers(self, headers):
        headers = list(map(lambda s: s.strip(), headers))
        self.headers = headers

    def _read_types(self, types):
        # check number of dimensions
        if len(types) != self.get_num_dimensions():
            raise DimensionError(2, types, message = 'Wrong number of dimensions')

        types = list(map(lambda s: s.strip(), types))
        # check if type is allowed 
        for t in types:
          if t not in Types:
              raise TypeError(t)
        self.types = types 

        
        self.header2col = {}
        nmc_idx = 0  # numeric data index  
        nnmc_idx = 0 # non-numeric data index 
        for i, header in enumerate(self.headers):
            if types[i] == 'numeric':
                self.header2col[header] = (DataType.Numeric, nmc_idx)
                nmc_idx += 1
            else:
                self.header2col[header] = (DataType.NonNumeric, nnmc_idx)
                nnmc_idx += 1 

    def _read_content(self, csv_reader):
        nmc_data = []  # numeric data 
        nnmc_data = [] # non-numeric data 
        N = self.get_num_dimensions()

        for line, values in enumerate(csv_reader):
            # check if there are enough values 
            if len(values) != N:
                raise DimensionError(line, values,
                        'Data on line {} does not have the right dimension'.format(line))

            nmc_d = [] 
            nnmc_d = [] 
            for i, item in enumerate(values):
                if self.types[i] == 'numeric':
                    # handle data omission 
                    try:
                        if int(item) == -9999:
                            d.append(0)
                            continue 
                    except:
                        pass 

                    try: 
                        nmc_d.append(float(item))
                    except (ValueError, OverflowError) as e: 
                        raise ConversionError(line, i+1, item, e)
                else:
                    # TODO: for now, ignore error checking for string, enum and date 
                    nnmc_d.append(item)
            nmc_data.append(nmc_d) 
            nnmc_data.append(nnmc_d)

        self.nmc_data = np.matrix(nmc_data)
        self.nnmc_data = np.matrix(nnmc_data)

    def __str__(self):
        str_list = []
        for header in self.headers:
            str_list.append(header)
            str_list.append(' ')
        str_list.append('\n')

        for type in self.types:
            str_list.append(type)
            str_list.append(' ')
        str_list.append('\n')

        str_list.append(str(self.data))
        
        return ''.join(str_list)
        
        
