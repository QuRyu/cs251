# Qingbo Liu 
# Spring 2020 

import csv  
import numpy as np
import string 
from enum import Enum 
import time 
import os 
import xml.etree.ElementTree as ET
import pands as pd 

Types = ['numeric', 'string', 'enum', 'date']

# Unsupported types other than numeric, string, enum and date 
class TypeError(Exception):
    def __init__(self, type):
        self.type = type 

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

# Parses csv files according to following rules: 
# | The data should be in CSV format with commas separating different entries.
# | The first row of the CSV data file should be the variable names. There must be a non-empty name for each column.
# | The second row of the data should be the variable types: numeric, string, enum, and date. Numeric types can be either integers or floating point values; strings are arbitrary strings; enum implies there are a finite number of values but they can be strings or numbers;a date should be interpreted as a calendar date.
# | Missing numeric data should be specified by the number -9999 in integer format. A decimal would imply an actual value.
# | Any line that begins with a hash symbol should be ignored by the reader.
# 
# May throw `TypeError`, `DimensionError`, or `ConversionError` when reading data files during initialization stage. 
class Data:

    # reads in a file 
    def __init__(self, filename = None, headers = [], types = []): 
        # declare and initialize fields 

        # if filename is not None 
        if filename:
            name, ext = os.path.splitext(filename)
            # if ext == '.xml':
                # if not headers or not types:
                    # raise ValueError("for xml file, supply headers and types")
                # self.read_xml(filename, headers, types)
            if ext == '.csv':
                self.read(filename)

    # def read_xml(self, filename, headers, types):
        # # TODO: test this 
        # tree = ET.parse(filename)
        # root = tree.getroot()
        # get_range = lambda col: range(len(col))
	# l = [{r[i].tag:r[i].text for i in get_range(r)} for r in root]

	# df = pd.DataFrame.from_dict(l)
        # csv_file = '{}.csv'.format(name)
	# df.to_csv(csv_file) 

        # with open(csv_file, mode='r+') as f:
            # f.seek(0, 0)
            # for i in range(len(headers)-1):
                # f.write(headers[i] + ', ')
            # f.write(headers[-1])
            # f.write('\n')

            # for i in range(len(types)-1): 
                # f.write(types[i] + ', ')
            # f.write(types[-1])
            # f.write('\n')

        # self.read_csv(csv_file)

        # os.remove(csv_file)

    def read_csv(self, filename):
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

    # returns a list of all of the dictionary mappings between variable name and column index
    def get_mappings(self):
        mapping = {} 
        for i, header in enumerate(self.headers):
            mapping[header] = i

        return mapping  

    # returns the number of columns
    def get_num_dims(self):
        return len(self.headers)

    # returns the number of samples/rows in the data set
    def get_num_samples(self):
        return self.nmc_data.shape[0]

    # returns the rowInd-th data sample as a Numpy matrix 
    def get_sample(self, rowInd):
        # TODO: should this preserve the same dimension as the origin csv file? 
        return np.concatenate((self.nmc_data[rowInd], self.nnmc_data[rowInd]), axis=1)

    # returns cols specified by headers as a Numpy matrix 
    # optionally the user can specify range of rows to return 
    # def get_cols(self, headers, rows = []): 
        # # check headers are defined 
        # for header in headers:
            # if header not in self.header2col:
                # raise ValueError("header {} is not defined".format(header)) 
        
        # cols = [] 
        # for header in headers: 
            # (datatype, idx) = self.header2col[header]
            # if datatype is DataType.Numeric:
                # if not rows: 
                    # col = self.nmc_data[:, idx]
                # else:
                    # col = self.nmc_data[rows, idx]
            # else:
                # if not rows: 
                    # col = self.nnmc_data[:, idx]
                # else:
                    # col = self.nnmc_data[rows, idx]
            # cols.append(col)
        
        # return np.hstack(cols)


    # returns the specified value in the give column
    # def get_value(self, header, rowIndex):
        # (datatype, col) = self.header2col[header]
        # if datatype is DataType.Numeric:
            # return self.nmc_data[rowIndex, col]
        # else:
            # return self.nnmc_data[rowIndex, col]
    
    # returns the dictionary mapping enum to numeric values.
    # 
    # Errors:
    #    ValueError if the header is not defined or does not have type `date`
    # def get_enum_dict(self, header):
        # # check header is present and the type is enum 
        # for i, item in enumerate(self.headers):
            # if item == header and self.types[i] == 'enum':
                # return self.enum_dict[i]
                
        # raise ValueError("header {} not present or has wrong type".format(header))
                
    # add a new column to existing data 
    # 
    # Errors:
    #    ValueError: header is already named, type is not defined,
    #               or data does not have correct length 
    #    ConversionError: data is numeric and conversion fails 
    # def add_column(self, header, type, data):
        # if header in self.headers:
            # raise ValueError("header {} is already named".format(header))
        # elif type not in Types:
            # raise ValueError("type {} is not defined".format(type))
        # elif not len(data) == self.get_num_points():
            # raise ValueError("data {} does not have right length".format(data))

        # self.headers.append(header)
        # self.types.append(type)
        # if not type == 'enum': 
            # items = [] 
            # for line, d in enumerate(data): 
                # if type == 'numeric':
                    # items.append(self._parse_numeric(d, line, -1))
                # elif type == 'date':
                    # items.append(self._parse_date(d, line, -1))
                # else:
                    # items.append(d)

            # if type == 'numeric' or type == 'date':
                # self.nmc_data = np.hstack((self.nmc_data, (np.matrix(items)).T))
                # idx = self.nmc_data.shape[1]
                # self.header2col[header] = (DataType.Numeric, idx-1)
            # else:
                # self.nnmc_data = np.hstack((self.nnmc_data, (np.matrix(items)).T))
                # idx = self.nnmc_data.shape[1]
                # self.header2col[header] = (DataType.NonNumeric, idx-1)
        # else:
            # dic = {}
            # idx = 0 
            # items = [] 
            # for d in data:
                # if d not in dic:
                    # dic[d] = idx 
                    # idx += 1 
                # items.append(dic[d])

            # self.nmc_data = np.hstack((self.nmc_data, np.matrix(items).T))
            # idx = self.nnmc_data.shape[1]
            # self.header2col[header] = (DataType.Numeric, idx-1)
            # self.enum_dict[len(self.headers)-1] = dic 


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
        N = self.get_num_dimensions()

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
        self.nnmc_data = np.matrix(nnmc_data)

    # parse date with several formats and returns the epoch time 
    def _parse_date(self, date, line, col):
        date = date.strip()
        try:
            return time.mktime(time.strptime(date, "%b %d %Y")) # try format "Jan 1 2020"
        except ValueError:
            try:
                return time.mktime(time.strptime(date, "%d/%m/%Y")) # try format "1/1/2020"
            except ValueError:
                try:
                    return time.mktime(time.strptime(date, "%d-%b-%Y")) # try format "1-Jan-2020"
                except ValueError as e:
                    raise ConversionError(line, col+1, date, e)
                    

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
        
        
