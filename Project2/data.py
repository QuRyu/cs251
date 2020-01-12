# Qingbo Liu 
# Spring 2020 

import csv  
import numpy 
import string 

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
        return self.data.shape[0]

    # returns the specified row as a NumPy matrix
    def get_row(self, rowIndex):
        return self.data[rowIndex]

    # returns the specified value in the give column
    def get_value(self, header, rowIndex):
        col = self.header2col[header]
        return self.data[rowIndex, col]

    def _read_headers(self, headers):
        headers = list(map(lambda s: s.strip(), headers))
        self.headers = headers

        self.header2col = {}
        for i, header in enumerate(headers):
            self.header2col[header] = i 


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

    def _read_content(self, csv_reader):
        data = []
        N = self.get_num_dimensions()
        for line, values in enumerate(csv_reader):
            # check if there are enough values 
            if len(values) != N:
                raise DimensionError(line, values,
                        'Data on line {} does not have the right dimension'.format(line))

            d = [] 
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
                        d.append(float(item))
                    except (ValueError, OverflowError) as e: 
                        raise ConversionError(line, i+1, item, e)
                else:
                    # TODO: for now, ignore error checking for string, enum and date 
                    d.append(item)
            data.append(d) 

        self.data = numpy.matrix(data)

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
        
        
