# Qingbo Liu
# Spring 2020 

from functools import reduce 

import numpy as np

# Takes in a list of column headers and the Data object and returns a list of
# 2-element lists with the minimum and maximum values for each column.
# Works only for numeric data. 
def data_range(headers, data):
    mins = np.amin(data, axis=0).tolist()[0]
    maxs = np.amax(data, axis=0).tolist()[0]
    
    return list(map(lambda x: [x[0], x[1]], zip(mins, maxs)))


# Takes in a list of column headers and the Data object and returns a list
# of the mean values for each column.
# Works only for numeric data. 
def mean(headers, data):
    return np.mean(data, axis=0)


# Takes in a list of column headers and the Data object and returns a list of
# the standard deviation for each specified column. 
# Works only for numeric data. 
def stdev(headers, data):
    return np.std(data, axis=0)

# Takes in a list of column headers and the Data object and returns a matrix 
# with each column normalized so its minimum value is mapped to zero and its 
# maximum value is mapped to 1.
# Works only for numeric data. 
def normalize_columns_separately(headers, data):
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    range = maxs - mins

    m = data - mins 
    return m / range 

# Takes in a list of column headers and the Data object and returns a matrix
# with each entry normalized so that the minimum value (of all the data in this
# set of columns) is mapped to zero and its maximum value is mapped to 1.
# Works only for numeric data. 
def normalize_columns_together(headers, data):
    range = data_range(headers, data) 
    (min_v, max_v) = reduce(lambda i, j: (min(i[0], j[0]), max(i[1], j[1])), range)
    range_v = max_v - min_v

    m = data - min_v
    return m / range_v
    
def main():
    import data 
    # read in data 
    data = data.Data("./data.csv")

    headers = ['\ufeffID', 'Age', 'Overall', 'Potential']
    cols = data.get_cols(headers)

    print("data range \n{}\n".format(data_range(headers, cols)))
    print("geometric mean \n{}\n".format(mean(headers, cols)))
    print("standard deviation \n{}\n".format(stdev(headers, cols)))
    print("column separate normalization \n{}\n".format(normalize_columns_separately(headers, cols)))
    print("column integral normalization \n{}\n".format(normalize_columns_together(headers, cols)))
    

if __name__ == "__main__":
    main()
    



    

