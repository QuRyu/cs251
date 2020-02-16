'''test05_read_non_numeric_data.py
Test handling of non_numeric data
CS 251 Data Analysis and Visualization
Spring 2020
Qingbo Liu
''' 

import numpy as np

from data import Data

def read_data_constructor(fp):
    data = Data(fp)

    assert data.get_headers(True) == ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']

    day_col = data.get_header_indices('day', 'enum')
    day_mapping = data.get_enum_mappings('day')
    day_data = data.head('enum')[:, day_col]
    assert ['Sun' for _ in range(5)] == [day_mapping[i] for i in np.squeeze(day_data)]



if __name__ == '__main__':
    print('---------------------------------------------------------------------------------------')
    print('Begining test 1 (Read data in constructor)...')
    print('---------------------------------------------')
    data_file = 'data/tips.csv'
    read_data_constructor(data_file)
    print('---------------------------------------------')
    print('Finished test 1!')
    print('---------------------------------------------------------------------------------------')
