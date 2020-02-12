# Qingbo Liu
# Spring 2020 

# test file for data class

import data
import sys


# test program for the Data class
def main(argv):

    # test command line arguments
    if len(argv) < 2:
        print( 'Usage: python %s <csv filename>' % (argv[0]))
        exit(0)

    # create a data object, which reads in the data
    dobj = data.Data(argv[1])

    # print out information about the dat
    print('Number of rows:    ', dobj.get_num_points() )
    print('Number of columns: ', dobj.get_num_dimensions() )

    # print out the headers
    print("\nHeaders:")
    headers = dobj.get_headers()
    s = headers[0]
    for header in headers[1:]:
        s += ", " + header
    print( s )

    # print out the types
    print("\nTypes")
    types = dobj.get_types()
    s = types[0]
    for type in types[1:]:
        s += ", " + type
    print( s )

    # print out a single row
    print("\nPrinting row index 2")
    print( dobj.get_row( 2 ) )

    # print out a cols 
    print("\nPrinting col header headers")
    print( dobj.get_cols(['headers']) )

    # print out all of the data
    print("\nData")
    headers = dobj.get_headers()
    for i in range(dobj.get_num_points()):
        s = '' 
        for j, header in enumerate(headers):
            if types[j] == 'numeric':
                s += "%10.3f" % (dobj.get_value( header, i ))
            else:
                s += ' ' 
                s += str(dobj.get_value(header, i))
        print(s)

    print("\nEnum")
    print( dobj.get_enum_dict("test") )

    print("\nDate")
    print( dobj.get_cols(['date']) )

    # test add_column 
    print("\ntesting add_column")
    dobj.add_column('o', 'numeric', ['12', '-9999', '3'])
    print(dobj.get_cols(['o']))
    dobj.add_column('i', 'enum', ['hello', 'yes', 'hello'])
    print(dobj.get_cols(['i']))
    print(dobj.get_enum_dict('i'))

    print("\nNumeric Data")
    print( dobj.nmc_data )

    print("\nNon Numeric Data")
    print( dobj.nnmc_data )


if __name__ == "__main__":
    main(sys.argv)

