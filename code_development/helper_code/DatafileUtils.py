#######
# The purpose of this module is to provide utilities that make
# manipulating datafiles easier.

def describeDatafile(inFilename, fileType = 'csv'):
    '''
    This function opens a datafile with the name given in "inFilename",
    reads its data into  a two column pandas dataframe in which the first
    column is the sequence and the second column is the genome type lable
    (and integer). It then computes and returns a dictionary with statistics
    that describe the data in the data file.

    Input:
        - the name of the file to open
        - the type of file to handle (default is "csv"). Currently can only
            handle csv files. [Genome file format should be added]

    Output:
        a dictionary has the following key names (which provide the information
        described below):
            - validFlag: A boolean indicating whether the datafile was
                successfully read; Reasons for being false are: the data
                file have the wrong number of columns, the file not being
                found in the location indicated.
                [Add more as code to handle other reasons is written]
            - numRows: The number of rows in the data file (int)
            - allLetters: A list of each unique letter found in the file
                sequences
            - totalNumLetters: The total number of letters in the file (int)
            - avgLettsPerSeq: The average number of letters per sequence (float)
            - medLettsPerSeq: The median number of letters per sequnce (float)
            - maxLettsPerSeq: The maximum number of letters in a sequence (int)
            - minLettsPerSeq: The minimum number of letters in a sequence (int)
            - lettsInLine: A numpy array, the nth entry of which contains the
                number of letters that are in the nth row of the file (int)

    '''

    raise ReferenceError("This function is not yet implemented")
    
