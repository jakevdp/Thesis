import numpy

def get_ids(filename):
    F = open(filename)

    #cycle through comments at top
    line = F.next()
    while line.startswith('\\'):
        line = F.next()

    #cycle through header data
    colnames = map(lambda x: x.strip(),
                   line.split('|')[1:-1])
    return colnames
    

def read_cosmos_table(filename,cols,N):
    """
    read N columns from filename
    """
    F = open(filename)

    #cycle through comments at top
    line = F.next()
    while line.startswith('\\'):
        line = F.next()

    #cycle through header data
    colnames = map(lambda x: x.strip(),
                   line.split('|')[1:-1])
    line = F.next()
    dtypes = map(lambda x: x.strip(),
                 line.split('|')[1:-1])
    while line.startswith('|'):
        line = F.next()

    indices = numpy.array([colnames.index(col) for col in cols])
    values = numpy.zeros( (N,len(indices)) )

    for i in range(N):
        line = numpy.fromstring(line.replace('null','NaN'),
                                float,sep=' ')
        values[i] = line[indices]

        try:
            line = F.next()
        except StopIteration:
            break

    return values[:i+1,:]

if __name__ == '__main__':
    from filenames import small_file, full_file
    print get_ids(small_file)
