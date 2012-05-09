
class iter_product(object):
    def __init__(self,*args):
        """
        Create an iterator object that returns all combinations
        of values in arguments.  This is equivalent to the function
        itertools.product, available in python 2.6 and above.
        
        Example
        -------
        >>> for tup in iter_product(range(3), range(2)):
        >>>     print tup
        (0, 0)
        (0, 1)
        (1, 0)
        (1, 1)
        (2, 0)
        (2, 1)
        """
        # take care of generator objects which have no length attribute
        self.N = len(args)
        self.args = self.N * [None]
        for i,arg in enumerate(args):
            if not hasattr(arg, '__len__'):
                self.args[i] = [a for a in arg]
            else:
                self.args[i] = arg
        self.indices = self.N * [0]
        self.indices[-1] = -1
        self.max_indices = [len(self.args[i]) for i in xrange(self.N)]

    def next(self):
        for j in xrange(self.N-1,-1,-1):
            self.indices[j] += 1
            if self.indices[j] == self.max_indices[j]:
                if j == 0:
                    raise StopIteration
                else:
                    self.indices[j] = 0
            else:
                break
        return tuple(self.args[j][self.indices[j]] for j in xrange(self.N))

    def __iter__(self):
        return self


if __name__ == '__main__':
    print '(0, 1, 2) x (0, 1)'
    for t in iter_product((0,1,2), (0,1)):
        print ' ', t
