"""
This contains the class that allows specification of parameters
for shear_KL
"""

import numpy
import os

class param_container(object):
    """
    param_container is a borg class.  Any instance points to the
    common dictionary __shared_state.  Any modification to a single
    param_container instance modifies the state of all 
    param_container instances.
    """

    #create shared parameter dictionary
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state

    def __getitem__(self,item):
        return getattr(self,item)

    def __setitem__(self,item,val):
        return setattr(self,item,val)

    def update_param_dict(self,D):
        self.__dict__.update(D)

    def clear_param_dict(self):
        keys = self.__dict__.keys()
        for k in keys: del self.__dict__[k]

    def get_keys(self):
        return self.__dict__.keys()

    keys = property(get_keys)

    def dump(self):
        """
        dump()
          print all parameters and types
        """
        print "parameters"

        lines = [ (p,str(v),type(v).__name__) \
                      for p,v in self.__dict__.iteritems() ]

        if len(lines)==0:
            print ' [no parameters]'

        else:
            L0 = max([len(l[0]) for l in lines])
            L1 = max([len(l[1]) for l in lines])

            for p,v,t in lines:
                print ' %s %s %s' % (p.ljust(L0),
                                     v.ljust(L1),
                                     t)
    #---

    def load(self,filename,
             clear_current=False):
        """
        load(filename,clear_current=False)

        Fill the package variable params with the parameters specified
        in the filename.
        
        if clear_current is True, then first empty current parameter settings
        
        filename has one parameter per line, space delimited:
        name val [type] #comment
        
        if type is not specified, string type is assumed
        
        For example:
          #start params.dat
          dir          /local/tmp              #no type code: default is string
          pi           3.14                    float
          num_entries  3                       i
          file         $(dir)/myfile.txt       #dollar sign indicates variable
                                               # substitution
          #end params.dat
        """
        D = {}
        linecount = 0
        for line in open(filename):
            linecount += 1
            line = ' '.join(line.split('#')[:1]).split()
            if len(line)==0: 
                continue
            if len(line)==1:
                raise ValueError, "Must specify value for parameter %s" \
                    % line[0]
            elif len(line)>3:
                raise ValueError, "cannot understand line %i of %s" \
                    % (linecount,filename)

            if (line[0][0]>='0' and line[0][0]<='9'):
                raise ValueError, "invalid variable name %s" % line[0]
            
            #check for variables in the value
            if '$' in line[1]:
                line[1] = replace_vars(line[1],D)

            if len(line)==2:
                D[line[0]] = line[1]
            else:
                try:
                    T = numpy.dtype(line[2])
                except:
                    raise ValueError, "%s: line %i: type '%s' not understood" \
                        % ( filename,linecount,line[2])
                D[line[0]] = T.type(line[1])

        if clear_current:
            self.clear_param_dict()

        self.update_param_dict(D)
        self.calc_extra_params()

    def calc_extra_params(self):
        """
        calculate and save some extra parameters
        """
        D = self.__dict__
        if 'field_width' in D and 'Npix' in D:
            self.update_param_dict({'dtheta': self.field_width/self.Npix})
            if 'ngal_arcmin' in D:
                self.update_param_dict({'ngal':
                                        self.ngal_arcmin * self.dtheta**2})

        if 'RAmin' in D and 'RAmax' in D:
            self.update_param_dict({'RAlim':(self.RAmin,self.RAmax)})
            NRA = int(numpy.round((self.RAmax-self.RAmin)*60./self.dtheta))
            self.update_param_dict({'NRA': NRA})
        if 'DECmin' in D and 'DECmax' in D:
            self.update_param_dict({'DEClim':(self.DECmin,self.DECmax)})
            NDEC = int(numpy.round((self.DECmax-self.DECmin)*60./self.dtheta))
            self.update_param_dict({'NDEC': NDEC})

        if 'z0' in D and 'nz_a' in D and 'nz_b' in D:
            self.n_z = lambda z,z0=self.z0: z**self.nz_a \
                       * numpy.exp(-(z/z0)**self.nz_b)

        if 'zmin' in D and 'zmax' in D:
            self.zlim = (self.zmin,self.zmax)
            
            
    #---
#---

def replace_vars(s,D):
    """
    given a string s and a dictionary of variables D, replace all variable
    names with the value from the dict D

    variable names in s are denoted by '$(' at the beginning and ')' at
    the end, or '$' at the beginning and a non-variable character at the
    end.  Variables must be valid python variable names, that is they
    consist of only alphanumeric characters (A-Z,a-z,0-9) and underscores,
    and cannot start with a number.
    
    example:
      >> D = {'my_var1' : 'abc',
              'my_var2' : '123' }
      >> s = "I know my $(my_var1)s and $my_var2's"
      >> print replace_vars(s,D)
      
      I know my abcs and 123's
    """
    s_in = str(s) 
    s_out = ''

    while True:
        i = s_in.find('$')
        if i==-1:
            s_out += s_in
            break
        
        s_out += s_in[:i]
        s_in = s_in[i+1:]
        
        if len(s_in)==0:
            raise ValueError, "trailing $"

        elif s_in[0] == '(':
            i = s_in.find(')')
            if i==-1:
                raise ValueError, "unmatched '('"
            var = s_in[1:i]

            s_in = s_in[i+1:]
            s_out += str(D[var])

        else:
            var = ''
            i = 0
            while True:
                if i>=len(s_in):
                    break
                s = s_in[i]
                if (s >= 'a' and s <= 'z') \
                        or (s >= 'A' and s <= 'Z') \
                        or (s >= '0' and s <= '9') \
                        or s=='_':
                    var += s
                    i += 1
                else:
                    break
            s_in = s_in[i:]
            s_out += str(D[var])
                        
            
    return s_out  
        
