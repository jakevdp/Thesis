from thin_lens import *
import numpy
import sys

"""
generate_3D_shear.py

usage: generate_3D_shear.py <filename> <options>
         options:
           -N <num_clusters>
              # number of clusters to use:
              #  default = 10
           -x <xmin> <xmax>
              # x-range of the clusters (arcmin):
              #  default = 0 10
           -y <xmin> <xmax>
              # y-range of the clusters (arcmin)
              #  default = 0 10
           -z <zmin> <zmax>
              # reshift range of the clusters (dimensionless)
              #  default = 0.1 0.3
           -sig <sig_min> <sig_max>
              # velocity dispersion range of the clusters (km/s)
              #  default = 200 500
              
           -xbins <x1> <x2> <x3> ...
              # x-positions of outputs (arcmin)
           -xrange <xmin> <xmax> <Nx> ...
              # alternative to xbins: specify a min, max, and number
              
           -ybins <y1> <y2> <y3> ...
              # y-positions of outputs (arcmin)
           -yrange <ymin> <ymax> <Ny> ...
              # alternative to ybins: specify a min, max, and number
              
           -zbins <z1> <z2> <z3> ...
              # z-positions of outputs (dimensionless)
           -zrange <zmin> <zmax> <Nz> ...
              # alternative to zbins: specify a min, max, and number

           -rseed <r>
              # random seed
"""

def create_random_profiles(N_clusters,
                           zmin,zmax,
                           sigmin,sigmax,
                           theta1min,theta1max,
                           theta2min,theta2max,
                           rseed = None):
    if rseed is not None:
        numpy.random.seed(rseed)
    
    z = numpy.random.random(N_clusters)*(zmax-zmin)+zmin
    theta1 = numpy.random.random(N_clusters)*(theta1max-theta1min)+theta1min
    theta2 = numpy.random.random(N_clusters)*(theta2max-theta2min)+theta2min
    theta = theta1 + 1j*theta2
    sig = numpy.random.random(N_clusters)*(sigmax-sigmin)+sigmin

    PS = ProfileSet()
    for i in range(N_clusters):
        PS.add( SIS(z[i],sig[i],theta[i]) )

    return PS

def generate_3D_shear( filename,
                       N_clusters,
                       zmin,zmax,
                       sigmin,sigmax,
                       theta1min,theta1max,
                       theta2min,theta2max,
                       z_out_range,
                       theta1_out_range,
                       theta2_out_range,
                       rseed = None):
    PS = create_random_profiles(N_clusters,
                                zmin,zmax,
                                sigmin,sigmax,
                                theta1min,theta1max,
                                theta2min,theta2max,
                                rseed)

    PS.write_to_file(filename, z_out_range,
                     theta1_out_range, theta2_out_range)

def read_shear_file(filename):
    L = []
    z = []
    theta1 = []
    for line in open(filename):
        line = line.strip()
        if line.startswith('#'):
            continue
        line = map(float,line.split())
        
        if z==[] or line[0] != z[-1]:
            L.append([])
            z.append(line[0])
            theta1 = []
            
        if theta1==[] or line[1] != theta1[-1]:
            L[-1].append([])
            theta1.append( line[1] )
            theta2 = []
        theta2.append(line[2])
        
        L[-1][-1].append(line[3:])

    return map(numpy.array,(z,theta1,theta2,L))
    
    

if __name__ == '__main__':
    N = 10
    xmin = 0
    xmax = 10
    ymin = 0
    ymax = 10
    zmin = 0.1
    zmax = 0.3
    sig_min = 200
    sig_max = 500

    xbins = None
    ybins = None
    zbins = None

    rseed = None
    
    if len(sys.argv)==1:
        print __doc__
    else:
        filename = sys.argv[1]
        i = 2
        while i<len(sys.argv):
            arg = sys.argv[i]
            if arg == '-N':
                N = int(sys.argv[i+1])
                i+=1
            elif arg=='-x':
                xmin = float(sys.argv[i+1])
                xmax = float(sys.argv[i+2])
                i+=2
            elif arg=='-y':
                ymin = float(sys.argv[i+1])
                ymax = float(sys.argv[i+2])
                i+=2
            elif arg=='-z':
                zmin = float(sys.argv[i+1])
                zmax = float(sys.argv[i+2])
                i+=2
            elif arg=='-sig':
                sig_min = float(sys.argv[i+1])
                sig_max = float(sys.argv[i+2])
                i+=2
            elif arg=='-xbins':
                if xbins != None:
                    raise ValueError, "xbins supplied twice"
                xbins = []
                while( i+1<len(sys.argv) ):
                    try:
                        xval = float(sys.argv[i+1])
                    except:
                        break
                    xbins.append(xval)
                    i+=1
            elif arg=='-ybins':
                if ybins != None:
                    raise ValueError, "ybins supplied twice"
                ybins = []
                while( i+1<len(sys.argv) ):
                    try:
                        yval = float(sys.argv[i+1])
                    except:
                        break
                    ybins.append(yval)
                    i+=1
            elif arg=='-zbins':
                if zbins != None:
                    raise ValueError, "zbins supplied twice"
                zbins = []
                while( i+1<len(sys.argv) ):
                    try:
                        zval = float(sys.argv[i+1])
                    except:
                        break
                    zbins.append(zval)
                    i+=1

            elif arg=='-xrange':
                if xbins != None:
                    raise ValueError, "xbins supplied twice"
                x0 = float(sys.argv[i+1])
                x1 = float(sys.argv[i+2])
                Nx   = float(sys.argv[i+3])
                xbins = numpy.linspace(x0,x1,Nx)
                i+=3

            elif arg=='-yrange':
                if ybins != None:
                    raise ValueError, "ybins supplied twice"
                y0 = float(sys.argv[i+1])
                y1 = float(sys.argv[i+2])
                Ny   = float(sys.argv[i+3])
                ybins = numpy.linspace(y0,y1,Ny)
                i+=3

            elif arg=='-zrange':
                if zbins != None:
                    raise ValueError, "zbins supplied twice"
                z0 = float(sys.argv[i+1])
                z1 = float(sys.argv[i+2])
                Nz   = float(sys.argv[i+3])
                zbins = numpy.linspace(z0,z1,Nz)
                i+=3

            elif arg=='-rseed':
                rseed = int(sys.argv[i+1])
                i+=1
                
            else:
                raise ValueError, "unrecognized argument: " + arg

            i+=1
    #
    
    if xbins is None:
        raise ValueError, "must supply xbins"
    if ybins is None:
        raise ValueError, "must supply ybins"
    if zbins is None:
        raise ValueError, "must supply zbins"


    print "saving shear data to %s" % filename
    generate_3D_shear( filename,
                       N,
                       zmin,zmax,
                       sig_min,sig_max,
                       xmin,xmax,
                       ymin,ymax,
                       zbins,xbins,ybins,
                       rseed)
