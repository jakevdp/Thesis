from read_catalog import read_cosmos_table
import numpy
import pylab
from matplotlib import ticker

def plot_hist(filename,
              N=400000):
    values = read_cosmos_table(filename,
                               ['zp_best','type'],
                               N)
    #select only galaxies
    i = numpy.where(values[:,1]==0)
    values = values[i[0],:]

    #select only objects with valid photometric redshift
    i = numpy.where(~numpy.isnan(values[:,0]))
    values = values[i[0],:]

    print "selected %i sources" % values.shape[0]

    pylab.hist(values[:,0],bins=300,linewidth=0.1)
    pylab.xlabel('z_phot')
    pylab.ylabel('N(z)')
    pylab.text(0.95,0.95,"%i sources" % values.shape[0],
               transform = pylab.gca().transAxes,
               fontsize=14,
               va='top',ha='right')
    pylab.title('COSMOS photo-z distribution')

def plot_by_mag(filename,
                bandid = 'i_auto',
                N=400000,
                mag_bins=numpy.arange(16,26) ):
    values = read_cosmos_table(filename,
                               ['zp_best','type',bandid],
                               N)
    print " - %i total objects" % values.shape[0]
    
    #select only galaxies
    i = numpy.where(values[:,1]==0)
    values = values[i[0],:]
    print " - %i galaxies" % values.shape[0]

    #select only objects with valid photometric redshift
    i = numpy.where(~numpy.isnan(values[:,0]))
    values = values[i[0],:]
    print " - %i with redshift" % values.shape[0]

    #select only objects with valid <bandid> magnitude
    i = numpy.where(~numpy.isnan(values[:,2]))
    values = values[i[0],:]
    print " - %i with valid mag" % values.shape[0]

    """
    mags = values[:,2]
    mags = mags[numpy.where(mags!=-99)]
    mag_min = min(mags)
    mag_max = max(mags)

    mag_min = int(numpy.floor(mag_min))
    mag_max = int(numpy.ceil(mag_max))
    mag_bins = numpy.arange(mag_min,mag_max+1)
    """

    Nmag = len(mag_bins)-1

    pylab.figure(figsize=(10,10))
    
    for i in range(Nmag):
        ind = numpy.where( (values[:,2]>=mag_bins[i])\
                         &(values[:,2]<mag_bins[i+1]) )
        z = values[ind[0],0]
        pylab.subplot(331+i)
        
        print "%i sources with %.1f<%s<%.1f" % (len(z),
                                                mag_bins[i],
                                                bandid,
                                                mag_bins[i+1])

        #Nbins = len(z)/50
        #Nbins = max(Nbins,30)
        #Nbins = min(Nbins,300)
        bins = numpy.linspace(0,4,201)

        if bandid=='i_auto': bandid='I'
        
        pylab.hist(z,bins,linewidth=0)
        if i in (6,7,8):
            pylab.xlabel('z_phot')
        if i in (0,3,6):
            pylab.ylabel('N(z)')
        pylab.text(0.95,0.95,"%.1f<%s<%.1f\n%i sources" % (mag_bins[i],
                                                           bandid,
                                                           mag_bins[i+1],
                                                           len(z) ),
                   transform = pylab.gca().transAxes,
                   fontsize=12,
                   va='top',ha='right')
        pylab.xlim(0,4)
        pylab.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
                


if __name__ == '__main__':
    from filenames import small_file,full_file

    #plot_hist(full_file)
    #pylab.savefig('COSMOS_photoz_hist.pdf')

    plot_by_mag(full_file)
    pylab.savefig('COSMOS_photoz_hist_binned.pdf')
    pylab.show()

    
