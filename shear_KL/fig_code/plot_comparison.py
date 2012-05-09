import os
import sys
import pylab
from matplotlib.ticker import MultipleLocator, NullFormatter

sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('../run/base_params.dat')

from plot_Map_by_name import plot_peak_func

def get_ax_bbox(ax):
    r,w = ax.get_window_extent()._bbox.get_points()
    print r
    print w
    return [r[0],r[1],w[0]-r[0],w[1]-r[1]]

filename = lambda s: os.path.join(params.shear_recons_dir,'Map_'+s+'.npz')

rmin = 3.25
rmax = 6.5
Mmin = 0.01
Mmax = 0.04
bins = 40

colors = 'krg'

label_axes = lambda L,ax: ax.text(0.01,0.98,L,
                                 ha = 'left', va='top',
                                 transform = ax.transAxes,
                                 fontsize=20)

pylab.figure( figsize=(11,8) )


noiseonly = (len(sys.argv)>1 and sys.argv[1] == '--noiseonly')

if noiseonly:
    print "------------------------------------------------------------"
    print "plotting for noise only"
    print "------------------------------------------------------------"

#--------------------------------------------------
ax = pylab.subplot(221,yscale='log')
label_axes('(a)',ax)

if noiseonly:
    tags = [ 'noiseonly_n',
             'noiseonly_y_normed',
             'noiseonly_y' ]
else:
    tags = [ 'noisy_n',
             'noisy_y_normed',
             'noisy_y' ]

plot_peak_func(filename(tags[0]),
               color=colors[0],
               label='unmasked',
               scale_by_noise = False,
               rmin=Mmin,
               rmax=Mmax,
               bins = bins)
plot_peak_func(filename(tags[1]),
               color=colors[1],
               label='weighted',
               scale_by_noise = False,
               rmin=Mmin,
               rmax=Mmax,
               bins = bins)
plot_peak_func(filename(tags[2]),
               color=colors[2],
               label='unweighted',
               scale_by_noise = False,
               rmin=Mmin,
               rmax=Mmax,
               bins = bins)
plot_peak_func(filename('perfect_n'),
               color='b',
               label='noiseless',
               scale_by_noise = False,
               rmin=Mmin,
               rmax=Mmax,
               bins = bins)
pylab.legend()
pylab.title('')
ax.xaxis.set_major_locator(MultipleLocator(0.01))
pylab.xlim(Mmin,Mmax)
pylab.ylim(1,1E4)

#--------------------------------------------------

#ax = pylab.subplot(223)
#ax = pylab.axes([0.125, 0.1, 0.352, 0.364])
ax = pylab.axes([0.125, 0.25, 0.352, 0.214])
label_axes('(c)',ax)

plot_peak_func(filename(tags[0]),
               color=colors[0],
               label='unmasked',
               scale_by_noise = True,
               cumulative = True,
               rmin = rmin,
               rmax = rmax,
               bins = bins)
plot_peak_func(filename(tags[1]),
               color=colors[1],
               label='weighted',
               scale_by_noise = True,
               cumulative = True,
               rmin = rmin,
               rmax = rmax,
               bins = bins)
plot_peak_func(filename(tags[2]),
               color=colors[2],
               label='unweighted',
               scale_by_noise = True,
               cumulative = True,
               rmin = rmin,
               rmax = rmax,
               bins = bins)
pylab.title('')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.xaxis.set_major_formatter(NullFormatter())

ylim = pylab.ylim()
pylab.xlim(rmin,rmax)
if noiseonly:
    pylab.ylim(1,2000)
else:
    pylab.ylim(1,ylim[1])

#--------------------------------------------------
L = ax.lines
xlabel = ax.get_xlabel()
pylab.xlabel('')
ax = pylab.axes([0.125, 0.1, 0.352, 0.15])
for i in range(3):
    E = L[2*i].get_ydata()
    B = L[2*i+1].get_ydata()
    x = L[2*i].get_xdata()
    pylab.plot(x,B/E,
               color=colors[i])
pylab.xlabel(xlabel)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
pylab.ylabel(r'$N_B/N_E$')

if noiseonly:
    pylab.plot([rmin,rmax],[1,1],':k')
    pylab.ylim(0.75,1.25)
else:
    pylab.ylim(0.0,0.65)
pylab.xlim(rmin,rmax)


#--------------------------------------------------

ax = pylab.subplot(222,yscale='log')
label_axes('(b)',ax)

if noiseonly:
    tags = ['900_n_a0.15_no','900_y_a0.15_no']
else:
    tags = ['900_n_a0.15','900_y_a0.15']
    

plot_peak_func(filename(tags[0]),
               color=colors[0],
               label='KL unmasked',
               scale_by_noise = False,
               rmin=Mmin,
               rmax=Mmax,
               bins = bins)
plot_peak_func(filename(tags[1]),
               color=colors[1],
               label='KL masked',
               scale_by_noise = False,
               rmin=Mmin,
               rmax=Mmax,
               bins = bins)
plot_peak_func(filename('perfect_n'),
               color='b',
               label='noiseless (no KL)',
               scale_by_noise = False,
               rmin=Mmin,
               rmax=Mmax,
               bins = bins)
pylab.legend()
pylab.title('')
ax.xaxis.set_major_locator(MultipleLocator(0.01))
pylab.xlim(Mmin,Mmax)
pylab.ylim(1,1E4)

#--------------------------------------------------

#ax = pylab.subplot(224)
#ax = pylab.axes([0.548,0.1,0.352,0.364])
ax = pylab.axes([0.548, 0.25, 0.352, 0.214])
label_axes('(d)',ax)

plot_peak_func(filename(tags[0]),
               color=colors[0],
               label='KL unmasked',
               scale_by_noise = True,
               cumulative = True,
               rmin = rmin,
               rmax = rmax,
               bins = bins)
plot_peak_func(filename(tags[1]),
               color=colors[1],
               label='KL masked',
               scale_by_noise = True,
               cumulative = True,
               rmin = rmin,
               rmax = rmax,
               bins = bins)
pylab.title('')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.xaxis.set_major_formatter(NullFormatter())

ylim = pylab.ylim()
if noiseonly:
    pylab.ylim(1,700)
else:
    pylab.ylim(1,ylim[1])
pylab.xlim(rmin,rmax)

#--------------------------------------------------
L = ax.lines
xlabel = ax.get_xlabel()
pylab.xlabel('')
ax = pylab.axes([0.548, 0.1, 0.352, 0.15])
for i in range(2):
    E = L[2*i].get_ydata()
    B = L[2*i+1].get_ydata()
    x = L[2*i].get_xdata()
    pylab.plot(x,B/E,colors[i])
pylab.xlabel(xlabel)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
pylab.ylabel(r'$N_B/N_E$')

if noiseonly:
    pylab.plot([rmin,rmax],[1,1],':k')
    pylab.ylim(0.75,1.25)
else:
    pylab.ylim(0.0,0.65)
pylab.xlim(rmin,rmax)

#--------------------------------------------------

#if noiseonly:
#    pylab.figtext(0.5,0.95,'Peak Functions: Noise Only',
#                  ha='center',va='center',
#                  fontdict={'size':18})
#    
#    pylab.savefig('figs/peak_comparison_noiseonly.pdf')
#    pylab.savefig('figs/peak_comparison_noiseonly.eps')
#
#else:
#    pylab.figtext(0.5,0.95,'Peak Functions',
#                  ha='center',va='center',
#                  fontdict={'size':18})
#
#    pylab.savefig('figs/peak_comparison.pdf')
#    pylab.savefig('figs/peak_comparison.eps')

pylab.show()
