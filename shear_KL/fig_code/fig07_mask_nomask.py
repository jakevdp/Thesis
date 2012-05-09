import os
import sys
import pylab
from matplotlib.ticker import MultipleLocator, NullFormatter

sys.path.append(os.path.abspath('../'))
from shear_KL_source import params
params.load('../run/base_params.dat')

sys.path.append('../fig_code')
from plot_Map_by_name import plot_peak_func

filename = lambda s: os.path.join(params.shear_recons_dir,'Map_'+s+'.npz')

rmin = 3.25
rmax = 6.5
Mmin = 0.01
Mmax = 0.04
bins = 40

colors = 'krg'

pylab.figure(figsize=(10,4.5))

ax = pylab.axes( (0.1,0.13,0.43,0.77),yscale='log')

tags = [ 'noisy_n',
         'noisy_y_normed',
         'noisy_y' ]

labels = ['unmasked','weighted','unweighted']

lines = []

for i in range(3):
    l = plot_peak_func(filename(tags[i]),
                       color=colors[i],
                       label=labels[i],
                       scale_by_noise = False,
                       rmin=Mmin,
                       rmax=Mmax,
                       bins = bins)
    lines.append(l)
    
leg = pylab.legend(lines[0],['E-mode','B-mode'],loc=3,frameon=False,)
pylab.legend()
ax.add_artist(leg)

pylab.xlim(0.01,0.039)
pylab.title('Without KL')

ax = pylab.axes( (0.53,0.13,0.43,0.77),yscale='log')

tags = ['900_n_a0.15y','900_y_a0.15y']
labels = ['KL umasked','KL masked']

for i in range(2):
    plot_peak_func(filename(tags[i]),
                   color=colors[i],
                   label=labels[i],
                   scale_by_noise = False,
                   rmin=Mmin,
                   rmax=Mmax,
                   bins = bins)
    
pylab.legend()
    
pylab.ylabel('')
ax.yaxis.set_major_formatter(NullFormatter())
pylab.title('With KL')

pylab.savefig('figs/fig07_mask_vs_nomask.pdf')
pylab.savefig('figs/fig07_mask_vs_nomask.eps')
pylab.show()
