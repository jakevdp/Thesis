import numpy as np
import pylab as pl
from matplotlib.patches import Polygon

def draw_line(start, end, frac, *args, **kwargs):
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    pl.plot([start[0] + frac[0] * dx, start[0] + frac[1] * dx],
            [start[1] + frac[0] * dy, start[1] + frac[1] * dy],
            *args, **kwargs)

fig = pl.figure(figsize=(8, 4), facecolor='w')
ax = pl.axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])

plane_array = np.array([[0.8, 0.3],
                        [1.0, 0.1],
                        [1.0, 0.7],
                        [0.8, 0.9]])

plane_array_2 = plane_array.copy()
plane_array_2[:, 0] += 0.9

ax.add_patch(Polygon(plane_array, fc='none', ec='k', lw=1))
ax.add_patch(Polygon(plane_array_2, fc='none', ec='k', lw=1))

draw_line([0.05, 0.65], [0.86, 0.65],
          [0, 1], '--k')
draw_line([1.0, 0.65], [1.76, 0.65],
          [0, 1], '--k')

frac = 0.5
draw_line([0.05, 0.65], [0.86 + 0.04, 0.65 - 0.3],
          [0, 1], '-k')
draw_line([0.05, 0.65], [0.86 + 0.04, 0.65 - 0.3],
          [1.12, 1.7], ':k')
draw_line([0.86 + 0.04, 0.65 - 0.3], [1.76 + frac * 0.04, 0.65 - frac * 0.3],
          [0.12, 1], '-k')

draw_line([0.86, 0.65], [0.86 + 0.04, 0.65 - 0.3],
          [0, 1], '-k')
draw_line([1.76, 0.65], [1.76 + frac * 0.04, 0.65 - frac * 0.3],
          [0, 1], '-k')

draw_line([0.05, 0.65], [1.76 + frac * 0.04, 0.65 - frac * 0.3],
          [0, 0.48], ':k')
draw_line([0.05, 0.65], [1.76 + frac * 0.04, 0.65 - frac * 0.3],
          [0.55, 1.0], ':k')

pl.annotate(' ', [0.05, 0.95], [1.76, 0.95], arrowprops=dict(arrowstyle='<->'))

pl.text(0.5, 0.65, r'$\rm D_{L}$', fontsize=14)
pl.text(1.2, 0.65, r'$\rm D_{LS}$', fontsize=14)
pl.text(1.0, 0.95, r'$\rm D_{S}$', fontsize=14)

pl.annotate(' ', (1.2, 0.24),  (1.22, 0.41),
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=-0.3"))
pl.text(1.25, 0.3, r'$\rm \hat{\alpha}$', fontsize=14)

pl.annotate(' ', (0.35, 0.54),  (0.37, 0.66),
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=-0.2"))
pl.text(0.38, 0.55, r'$\rm \theta$', fontsize=14)

pl.annotate(' ', (0.62, 0.59),  (0.63, 0.66),
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=-0.1"))
pl.text(0.65, 0.59, r'$\rm \beta$', fontsize=14)

pl.text(0.78, 0.12, "lens\nplane")
pl.text(1.65, 0.12, "source\nplane")

pl.xlim(0, 2)
pl.ylim(0.05, 1.05)

pl.savefig('lensing_geometry.eps')
pl.show()
