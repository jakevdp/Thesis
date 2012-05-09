import pylab
from matplotlib.ticker import *

class MultiAxesIterator:
   def __init__(self,
                xmin,xmax,Nx,
                ymin,ymax,Ny):
       self.xmin = xmin
       self.Nx = Nx
       self.ymin = ymin
       self.Ny = Ny

       self.dx = (xmax-xmin)*1./Nx
       self.dy = (ymax-ymin)*1./Ny

       self.i = -1
       self.j = 0
      
   def __iter__(self):
       return self
  
   def next(self):
       self.i += 1
       if self.i>=self.Nx:
           self.i = 0
           self.j += 1

       if self.j<self.Ny:
           xmin = self.xmin + self.i*self.dx
           ymin = self.ymin + (self.Ny-1-self.j)*self.dy
           ax = pylab.axes( (xmin,ymin,self.dx,self.dy) )
           ax.xaxis.set_major_formatter(NullFormatter())
           ax.yaxis.set_major_formatter(NullFormatter())
           return ax
       else:
           raise StopIteration

def test_MultiAxes():
    pylab.figure(figsize=(10,6))
    
    i=0
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    MA = MultiAxesIterator(0.1,0.9,4,
                           0.1,0.9,4)
    
    for ax in MA:
        pylab.text(0.5,0.5,'%s: (%i,%i)' % (alphabet[i],MA.i,MA.j),
                   fontsize = 20,
                   ha='center',va='center',
                   transform = ax.transAxes)

        if MA.j==3:
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_major_locator(FixedLocator( (0,0.2,0.4,0.6,0.8) ))
        i+=1
    pylab.show()
