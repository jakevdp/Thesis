import numpy
from scipy import integrate

def integrate_onecall_2D(f,
                         xmin,xmax,Nx,
                         ymin,ymax,Ny):
    x = numpy.linspace(xmin,xmax,Nx)
    y = numpy.linspace(ymin,ymax,Ny)
    X,Y = numpy.meshgrid(x,y)

    F = f(X,Y)

    I = integrate.simps(F,dx=x[1]-x[0],axis=1)
    return integrate.simps(I,dx = y[1]-y[0])

def integrate_log_2D(f,
                     xmin,xmax,Nx,
                     ymin,ymax,Ny):
    """
    integrate in logspace using a single function call and simpsons rule
      int{ f(x) dx }  =>  int{ f'(x') dx' }
        with x' = log(x)
    """
    f_star = lambda x,y: numpy.exp(x+y)*f(numpy.exp(x),numpy.exp(y))
    return integrate_onecall_2D(f_star,
                                numpy.log(xmin),numpy.log(xmax),Nx,
                                numpy.log(xmin),numpy.log(xmax),Ny)
                               

    
def integrate_lin(f,xmin,xmax,N=10000):
    x = numpy.linspace(xmin,xmax,N)
    return integrate.simps( f(x),dx=x[1]-x[0] )

def integrate_log(f,xmin,xmax,N=10000):
    f_star = lambda y: numpy.exp(y)*f(numpy.exp(y))
    return integrate_lin(f_star,numpy.log(xmin),numpy.log(xmax))

def test_integ_log():
    f = lambda x: x*numpy.sin(x)
    print integrate_lin(f,1,100)
    print integrate_log(f,1,100)


if __name__ == '__main__':
    f = lambda x,y: numpy.sin(x)*numpy.sin(y)

    print integrate_onecall_2D(f,numpy.pi,3*numpy.pi,1000,
                               numpy.pi,3*numpy.pi,1000)
    print integrate_log_2D(f,numpy.pi,3*numpy.pi,1000,
                           numpy.pi,3*numpy.pi,1000)
