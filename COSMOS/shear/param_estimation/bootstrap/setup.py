from numpy.distutils.core import setup
from numpy.distutils.extension import Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("bootstrap", ["bootstrap.pyx"])],
    include_dirs = get_numpy_include_dirs()
)
