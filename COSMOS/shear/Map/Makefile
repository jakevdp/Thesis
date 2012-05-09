all: peaks

peaks: peaks.pyx
	python setup.py build_ext --inplace

clean:
	rm peaks.so
	rm *~
	rm -rf build