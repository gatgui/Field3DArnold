# Field3DArnold
A Field3D volume procedural for Arnold renderer

# Build
scons (OPTIONS) -jN arnold

## Options
- with-arnold={arnold_prefix}
- with-arnold-inc={arnold_incdir}
- with-arnold-lib={arnold_libdir}
- with-hdf5={hdf5_prefix}
- with-hdf5-inc={hdf5_incdir}
- with-hdf5-lib={hdf5_libdir}
- hdf5-static=0|1
- with-ilmbase={ilmbase_prefix}
- with-ilmbase-inc={ilmbase_incdir}
- with-ilmbase-lib={ilmbase_libdir}
- ilmbase-static=0|1
- ilmbase-libsuffix={ilmbase_suffix}
- with-boost={boost_prefix}
- with-boost-inc={boost_incdir}
- with-boost-lib={boost_libdir}
- boost-static=0|1
- boost-libsuffix={boost_suffix} (i.e.: "-mt")
- use-c++11=0|1 (OSX >= 10.9)
- use-stdc++=0|1 (OSX >= 10.9)
- warnings=none|std|all
- debug=0|1

## Notes
- As Field3D does handle thread safety by itself, don't use the thread safe version for HDF5 library

# TODO
- read procedural parameters from node's user attributes
- motion blur using velocity grid(s)
