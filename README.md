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
- Prefer static builds of the dependencies

# Usage

Data string flags:

- **-file {file}**: Path to Field3D sequence. Frame pattern can be specified using sharp ('#') letters (their count is used as padding, i.e. ### for a 3 padded frame number), printf like syntax (%d, %03d) or <<frame:N>> tokens, where ':N' is optional and sets the padding for the frame number if specified.
- **-partition {partition}**: Retrict to read fields from a specific partition.
- **-verbose**: Enable verbose mode (console).
- **-ignoreXform**: Ignore fields mapping.
- **-fps {fps}**: Set framerate. If not specified, a 'fps' attribute is looked up on the options node. Defaults to 24.
- **-frame {frame}**: Set the frame the be read. If not specified, a 'frame' attribute is looked up on the options node. Defaults to 1.
- **-merge field0=add|min|max|average ... fieldN=add|min|max|average**: Specify or to merge overlapping fields values. Defaults to add for all fields. The only special case if for the velocity field used for motion blur (as specified below), when it is always 'average'.
- **-motionStartFrame {relative_frame}**: Start of the motion range in frames relative to current frame. Defaults to current frame.
- **-motionEndFrame {relative_frame}**: End of the motion range in frames relative to current frame. Defaults to current frame.
- **-shutterTimeType normalized|frame_relative|absolute_frame**: Specify how to interpret the arnold time values (sg->time). 'normalized' mode remaps motionStartFrame to 0 and motionEndFrame to 1. Defaults to 'normalized'.
- **-velocityField {fields}**: The name of 1 vector field or 3 scalar fields to use for the velocity.
- **-velocityScale {scale}**: Global velocity scale. Default to 1.
- **-worldSpaceVelocity**: The values read from the velocity field(s) are expressed in volume's world space.

Any of those flags can be overridden using constant user attributes named after the flag.

Here is the list of accepted types for each of the parameters:

- **file**: STRING
- **partition**: STRING
- **verboes**: BOOLEAN, BYTE, INT, UINT
- **ignoreXform**: BOOLEAN, BYTE, INT, UINT
- **frame**: FLOAT, INT, UINT, BYTE
- **merge**: STRING, STRING[]
- **motionStartFrame**: FLOAT, INT, UINT, BYTE
- **motionEndFrame**: FLOAT, INT, UINT, BYTE
- **shutterTimeType**: STRING
- **velocityField**: STRING, STRING[]
- **velocityScale**: FLOAT, INT, UINT, BYTE
- **worldSpaceVelocity**: BOOLEAN, BYTE, INT, UINT

## MtoA

Use aiVolume node in 'Custom' mode and set the DSO path to volume_field3d

Either set the data string or user maya custom attributes prefixing names described above with mtoa_constant_


