import os
import sys
import glob
import excons
from excons.tools import arnold
from excons.tools import ilmbase
from excons.tools import boost
from excons.tools import hdf5
from excons.tools import dl

defs = []
incdirs = []
libdirs = []
libs = []
customs = []

# Field3D configuration
field3d_inc, field3d_lib = excons.GetDirs("field3d")
if not field3d_inc and not field3d_lib:
  # Build Field3D as a static lib
  field3d_static = True
  excons.SetArgument("field3d-static", 1)
    
  SConscript("Field3D/SConstruct")

else:
  field3d_static = (excons.GetArgument("field3d-static", 0, int) != 0)

if field3d_inc:
  incdirs.append(field3d_inc)

if field3d_lib:
  libdirs.append(field3d_lib)

libs.append("Field3D")

if field3d_static:
  defs.append("FIELD3D_STATIC")

if sys.platform == "win32":
  defs.append("NO_TTY")

customs = [hdf5.Require(hl=False, verbose=True),
           ilmbase.Require(ilmthread=False, iexmath=False),
           boost.Require(libs=["system", "regex"]),
           arnold.Require]

test = (excons.GetArgument("test", 0, int) == 1)
if test:
  defs.append("ARNOLD_F3D_TEST")
  customs.append(dl.Require)

target = {"name": "arnold-%s/volume_field3d" % arnold.Version(),
          "alias": "arnold",
          "type": ("program" if test else "dynamicmodule"),
          "defs": defs,
          "srcs": glob.glob("src/*.cpp"),
          "incdirs": incdirs,
          "libdirs": libdirs,
          "libs": libs,
          "custom": customs}

if not test:
  target["ext"] = arnold.PluginExt()

env = excons.MakeBaseEnv()

excons.DeclareTargets(env, [target])

Default(["arnold"])
