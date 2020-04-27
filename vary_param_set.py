import os
import sys

import configparser
import getopt

# PARSE CLI INPUT
vals = []
try:
    opts, args = getopt.getopt(sys.argv[1:], ":f:p:v:")
except getopt.GetoptError:
    print("Option Error. Please conform to:")
    print("-v <int>")
for opt, value in opts:
    if opt == "-f":
        param_file = value
    elif opt == "-p":
        param = value
    elif opt == "-v":
        if param == "nv" or param == "nh":
            value = int(value)
        else:
            value = float(value)
        vals.append(value)
    else:
        print("Parameter %s not recognized." % opt)
        sys.exit(2)


# LOAD PARAMETER FILE
param_path = os.path.join("params", param_file)
config = configparser.ConfigParser()
config.read(param_path)
# Find section that has parameter to modify
section = [s for s in config.sections() if config.has_option(s, param)][0]
orig_val = config.get(section, param)
print("Parameter File:", param_file)
print("  Param to modify:", param)
print("   from section:", section)
print("  Original Value:", orig_val)
print("  New range:", vals)
print("  Modifying...")
for val_ix, val in enumerate(vals):
    config.set(section, param, str(val))
    val_name = "%s%02d" % (param, val_ix)  # this is preferable to the value, as it may be a float
    file_name = "param_" + val_name + ".ini"
    print("   %s = %s to %s" % (param, val, file_name))
    with open(os.path.join("params", file_name), "w") as config_file:
        config.write(config_file)
print("Done.")
