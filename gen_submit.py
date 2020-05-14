import os
import sys
import getopt
from datetime import datetime
import numpy as np


print("=" * 80)
print("Generating submission script for forward-dream")
print("-" * 80)
print("Command: %s" % " ".join(sys.argv))
print("Run on host: %s" % os.uname().nodename)
print("Operating system: %s" % os.uname().sysname)
print("Machine: %s" % os.uname().machine)
print("Started at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)

# PARSE CLI
print("Parsing user inputs...")
try:
    opts, args = getopt.getopt(sys.argv[1:], ":i:e:p:s:m:") # [(opt, arg), (opt, arg), (opt, arg)]
except getopt.GetoptError:
    print("Option Error. Please conform to:")
    print("-i <clonal/mixed/all>")

migration = False
for opt, value in opts:
    if opt == "-e":
        expt_name = value
        print("Experiment Name:", expt_name)
    elif opt == "-i":
        iters = int(value)
        print("Number of Iterations:", iters)
    elif opt == "-p":
        param_path = value
        print("Parameter Path:", param_path)
    elif opt == "-s":
        seed_method = value
        print("Seed Method:", seed_method)
    elif opt == "-m":
        migration = True
        migration_dir = value
        print("Migration Dir:", migration_dir)
    else:
        print("Parameter %s not recognized." % opt)
        sys.exit(2)
print("Done.")
print("")


# CREATE OUTPUT DIRECTORY
print("Creating output directory...")
out_dir = os.path.join("results", expt_name)
if not os.path.isdir(out_dir):
    print("Making output directory for experiment:", out_dir)
    os.mkdir(out_dir)
print("Done.")
print("")


# GENERATE SUBMISSION SCRIPT
print("Generating `submit_simulations.sh`...")
ti = datetime.now()
f = open("submit_simulation.sh", "w")
f.write("\n# Submit a Suite of Simulations to Rescomp1")
f.write("\n# Generated at: %s/%s/%s %s:%s:%s" % (ti.day, ti.month, ti.year, ti.hour, ti.minute, ti.second))
f.write("\n# Experiment Name: %s" % expt_name)
f.write("\n# Iterations: %s" % iters)
f.write("\n# Parameter File: %s" % param_path)
f.write("\n# ----------------------")
f.write("\n")
flags = "-e %s -p %s -s %s" % (expt_name, param_path, seed_method)
if migration:
    flags += " -m %s" % migration_dir
for i in range(iters):
    f.write("qsub run_simulation.sh %s\n" % flags)
f.write("\n")
f.close()
os.chmod("submit_simulation.sh", 0o777)
print("Done.")
print("")

print("-" * 80)
print("Finished at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)
