import os
import sys
import getopt
import datetime
import numpy as np


try:
    opts, args = getopt.getopt(sys.argv[1:], ":i:e:p:s:") # [(opt, arg), (opt, arg), (opt, arg)]
except getopt.GetoptError:
    print("Option Error. Please conform to:")
    print("-i <clonal/mixed/all>")

for opt, value in opts:
    if opt == "-e":
        expt_name = value
        print(" Preparing Submission Script for Experiment:", expt_name)
    elif opt == "-i":
        iters = int(value)
        print(" Number of Iterations:", iters)
    elif opt == "-p":
        param_path = value
        print("  Parameter Path:", param_file)
    elif opt == "-s":
        seed_method = value
        print("Seed Method:", seed_method)
    else:
        print("Parameter %s not recognized." % opt)
        sys.exit(2)

out_dir = os.path.join("results", expt_name)
if not os.path.isdir(out_dir):
    print("  Making output directory for experiment:", out_dir)
    os.mkdir(out_dir)


print("  Generating `submit_simulations.py`")
ti = datetime.datetime.now()
f = open("submit_simulation.sh", "w")
f.write("\n# Submit a Suite of Simulations to Rescomp1")
f.write("\n# Generated at: %s/%s/%s %s:%s:%s" % (ti.day, ti.month, ti.year, ti.hour, ti.minute, ti.second))
f.write("\n# Experiment Name: %s" % expt_name)
f.write("\n# Iterations: %s" % iters)
f.write("\n# Parameter File: %s" % param_file)
f.write("\n# ----------------------")
f.write("\n")
for i in range(iters):
    f.write("qsub run_simulation.sh -e %s -p %s -s %s\n" % (expt_name, param_path, seed_method))
f.write("\n")
f.close()
os.chmod("submit_simulation.sh", 0o777)
print("Done.")
