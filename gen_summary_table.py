import os
import sys
import getopt
from datetime import datetime
import pandas as pd


print("=" * 80)
print("Combine all simulation results from a forward-dream experiment")
print("-" * 80)
print("Command: %s" % " ".join(sys.argv))
print("Run on host: %s" % os.uname().nodename)
print("Operating system: %s" % os.uname().sysname)
print("Machine: %s" % os.uname().machine)
print("Started at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)


# Parse user inputs
print("Parsing user inputs...")
try:
    opts, args = getopt.getopt(sys.argv[1:], ":e:")
except getopt.GetoptError:
    print("Option error. Please conform to:")
    print("-e <experiment_name>")

for opt, value in opts:
    if opt == "-e":
        expt_name = value
        print("  Generating summary table for:", expt_name)
    else:
        print("Parameter %s not recognized." % opt)
        sys.exit(2)
print("Done.")
print("")

        
# Directories
print("Preparing directories...")
expt_dir = os.path.join("results", expt_name)
output_dir = "summary_tables"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("  Input directory:", expt_dir)
print("Done.")
print("")
    
    
# Combining all simulations in experiment
print("Combining all simulation results...")
dfs = []
e = 0
print("  Number of simulations: %d" % len(os.listdir(expt_dir)))
for sim in os.listdir(expt_dir):
    sim_path = os.path.join(expt_dir, sim)
    if "epoch_df.csv" in os.listdir(sim_path):
        # Load data
        og = pd.read_csv(os.path.join(sim_path, "og.csv"))
        op = pd.read_csv(os.path.join(sim_path, "op.csv"))
        epoch_df = pd.read_csv(os.path.join(sim_path, "epoch_df.csv"))
        
        # Annotate epochs
        t0s = op.t0
        epoch = []
        x_h = []
        x_v = []
        for _, row in epoch_df.iterrows():
            k = ((row["t0"] <= t0s) & (t0s <= row["t1"])).sum()
            epoch.extend([row["name"]]*k)
            x_h.extend([row["x_h"]]*k)
            x_v.extend([row["x_v"]]*k)
        op["epoch"] = epoch
        op["HX_expected"] = x_h
        op["VX_expected"] = x_v
        
        # Merge
        df = pd.merge(left=op,
              right=og,
              on="t0",
              how="outer")
        df["id"] = sim
        
        # Store
        dfs.append(df)
        
    else:
        e += 1
        print("  Extinction occured in %s." % sim)
print("  Number of extinctions: %d" % e)

final_df = pd.concat(dfs)

print("Done.")
print("")


# Write output
print("Storing results...")
output_path = os.path.join(output_dir, expt_name + ".csv")
final_df.to_csv(output_path)
print("  Output:", output_path)
print("Done.")
print("")

    
print("-" * 80)
print("Finished at: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 80)

