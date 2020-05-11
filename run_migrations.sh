#/bin/bash

# Generate and submit a series of
# migration-sink simulations based
# on an already run set of migration-
# source simulations

echo "**********************************************************************"
echo "Command: " $0 $*
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
SECONDS=0
echo "**********************************************************************"

while [[ $# -ge 1 ]]; do
	tag=$1
	case $tag in
		-e)
			expt_name=$2
			echo "Experiment Name:" $expt_name
			shift
			;;
		-m)
			mig_source_dir=$2
			echo "Migration Source Directory:" $mig_source_dir
			shift
			;;
		-i)
			iters=$2
			echo "Number of Iterations:" $iters
			shift
			;;
		-p)
			param_set=$2
			echo "Parameter Set:" $param_set
			shift
			;;
		-s)
    			seed_method=$2
    			echo "Seed Method:" $seed_method
    			shift
    			;;
		*)
			echo "Error, unidentified tag."
			exit 1
			;;
	esac
	shift  # past the tag value
done

mig_sims=$(ls $mig_source_dir)
echo "Preparing to submit migration experiments..."
echo " No. Migration Source Simulations:" `echo $mig_sims | wc -w`
echo " Running..."
for mig_sim in $mig_sims; do
  python gen_submit.py -e $expt_name -p $param_set -m $mig_source_dir$mig_sim -i $iters -s $seed_method
  echo "  Submitting..."
  ./submit_simulation.sh
  echo "  Done."
  echo ""
done
echo "Done."
echo ""


echo "**********************************************************************"
echo "Finished at: "`date`
printf 'Time elapsed: %dh:%dm:%ds\n' $(($SECONDS/3600)) $(($SECONDS%3600/60)) $(($SECONDS%60))
echo "**********************************************************************"
echo ""
echo ""
echo ""
