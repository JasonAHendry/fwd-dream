#/bin/bash

# Generate and submit a series of simulations
# varying across `vary_param`; each with there own
# parameter value, under a common experiment name


while [[ $# -ge 1 ]]; do
	tag=$1
	case $tag in
		-e)
			expt_name=$2
			echo "Experiment Name:" $expt_name
			shift
			;;
		-v)
			vary_param=$2
			echo "Varying Parameter:" $vary_param
			shift
			;;
		-i)
			iters=$2
			echo "Number of Iterations:" $iters
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

param_sets=$(ls params/param_$vary_param*)
echo "Preparing to submit correlation experiment."
echo " Experiment Name:" $expt_name
echo " Varying Parameter:" $vary_param
echo " No. Param Sets:" `echo $param_sets | wc -w`
echo " Go..."
for param_set in $param_sets; do
  echo "  Parameter Set:" $param_set
  echo "  Generating Submission Script..."
  python gen_submit.py -e $expt_name -p $param_set -i $iters -s $seed_method
  echo "  Submitting..."
  ./submit_simulation.sh
done

