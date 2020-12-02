#/bin/bash

#$ -N sim
#$ -P mcvean.prjc -q short.qc
#$ -o stdout_sim -e stderr_sim -j y
#$ -cwd -V

### Submitting a forward-dream simulation to Rescomp1
## JHendry, 2017/12/12

echo "**********************************************************************"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
SECONDS=0
echo "**********************************************************************"

analysis_settings=$@ # this is the entire command line input
echo "Analysis Settings:" $analysis_settings

while [[ $# -ge 1 ]]
do
  tag=$1
  case $tag in
    -e)
	  expt_name=$2
	  echo "Experiment Name:" $expt_name
	  shift
	  ;;
	-p)
	  param_path=$2
	  echo "Parameter Path:" $param_path
	  shift
	  ;;
	-s)
	  seed_method=$2
	  echo "Seed Method:" $seed_method
	  shift
	  ;;
    -m)
	  migration_dir=$2
	  echo "Migration Directory:" $migration_dir
	  shift
	  ;;
	*)
	  ;;
  esac
  shift
done

echo "Running simulation.py..."
python simulation.py $analysis_settings
echo "simulation.py complete."


echo "**********************************************************************"
echo "Finished at: "`date`
printf 'Time elapsed: %dh:%dm:%ds\n' $(($SECONDS/3600)) $(($SECONDS%3600/60)) $(($SECONDS%60))
echo "**********************************************************************"
echo ""
echo ""
echo ""
