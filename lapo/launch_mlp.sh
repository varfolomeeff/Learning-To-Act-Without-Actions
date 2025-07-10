#!/bin/bash
# which micromamba


eval "$(micromamba shell hook --shell=bash)"
micromamba activate lapo
declare -A tasks=(
	[0]="bigfish"
	# [1]="bossfight"
	# [2]="caveflyer"
	# [3]="chaser"
	# [4]="climber"
	# [5]="coinrun"
	# [6]="dodgeball"
	# [7]="fruitbot"
	# [8]="heist"state_dicts
	# [9]="jumper"
	# [10]="leaper"
	# [11]="maze"
	# [12]="miner"
	# [13]="ninja"
	# [14]="plunder"
	# [15]="starpilot"
)

# sweep_name=$(python -c "import doy; print(doy.random_proquint(2))")
# echo $sweep_name
# sweep_name="davup-bigof"
sweep_name="kanip-lipub"
# sweep_name="test_$(date +%d_%m_%Y_%H_%M_%S)"
# for ind in {0..5}; do
ind=0

gpu=$1  # Set your desired GPU index here
echo $gpu
	# generate a random experiment name that's the same across stages 1-3
	exp_name="${ind}_${sweep_name}"
	echo $exp_name
tr_list=(16 64 256 1024 4096)
# for ind in "${tr_list[@]}"; do
	# python extract_actions.py --N $ind
	# sleep 3
	# echo "stage 1 done"
	# # NOTE: mlp_mapping.py should parse env_name and exp_name from sys.argv as key=value
	python mlp_mapping.py gpu="$gpu" --npz 256 --env_name "${tasks[0]}" --exp_name "${exp_name}" --base_exp_name "${exp_name}"
	echo "stage 2 done"
	# sleep 3
	# python evaluate.py "${exp_name}" 4096 "bigfish" 100 cuda:1
	# sleep 3
	# echo "stage 3 done"
	# sleep 10
# done