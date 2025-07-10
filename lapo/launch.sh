#!/bin/bash
# which micromamba


eval "$(micromamba shell hook --shell=bash)"
micromamba activate lapo
declare -A tasks=(
	# [0]="bigfish"
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
	[0]="maze"
	# [12]="miner"
	# [13]="ninja"
	# [14]="plunder"
	# [15]="starpilot"
)

# sweep_name=$(python -c "import doy; print(doy.random_proquint(2))")
# echo $sweep_name
# sweep_name="davup-bigof"
sweep_name="maze_second_run"
# sweep_name="test_$(date +%d_%m_%Y_%H_%M_%S)"
# for ind in {0..5}; do
ind=0

gpu=$1  # Set your desired GPU index here
echo $gpu
	# generate a random experiment name that's the same across stages 1-3
	exp_name="${ind}_${sweep_name}"
	echo $exp_name
# python stage1_idm.py env_name="${tasks[${ind}]}" exp_name="${exp_name}" gpu="$gpu" 
python stage2_bc.py env_name="${tasks[${ind}]}" exp_name="${exp_name}" gpu="$gpu"
# python stage3_decoding.py env_name="${tasks[${ind}]}" exp_name="${exp_name}" gpu="$gpu"
# python mlp_mapping.py env_name="${tasks[${ind}]}" exp_name="${exp_name}" gpu="$gpu"