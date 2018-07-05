#!/bin/bash
for run_id in 12 13 14 15 16 17 18 19 20;
# 16 17 18 19 20;
do
for cluster_id in 4 16 24 32;
do
echo "Run_id: "$run_id
echo "Custer_id: "$cluster_id
python training_with_mdp_fresh_agent.py --base_dir save_dir_py04/ --exp_desc 80_percent --config_params cluster_num__$cluster_id --run_num $run_id
done
done
