#!/bin/bash
for run_id in 1 2 3 4 5 6 7;
# 3 4 5 6 7 8 9 10;
do
for cluster_id in 32; 
do
echo "Run_id: "$run_id
echo "Custer_id: "$cluster_id
python training_with_state_agg_buf.py --base_dir save_dir_py04/ --exp_desc 80_percent --config_params cluster_num__$cluster_id --run_num $run_id
done
done
