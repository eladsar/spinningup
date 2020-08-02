#!/usr/bin/env bash

ALG="egl"
NAME="debug_no_q_target_multiseed"
#AUX="--epochs 250 --seed 12 --eps 0.03 0.1 0.3 1"
AUX="--epochs 250"

GROUP1=" --seed 12"
GROUP2=" --seed 24"
GROUP3=" --seed 36"
GROUP4=" --seed 48"

CUDA_VISIBLE_DEVICES=0, python spinup/run.py $ALG --env Hopper-v2 --exp_name $NAME $AUX $GROUP1 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=1, python spinup/run.py $ALG --env Walker2d-v2 --exp_name $NAME $AUX $GROUP1 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env HalfCheetah-v2 --exp_name $NAME $AUX $GROUP1 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=3, python spinup/run.py $ALG --env Ant-v2 --exp_name $NAME $AUX $GROUP1 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=0, python spinup/run.py $ALG --env Humanoid-v2 --exp_name $NAME $AUX $GROUP1 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=1, python spinup/run.py $ALG --env Hopper-v2 --exp_name $NAME $AUX $GROUP2 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env Walker2d-v2 --exp_name $NAME $AUX $GROUP2 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=3, python spinup/run.py $ALG --env HalfCheetah-v2 --exp_name $NAME $AUX $GROUP2 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=0, python spinup/run.py $ALG --env Ant-v2 --exp_name $NAME $AUX $GROUP2 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=1, python spinup/run.py $ALG --env Humanoid-v2 --exp_name $NAME $AUX $GROUP2 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env Hopper-v2 --exp_name $NAME $AUX $GROUP3 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=3, python spinup/run.py $ALG --env Walker2d-v2 --exp_name $NAME $AUX $GROUP3 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=0, python spinup/run.py $ALG --env HalfCheetah-v2 --exp_name $NAME $AUX $GROUP3 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=1, python spinup/run.py $ALG --env Ant-v2 --exp_name $NAME $AUX $GROUP3 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env Humanoid-v2 --exp_name $NAME $GROUP3 $AUX > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=3, python spinup/run.py $ALG --env Hopper-v2 --exp_name $NAME $AUX $GROUP4 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=0, python spinup/run.py $ALG --env Walker2d-v2 --exp_name $NAME $AUX $GROUP4 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=1, python spinup/run.py $ALG --env HalfCheetah-v2 --exp_name $NAME $AUX $GROUP4 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env Ant-v2 --exp_name $NAME $AUX $GROUP4 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=3, python spinup/run.py $ALG --env Humanoid-v2 --exp_name $NAME $GROUP4 $AUX &