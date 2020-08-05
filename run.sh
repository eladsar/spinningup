#!/usr/bin/env bash

#ALG="gegl"

ALG="sac"

NAME="eval_full_rb"
#NAME=$1
AUX="--epochs 100 --seed 12 24 36 48 60"
#AUX="--epochs 250"

#GROUP1=" --seed 12"
#GROUP2=" --seed 24"
#GROUP3=" --seed 36"
#GROUP4=" --seed 48"

GROUP4=" "


CUDA_VISIBLE_DEVICES=0, python spinup/run.py $ALG --env InvertedDoublePendulum-v2 --exp_name $NAME $AUX $GROUP4 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=1, python spinup/run.py $ALG --env InvertedPendulum-v2 --exp_name $NAME $AUX $GROUP4 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env Reacher-v2 --exp_name $NAME $AUX $GROUP4 > /dev/null &
sleep 1
CUDA_VISIBLE_DEVICES=3, python spinup/run.py $ALG --env Swimmer-v2 --exp_name $NAME $AUX $GROUP4 &


#CUDA_VISIBLE_DEVICES=0, python spinup/run.py $ALG --env Hopper-v2 --exp_name $NAME $AUX $GROUP1 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=1, python spinup/run.py $ALG --env Walker2d-v2 --exp_name $NAME $AUX $GROUP1 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env HalfCheetah-v2 --exp_name $NAME $AUX $GROUP1 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=3, python spinup/run.py $ALG --env Ant-v2 --exp_name $NAME $AUX $GROUP1 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=0, python spinup/run.py $ALG --env Humanoid-v2 --exp_name $NAME $AUX $GROUP1 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=1, python spinup/run.py $ALG --env Hopper-v2 --exp_name $NAME $AUX $GROUP2 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env Walker2d-v2 --exp_name $NAME $AUX $GROUP2 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=3, python spinup/run.py $ALG --env HalfCheetah-v2 --exp_name $NAME $AUX $GROUP2 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=0, python spinup/run.py $ALG --env Ant-v2 --exp_name $NAME $AUX $GROUP2 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=1, python spinup/run.py $ALG --env Humanoid-v2 --exp_name $NAME $AUX $GROUP2 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env Hopper-v2 --exp_name $NAME $AUX $GROUP3 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=3, python spinup/run.py $ALG --env Walker2d-v2 --exp_name $NAME $AUX $GROUP3 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=0, python spinup/run.py $ALG --env HalfCheetah-v2 --exp_name $NAME $AUX $GROUP3 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=1, python spinup/run.py $ALG --env Ant-v2 --exp_name $NAME $AUX $GROUP3 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env Humanoid-v2 --exp_name $NAME $GROUP3 $AUX > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env Hopper-v2 --exp_name $NAME $AUX $GROUP4 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=0, python spinup/run.py $ALG --env Walker2d-v2 --exp_name $NAME $AUX $GROUP4 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=1, python spinup/run.py $ALG --env HalfCheetah-v2 --exp_name $NAME $AUX $GROUP4 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=2, python spinup/run.py $ALG --env Ant-v2 --exp_name $NAME $AUX $GROUP4 > /dev/null &
#sleep 1
#CUDA_VISIBLE_DEVICES=3, python spinup/run.py $ALG --env Humanoid-v2 --exp_name $NAME $GROUP4 $AUX &