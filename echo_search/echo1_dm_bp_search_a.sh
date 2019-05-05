#! /bin/sh

# 5 way search
tau=$(seq 10 10 100)
tau_dm=$(seq 10 10 200)
seeds=$(seq 1000 1000 20000)
for tau_i in $tau; do
for tau_dm_i in $tau_dm; do
for seed_i in $seeds; do
CUDA_VISIBLE_DEVICES=3 python echo1_dm_bp_search.py \
                      --seed=$seed_i \
                      --tau=$tau_i \
                      --tau_dm=$tau_dm_i \
                      --num_dm=5 \
                      --rho_scale=0.9 \
                      --logdir="echo_bp_5way_a" \
                      --valid  &

#
CUDA_VISIBLE_DEVICES=3 python echo1_dm_bp_search.py \
                      --seed=$seed_i \
                      --tau=$tau_i \
                      --tau_dm=$tau_dm_i \
                      --num_dm=5 \
                      --rho_scale=1.1 \
                      --logdir="echo_bp_5way_a" \
                      --valid  &

#
CUDA_VISIBLE_DEVICES=3 python echo1_dm_bp_search.py \
                      --seed=$seed_i \
                      --tau=$tau_i \
                      --tau_dm=$tau_dm_i \
                      --num_dm=5 \
                      --rho_scale=1.5 \
                      --logdir="echo_bp_5way_a" \
                      --valid  &

#
CUDA_VISIBLE_DEVICES=3 python echo1_dm_bp_search.py \
                      --seed=$seed_i \
                      --tau=$tau_i \
                      --tau_dm=$tau_dm_i \
                      --num_dm=5 \
                      --rho_scale=2.0 \
                      --logdir="echo_bp_5way_a" \
                      --valid



done
done
done
































#
