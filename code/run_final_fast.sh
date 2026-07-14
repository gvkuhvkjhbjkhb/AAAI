#!/usr/bin/env bash
# FINAL FAST RUN — maximal parallelism across 2 GPUs.
# A_remain: QQ(40) + GG(60) + QL(60) cells. D+E: 320 cells. Anti: 20 cells.
# B bos+pg: 160 cells. Total: ~640 cells.
#
# Strategy: hom workers (4-5GB) get packed 5-6/GPU, het (9GB) get 3/GPU.
# Stagger 90s for model loads. No orchestration overhead.
set -e
cd /data/lab/gsaca
export CUDA_VISIBLE_DEVICES=0; export GSACA=/data/lab/gsaca V2=/data/lab/results/v2
QWEN=Qwen/Qwen2.5-7B-Instruct
GLM=THUDM/GLM-4-9B-0414
LLAMA=NousResearch/Meta-Llama-3.1-8B-Instruct
CELLS4="hom_notom hom_gated_atom_talk hom_dp_gated_atom_talk hom_gsaca"
CELLS4HET="het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca"
ANTI_CELLS="het_gsaca het_role_asym het_hist_split het_adapt_interv het_combo_anti"
SEEDS5="42 43 44 45 46"
SEEDS20_1="42 43 44 45 46"; SEEDS20_2="47 48 49 50 51"; SEEDS20_3="52 53 54 55 56"; SEEDS20_4="57 58 59 60 61"

launch_gpu0() { CUDA_VISIBLE_DEVICES=0 nohup python3 run_experiment_local.py "$@" > /dev/null 2>&1 & }
launch_gpu1() { CUDA_VISIBLE_DEVICES=1 nohup python3 run_experiment_local.py "$@" > /dev/null 2>&1 & }

echo "Launching remaining A: QQ stag+bos, GG/QL deadlock+stag+bos..."
# === A_QQ stag_hunt (×5 seeds ×4 cells) ===
launch_gpu0 --games stag_hunt --seeds $SEEDS5 --episodes 30 --cells $CELLS4 --out_dir $V2/exp_a_fix/QQ --model_homo $QWEN --force --log_every 50
sleep 90
launch_gpu1 --games battle_of_the_sexes --seeds $SEEDS5 --episodes 30 --cells $CELLS4 --out_dir $V2/exp_a_fix/QQ --model_homo $QWEN --force --log_every 50
sleep 90
# === A_GG deadlock+stag+bos ===
launch_gpu0 --games deadlock --seeds $SEEDS5 --episodes 30 --cells $CELLS4 --out_dir $V2/exp_a_fix/GG --model_homo $GLM --force --log_every 50
sleep 90
launch_gpu1 --games stag_hunt --seeds $SEEDS5 --episodes 30 --cells $CELLS4 --out_dir $V2/exp_a_fix/GG --model_homo $GLM --force --log_every 50
sleep 90
launch_gpu0 --games battle_of_the_sexes --seeds $SEEDS5 --episodes 30 --cells $CELLS4 --out_dir $V2/exp_a_fix/GG --model_homo $GLM --force --log_every 50
sleep 90
# === A_QL deadlock+stag+bos ===
launch_gpu1 --games deadlock --seeds $SEEDS5 --episodes 30 --cells $CELLS4HET --out_dir $V2/exp_a_fix/QL --models_het $QWEN $LLAMA --force --log_every 50
sleep 90
launch_gpu0 --games stag_hunt --seeds $SEEDS5 --episodes 30 --cells $CELLS4HET --out_dir $V2/exp_a_fix/QL --models_het $QWEN $LLAMA --force --log_every 50
sleep 90
launch_gpu1 --games battle_of_the_sexes --seeds $SEEDS5 --episodes 30 --cells $CELLS4HET --out_dir $V2/exp_a_fix/QL --models_het $QWEN $LLAMA --force --log_every 50
sleep 90

echo "Anti-test: hawk_dove + deadlock remaining..."
# Anti-test on whichever GPU has room
launch_gpu0 --games hawk_dove --seeds $SEEDS5 --episodes 30 --cells $ANTI_CELLS --out_dir $V2/exp_anti_test --force --log_every 50
sleep 90
launch_gpu1 --games deadlock --seeds $SEEDS5 --episodes 30 --cells $ANTI_CELLS --out_dir $V2/exp_anti_test --force --log_every 50
sleep 90

echo "D1 noise sweep..."
for noise in 0.0 0.5 1.0 2.0; do
  tag="n${noise/./}"
  launch_gpu0 --games chicken hawk_dove deadlock stag_hunt battle_of_the_sexes --seeds $SEEDS5 --episodes 30 --cells het_gsaca --out_dir $V2/exp_d_fix/d1_$tag --models_het $QWEN $GLM --payoff_noise_std $noise --force --log_every 50
  sleep 90
  launch_gpu1 --games public_goods --seeds $SEEDS5 --episodes 20 --cells het_gsaca --out_dir $V2/exp_d_fix/d1_$tag --models_het $QWEN $GLM --payoff_noise_std $noise --force --log_every 50
  sleep 90
done

echo "D2 matching_pennies..."
launch_gpu0 --games matching_pennies --seeds $SEEDS5 --episodes 30 --cells $CELLS4HET --out_dir $V2/exp_d_fix/d2_QG --models_het $QWEN $GLM --force --log_every 50
sleep 90
launch_gpu1 --games matching_pennies --seeds $SEEDS5 --episodes 30 --cells $CELLS4HET --out_dir $V2/exp_d_fix/d2_QL --models_het $QWEN $LLAMA --force --log_every 50
sleep 90
launch_gpu0 --games matching_pennies --seeds $SEEDS5 --episodes 30 --cells $CELLS4 --out_dir $V2/exp_d_fix/d2_QQ --model_homo $QWEN --force --log_every 50
sleep 90

echo "E hyperparam: theta..."
for th in 0.3 0.45 0.6 0.75 0.9; do
  launch_gpu1 --games chicken battle_of_the_sexes --seeds $SEEDS5 --episodes 30 --cells het_dp_gated_atom_talk --out_dir $V2/exp_e_fix/theta_$th --models_het $QWEN $GLM --gate_trust_threshold $th --gate_ema_alpha 0.3 --force --log_every 50
  sleep 90
done

echo "E hyperparam: alpha..."
for al in 0.1 0.2 0.3 0.5; do
  launch_gpu0 --games chicken battle_of_the_sexes --seeds $SEEDS5 --episodes 30 --cells het_dp_gated_atom_talk --out_dir $V2/exp_e_fix/alpha_$al --models_het $QWEN $GLM --gate_ema_alpha $al --gate_trust_threshold 0.6 --force --log_every 50
  sleep 90
done

echo "E hyperparam: W (gsaca_warmup)..."
for W in 2 3 5 8 10; do
  launch_gpu1 --games chicken battle_of_the_sexes --seeds $SEEDS5 --episodes 30 --cells het_gsaca --out_dir $V2/exp_e_fix/warmup_$W --models_het $QWEN $GLM --gsaca_warmup $W --force --log_every 50
  sleep 90
done

echo "B battle_of_the_sexes + public_goods 20-seed fix..."
for seedgrp in "$SEEDS20_1" "$SEEDS20_2" "$SEEDS20_3" "$SEEDS20_4"; do
  launch_gpu0 --games battle_of_the_sexes --seeds $seedgrp --episodes 30 --cells $CELLS4HET --out_dir $V2/exp_b_20seed --models_het $QWEN $GLM --force --log_every 50
  sleep 90
  launch_gpu1 --games public_goods --seeds $seedgrp --episodes 20 --cells $CELLS4HET --out_dir $V2/exp_b_20seed --models_het $QWEN $GLM --force --log_every 50
  sleep 90
done

echo "ALL LAUNCHED at $(date). Wait for completion..."
wait
echo "DONE at $(date)"