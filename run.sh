export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
nohup python -u main_hyperparams.py > log 2>&1 &
tail -f log
