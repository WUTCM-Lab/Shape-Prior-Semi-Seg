export CUDA_VISIBLE_DEVICES=1

nohup python -u main.py > train_gan.log --dataset tn3k --expID 1 --cuda 2>&1 &