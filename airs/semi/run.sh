# train
python -u code/main.py > log/train.log --mode train  --manner semi --ratio 2 --batch_size 32  --GPUs 1\
        --dataset tn3k  --expID 1 --ckpt_name 'tn3k_1' 2>&1 & # tn3k-1289

    

# test
# python code/main.py  --mode test  --manner test --load_ckpt best --GPUs 5\
#      --dataset BUSI --expID 1 --ckpt_name 'semi_self_mymodel_busi_72'
