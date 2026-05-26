cd baselines/DeeProBot

python3 preprocess.py

python3 train.py --epochs 300 --lr 2e-3 --batch_size 64 --weighted_loss