cd baselines/BotDGT

python3 preprocess.py

python3 train.py --interval year --window_size 5 --epoch 30 --early_stop