cd baselines-moltbook/BotDGT
python preprocess.py
```


python evaluate.py --device cpu --model-path ../../baselines-weibo/BotDGT/output/Weibo/month+42+0.74468+0.68746.pt
python evaluate.py --device cpu --model-path ../../baselines-weibo/BotDGT/output/Weibo/month+42+0.74232+0.68534.pt
python evaluate.py --device cpu --model-path ../../baselines-weibo/BotDGT/output/Weibo/month+42+0.78177+0.52716.pt

python evaluate.py --device cpu --model-path ../../baselines-weibo/BotDGT/output/Weibo/month+42+0.79354+0.28571.pt
```