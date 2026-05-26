
```bash
cd baselines-weibo/BotRGCN
python train.py --weighted_loss
```

```bash
cd baselines-moltbook/moltbook-preprocess
python preprocess_moltbook.py
```

python evaluate.py --checkpoint ../../baselines-weibo/BotRGCN/checkpoints/best_model_20260503_165332.pt
python evaluate.py --checkpoint ../../baselines-weibo/BotRGCN/checkpoints/best_model_20260503_164505.pt
python evaluate.py --checkpoint ../../baselines-weibo/BotRGCN/checkpoints/best_model_20260503_164825.pt
