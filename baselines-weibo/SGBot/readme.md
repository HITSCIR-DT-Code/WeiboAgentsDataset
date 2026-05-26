# SGBot on Weibo

## 运行方式

先激活包含 `torch`、`scikit-learn` 等依赖的 Python 环境，再在当前目录下执行：

```bash
cd baselines/SGBot
python preprocess.py
python train.py 42
