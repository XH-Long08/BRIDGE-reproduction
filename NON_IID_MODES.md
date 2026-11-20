# NUSWIDE Non-IID 模式切换

Non-IID 结果记录在 `results_summary.md`。运行入口为：

```bash
python train.py --dataset NUSWIDE --cp_type 1 --miss_rate 0.3 --non_iid True
```

当前打包的 `dataloader.py` 在 NUSWIDE 的 `flag=True` 分支调用 `Form_Incomplete_Data_NonIID(...)`，对应类别/数量不平衡模式。

论文复现记录中的随机噪声模式使用 `Form_Incomplete_Data(...)` 生成缺失视图，再调用已有函数 `add_pure_random_noniid_noise(self.X_i)`。若要运行此模式，在 `dataloader.py` 的 `NUSWIDE.__init__` 中将 `if flag:` 分支改为以下逻辑（保留 `else` 中的 IID 分支）：

```python
if flag:
    self.X, self.Y, self.X_i, self.Y_i, self.index = Form_Incomplete_Data(
        seed,
        missrate=miss_rate,
        X=[self.X1, self.X2, self.X3, self.X4, self.X5],
        Y=[self.Y, self.Y, self.Y, self.Y, self.Y]
    )
    self.X_i = add_pure_random_noniid_noise(self.X_i)
```

两种 Non-IID 模式应分开运行并分别记录。更改分支后请保存代码版本或提交 Git commit，避免把一种模式的结果误标为另一种。
