# 权重准备与 Rollout 重叠

`--overlap-weight-preparation` 将完整权重的准备阶段放在 actor 训练结束后执行。启用此参数后，内置同步 rollout 入口会切换到 slime 的 fully async 后台 producer，使其在准备权重时继续用旧版权重生成；训练驱动仍在原有的权重更新边界暂停 rollout admission，再通知引擎加载新权重。评估继续使用同步 rollout 入口。

```bash
python train.py \
  --update-weight-mode full \
  --update-weight-transport nccl \
  --overlap-weight-preparation
```

NCCL 路径提前执行 TP/EP gather、HF 转换和 bucketing，并在训练进程中保留待广播的 tensor。权重更新时复用这些 bucket，随后按照原有流程暂停引擎、按 `--flush-cache-interval` 决定是否清理缓存、广播权重并恢复生成。显存需要容纳准备好的完整传输 payload。

共享盘路径也支持此参数：

```bash
python train.py \
  --update-weight-mode full \
  --update-weight-transport disk \
  --update-weight-disk-dir /shared/weights \
  --overlap-weight-preparation
```

disk 路径在训练后写好下一版本的 HF checkpoint。`--custom-update-weight-post-write-path` 指定的发布钩子仍在更新边界执行，以便拿到故障恢复后的引擎列表。`RayTrainGroup` 也在该边界负责 rollout 引擎的 pull、暂停、加载和恢复；`--update-weight-disk-keep-files` 仍控制加载后的文件清理。共享目录必须能被训练和 rollout 节点访问。

该参数只支持 full 模式及实时 rollout，并要求独立训练和 rollout GPU；不能同时开启 `--colocate`、训练 offload 或 rollout offload。自定义 rollout 入口需要自行提供等效的后台调度，目前与此参数不兼容。可对比日志里的 `prepare_weight_update`、`update_weights` 和 rollout 耗时评估收益。
