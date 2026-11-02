# 权重桶发送与准备重叠

在使用独立训练和 rollout GPU、全量 NCCL 权重同步时，添加：

```bash
--update-weight-mode full \
--update-weight-transport nccl \
--overlap-update-weight
```

默认不开启。开启后，当前权重桶在独立发送 stream 上异步 broadcast，
同时准备下一桶的 TP/EP gather 和 HF 转换；发送下一桶前等待上一桶通信和
rollout engine 加载完成。发送中的张量和 engine 锁保留至该桶完成。
非专家、专家两阶段之间仍保持同步屏障，更新期间仍暂停生成。

该功能只重叠一次权重更新内部的桶发送和准备，不提前准备整个模型权重。
`--update-weight-buffer-size` 继续控制分桶大小；重叠时会同时保留发送中的桶和下一桶，
因此需要额外显存。实际性能收益取决于设备和通信后端，需通过真实训练测量。

不支持 delta、disk、`--colocate` 或 `--offload-train`，不满足条件时参数校验会报错。
