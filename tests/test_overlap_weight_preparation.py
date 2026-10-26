"""Full-weight updaters prepare once before the normal rollout update."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.update_weight import update_weight_from_disk as disk_module

NUM_GPUS = 0


@pytest.fixture
def distributed_module(monkeypatch):
    # Load only this updater with the unavailable Megatron conversion modules
    # replaced; the actual preparation and send methods still run unchanged.
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace()
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    conversion = ModuleType("slime.backends.megatron_utils.megatron_to_hf")
    conversion.convert_to_hf = lambda *args, **kwargs: []
    common = ModuleType("slime.backends.megatron_utils.update_weight.common")
    common.all_gather_param = lambda *args: None
    common.named_params_and_buffers = lambda *args: []
    monkeypatch.setitem(sys.modules, conversion.__name__, conversion)
    monkeypatch.setitem(sys.modules, common.__name__, common)
    name = "slime.backends.megatron_utils.update_weight._overlap_distributed_test"
    path = Path(__file__).resolve().parents[1] / "slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    return module


@pytest.mark.unit
def test_nccl_send_uses_prepared_buckets_without_regathering(distributed_module):
    updater = distributed_module.UpdateWeightFromDistributed(
        SimpleNamespace(), [], lambda: {}, model_name="test", quantization_config=None
    )
    iterated = []
    sent = []
    tensor = torch.tensor([1.0])

    def chunks(kind):
        iterated.append(kind)
        yield [(kind, tensor)]

    updater._iter_non_expert_chunks = lambda: chunks("dense")
    updater._iter_expert_chunks = lambda: chunks("expert")
    updater._update_bucket_weights_from_distributed = lambda chunk, pbar: sent.append(chunk[0][0])
    updater.prepare_weights()
    assert iterated == ["dense", "expert"]
    with pytest.raises(RuntimeError, match="already prepared"):
        updater.prepare_weights()

    updater._send_weights(None)
    assert sent == ["dense", "expert"]
    assert iterated == ["dense", "expert"]
    assert updater._prepared_weight_chunks is None


@pytest.mark.unit
@pytest.mark.parametrize("prepare_early", [False, True])
def test_disk_checkpoint_is_published_once_before_group_reload(monkeypatch, tmp_path, prepare_early):
    writes = []
    hooks = []
    monkeypatch.setattr(disk_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(disk_module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(disk_module, "get_gloo_group", lambda: None)

    def write_weights(args, directory, model, **kwargs):
        writes.append(directory)
        (directory / "weights.safetensors").write_bytes(b"prepared")

    monkeypatch.setattr(disk_module, "save_hf_model_to_path", write_weights)
    args = SimpleNamespace(update_weight_disk_dir=str(tmp_path), custom_update_weight_post_write_path=None)
    updater = disk_module.UpdateWeightFromDisk(args, [], lambda: {}, model_name="test", quantization_config=None)
    updater.weight_version = 7
    updater.connect_rollout_engines(["engine"], None)
    updater._post_write_hook = lambda _args, path, engines: hooks.append((path, engines))
    version_dir = tmp_path / "weight_v000008"
    version_dir.mkdir()
    (version_dir / "stale").write_text("old")

    if prepare_early:
        updater.prepare_weights()
        assert updater.weight_version == 7
        assert hooks == []
        assert not (version_dir / "stale").exists()
        assert (version_dir / "weights.safetensors").read_bytes() == b"prepared"
        with pytest.raises(RuntimeError, match="already prepared"):
            updater.prepare_weights()

    # Engine recovery may change the destination between preparation and load.
    updater.connect_rollout_engines(["replacement"], None)
    updater.update_weights()
    assert writes == [version_dir]
    assert hooks == [(str(version_dir), ["replacement"])]
    assert updater.weight_version == 8
    assert updater._prepared_version_dir is None
