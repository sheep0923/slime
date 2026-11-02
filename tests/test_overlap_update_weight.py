"""CPU checks for bucket pipelining without importing Megatron or initializing GPUs."""

import argparse
import ast
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
UPDATER = ROOT / "slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py"


@pytest.fixture
def implementation():
    tree = ast.parse(UPDATER.read_text())
    definitions = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name in {"_PendingWeightUpdate", "UpdateWeightFromDistributed", "update_weights_from_distributed"}
    ]
    namespace = dict(
        dataclass=dataclass,
        nullcontext=nullcontext,
        torch=SimpleNamespace(no_grad=lambda: lambda f: f),
        ray=SimpleNamespace(get=Mock(side_effect=lambda x: x)),
        dist=SimpleNamespace(barrier=Mock(), broadcast=Mock()),
        accelerator=SimpleNamespace(),
        get_gloo_group=lambda: "gloo",
        time=SimpleNamespace(sleep=Mock()),
        __name__=__name__,
    )
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(UPDATER), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("source", [False, True])
def test_pipeline_prepares_next_bucket_before_waiting(implementation, overlap, source):
    impl = implementation
    updater = impl.UpdateWeightFromDistributed.__new__(impl.UpdateWeightFromDistributed)
    updater.args = SimpleNamespace(overlap_update_weight=overlap)
    events = []

    def chunks(prefix):
        for i in range(2):
            name = f"{prefix}{i}"
            events.append(f"prepare:{name}")
            if source:
                yield [(name, object())]

    updater._iter_non_expert_chunks = lambda: chunks("tp")
    updater._iter_expert_chunks = lambda: chunks("ep")
    updater._on_chunk = Mock()
    updater._start_bucket_weights_from_distributed = lambda bucket: events.append(f"start:{bucket[0][0]}") or bucket
    updater._finish_bucket_weights_from_distributed = lambda bucket, **_: events.append(f"finish:{bucket[0][0]}")
    updater._update_bucket_weights_from_distributed = lambda bucket, **_: events.append(f"sync:{bucket[0][0]}")
    updater._send_weights(None)
    if not source:
        assert events == ["prepare:tp0", "prepare:tp1", "prepare:ep0", "prepare:ep1"]
    else:
        expected = []
        for prefix in ("tp", "ep"):
            if overlap:
                expected.extend(
                    [
                        f"prepare:{prefix}0",
                        f"start:{prefix}0",
                        f"prepare:{prefix}1",
                        f"finish:{prefix}0",
                        f"start:{prefix}1",
                        f"finish:{prefix}1",
                    ]
                )
            else:
                expected.extend([f"prepare:{prefix}0", f"sync:{prefix}0", f"prepare:{prefix}1", f"sync:{prefix}1"])
        assert events == expected
    assert impl.dist.barrier.call_count == 2


def test_pending_bucket_is_finished_when_preparation_fails(implementation):
    impl = implementation
    updater = impl.UpdateWeightFromDistributed.__new__(impl.UpdateWeightFromDistributed)
    updater.args = SimpleNamespace(overlap_update_weight=True)
    bucket = [("weight", object())]

    def chunks():
        yield bucket
        raise RuntimeError("gather failed")

    updater._iter_non_expert_chunks = chunks
    updater._iter_expert_chunks = lambda: iter(())
    updater._on_chunk = Mock()
    updater._start_bucket_weights_from_distributed = Mock(return_value=bucket)
    updater._finish_bucket_weights_from_distributed = Mock()
    with pytest.raises(RuntimeError, match="gather failed"):
        updater._send_weights(None)
    updater._finish_bucket_weights_from_distributed.assert_called_once_with(bucket, pbar=None)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_broadcast_waits_for_producer_and_preserves_default_behavior(implementation, asynchronous):
    impl = implementation
    handle = Mock()
    impl.dist.broadcast.return_value = handle
    stream = Mock()
    impl.accelerator.current_stream = Mock(return_value="producer")
    impl.accelerator.stream = Mock(return_value=nullcontext())
    engine = SimpleNamespace(update_weights_from_distributed=SimpleNamespace(remote=Mock(return_value="ref")))
    tensor = SimpleNamespace(dtype="float32", shape=(2,), data="tensor")
    result = impl.update_weights_from_distributed(
        "group",
        "pg",
        3,
        [engine],
        [("weight", tensor)],
        wait_for_broadcast=not asynchronous,
        broadcast_stream=stream if asynchronous else None,
    )
    impl.dist.broadcast.assert_called_once_with("tensor", 0, group="pg", async_op=True)
    if asynchronous:
        assert result == (["ref"], [handle])
        handle.wait.assert_not_called()
        stream.wait_stream.assert_called_once_with("producer")
        impl.accelerator.stream.assert_called_once_with(stream)
    else:
        assert result == ["ref"]
        handle.wait.assert_called_once_with()
        impl.accelerator.stream.assert_not_called()
    assert engine.update_weights_from_distributed.remote.call_args.kwargs["weight_version"] == "3"


@pytest.mark.parametrize("engine_failure", [False, True])
def test_pending_bucket_keeps_tensors_and_lock_until_completion(implementation, engine_failure):
    impl = implementation
    updater = impl.UpdateWeightFromDistributed.__new__(impl.UpdateWeightFromDistributed)
    updater.rollout_engine_lock = SimpleNamespace(release=SimpleNamespace(remote=Mock(return_value="released")))
    bucket = [("weight", object())]
    handle = Mock()
    handle.wait.side_effect = lambda: bucket and True
    pending = impl._PendingWeightUpdate(bucket, ["ref"], [handle])

    def get(value):
        if value == ["ref"]:
            assert bucket
            updater.rollout_engine_lock.release.remote.assert_not_called()
            if engine_failure:
                raise RuntimeError("engine failed")

    impl.ray.get.side_effect = get
    pbar = Mock()
    if engine_failure:
        with pytest.raises(RuntimeError, match="engine failed"):
            updater._finish_bucket_weights_from_distributed(pending, pbar)
        pbar.update.assert_not_called()
    else:
        updater._finish_bucket_weights_from_distributed(pending, pbar)
        pbar.update.assert_called_once_with(1)
    assert bucket == []
    handle.wait.assert_called_once_with()
    updater.rollout_engine_lock.release.remote.assert_called_once_with()


@pytest.mark.parametrize("invalid", [None, "mode", "transport", "colocate", "offload"])
def test_flag_registration_and_configuration_validation(invalid):
    path = ROOT / "slime/utils/arguments.py"
    tree = ast.parse(path.read_text())
    registration = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and any(isinstance(a, ast.Constant) and a.value == "--overlap-update-weight" for a in n.value.args)
    )
    validation = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.BoolOp)
        and any(
            isinstance(child, ast.Constant) and child.value == "overlap_update_weight" for child in ast.walk(n.test)
        )
    )
    parser = argparse.ArgumentParser()
    exec(compile(ast.Module(body=[registration], type_ignores=[]), str(path), "exec"), {"parser": parser})
    assert not parser.parse_args([]).overlap_update_weight
    args = parser.parse_args(["--overlap-update-weight"])
    args.update_weight_mode = "delta" if invalid == "mode" else "full"
    args.update_weight_transport = "disk" if invalid == "transport" else "nccl"
    args.colocate = invalid == "colocate"
    args.offload_train = invalid == "offload"
    code = compile(ast.Module(body=[validation], type_ignores=[]), str(path), "exec")
    if invalid:
        with pytest.raises(ValueError, match="--overlap-update-weight requires"):
            exec(code, {"args": args})
        args.overlap_update_weight = False
    exec(code, {"args": args})


@pytest.mark.parametrize("source", [False, True])
def test_start_retains_bucket_and_reuses_send_stream(implementation, source):
    impl = implementation
    updater = impl.UpdateWeightFromDistributed.__new__(impl.UpdateWeightFromDistributed)
    updater._is_pp_src_rank = source
    updater._weight_send_stream = None
    updater._group_name = "group"
    updater._model_update_groups = "pg"
    updater.weight_version = 2
    updater.rollout_engines = []
    updater.rollout_engine_lock = SimpleNamespace(acquire=SimpleNamespace(remote=Mock(return_value=True)))
    stream = object()
    impl.accelerator.new_stream = Mock(return_value=stream)
    broadcast = Mock(return_value=(["ref"], ["handle"]))
    # Replace the global called by the actual start method, without importing Megatron.
    updater._start_bucket_weights_from_distributed.__func__.__globals__["update_weights_from_distributed"] = broadcast
    for _ in range(2):
        bucket = [("weight", object())]
        pending = updater._start_bucket_weights_from_distributed(bucket)
        if source:
            assert pending.tensors is bucket and pending.refs == ["ref"] and pending.handles == ["handle"]
            assert broadcast.call_args.kwargs == {"wait_for_broadcast": False, "broadcast_stream": stream}
        else:
            assert pending is None
    assert impl.accelerator.new_stream.call_count == int(source)
    assert updater.rollout_engine_lock.acquire.remote.call_count == 2 * int(source)


def test_transfer_failure_is_not_finished_twice(implementation):
    impl = implementation
    updater = impl.UpdateWeightFromDistributed.__new__(impl.UpdateWeightFromDistributed)
    updater.args = SimpleNamespace(overlap_update_weight=True)
    updater._iter_non_expert_chunks = lambda: iter([[("first", object())], [("second", object())]])
    updater._iter_expert_chunks = lambda: iter(())
    updater._on_chunk = Mock()
    updater._start_bucket_weights_from_distributed = Mock(return_value="pending")
    updater._finish_bucket_weights_from_distributed = Mock(side_effect=RuntimeError("transfer failed"))
    with pytest.raises(RuntimeError, match="transfer failed"):
        updater._send_weights(None)
    updater._finish_bucket_weights_from_distributed.assert_called_once_with("pending", pbar=None)
