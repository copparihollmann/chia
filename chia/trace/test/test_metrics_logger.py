"""Tests for chia.trace.metrics.MetricsLogger."""

import math
import os
import random
import tempfile

import pytest

from chia.trace.metrics import (
    MetricsBackend, MetricsLogger, NullBackend, TensorBoardBackend, _BACKENDS, register_backend,
)


class _RecordingBackend(MetricsBackend):
    """Minimal out-of-tree backend: keeps scalars in memory."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scalars = []

    def log_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture
def clean_registry():
    """Keep a test's registration out of the process-wide _BACKENDS table."""
    before = dict(_BACKENDS)
    yield
    _BACKENDS.clear()
    _BACKENDS.update(before)


def test_register_backend_makes_it_selectable(clean_registry):
    register_backend("recording", _RecordingBackend)

    m = MetricsLogger(backend="recording", run_dir="/tmp/x")
    m.log_scalar("loss", 0.5, step=3)

    assert isinstance(m._backend, _RecordingBackend)
    assert m._backend.kwargs == {"run_dir": "/tmp/x"}  # kwargs reach the backend ctor
    assert m._backend.scalars == [("loss", 0.5, 3)]
    m.close()


def test_register_backend_is_idempotent(clean_registry):
    register_backend("recording", _RecordingBackend)
    register_backend("recording", _RecordingBackend)  # re-import must not raise

    assert _BACKENDS["recording"] is _RecordingBackend


def test_register_backend_rejects_a_non_backend(clean_registry):
    with pytest.raises(TypeError, match="MetricsBackend subclass"):
        register_backend("bogus", dict)


def test_unknown_backend_still_raises():
    with pytest.raises(ValueError, match="Unknown metrics backend"):
        MetricsLogger(backend="never-registered")


def test_null_backend():
    m = MetricsLogger(backend="none")
    m.log_scalar("x", 1.0, step=0)
    m.flush()
    m.close()
    # double close is safe
    m.close()


def test_from_config_none():
    m = MetricsLogger.from_config(None)
    assert isinstance(m._backend, NullBackend)
    m.close()


def test_from_config_empty():
    m = MetricsLogger.from_config({})
    assert isinstance(m._backend, NullBackend)
    m.close()


def test_from_config_ignores_extra_kwargs():
    m = MetricsLogger.from_config({"backend": "none", "log_dir": "/tmp/unused"})
    assert isinstance(m._backend, NullBackend)
    m.close()


def test_unknown_backend_raises():
    try:
        MetricsLogger(backend="doesnotexist")
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "doesnotexist" in str(e)


def test_tensorboard_backend():
    # Imported here so the __main__ runner works in envs without pytest.
    import pytest
    pytest.importorskip("tensorboardX")
    with tempfile.TemporaryDirectory() as tmpdir:
        m = MetricsLogger.from_config({"backend": "tensorboard", "log_dir": tmpdir})
        assert isinstance(m._backend, TensorBoardBackend)
        for i in range(10):
            m.log_scalar("loss", 1.0 / (i + 1), step=i)
        m.flush()
        m.close()
        # tensorboardX writes event files into log_dir
        files = os.listdir(tmpdir)
        assert any("events.out.tfevents" in f for f in files), f"No event files in {files}"


def test_from_config_does_not_mutate_input():
    cfg = {"backend": "none", "extra": "value"}
    original = dict(cfg)
    MetricsLogger.from_config(cfg)
    assert cfg == original


TB_DEMO_DIR = "/tmp/chia_tb_demo"


def demo_tensorboard():
    """Log random curves to TensorBoard for visual inspection.

    Not collected by pytest (no ``test_`` prefix); run via
    ``python test/test_metrics_logger.py``.
    """
    m = MetricsLogger.from_config({"backend": "tensorboard", "log_dir": TB_DEMO_DIR})
    random.seed(42)
    noise = 0.0
    for i in range(100):
        noise += random.gauss(0, 0.05)
        m.log_scalar("train/loss", math.exp(-0.03 * i) + noise + 0.1 * random.random(), step=i)
        m.log_scalar("train/accuracy", min(1.0, 0.5 + 0.005 * i + 0.05 * random.random()), step=i)
        m.log_scalar("eval/loss", math.exp(-0.025 * i) + 0.15 * random.random(), step=i)
    m.close()
    print(f"\n  TensorBoard logs written to {TB_DEMO_DIR}")
    print(f"  View with: tensorboard --logdir {TB_DEMO_DIR}")
    print(f"  Then open: http://localhost:6006\n")


if __name__ == "__main__":
    test_null_backend()
    print("test_null_backend: PASS")
    test_from_config_none()
    print("test_from_config_none: PASS")
    test_from_config_empty()
    print("test_from_config_empty: PASS")
    test_from_config_ignores_extra_kwargs()
    print("test_from_config_ignores_extra_kwargs: PASS")
    test_unknown_backend_raises()
    print("test_unknown_backend_raises: PASS")
    test_from_config_does_not_mutate_input()
    print("test_from_config_does_not_mutate_input: PASS")
    try:
        import tensorboardX  # noqa: F401
    except ImportError:
        print("tensorboardX not installed; skipping tensorboard tests")
    else:
        test_tensorboard_backend()
        print("test_tensorboard_backend: PASS")
        demo_tensorboard()
        print("demo_tensorboard: PASS")
    print("\nAll tests passed!")
