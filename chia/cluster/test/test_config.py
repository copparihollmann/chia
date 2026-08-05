"""Cluster-config parsing tests.

Offline and dict-only: :func:`build_config` takes an already-loaded mapping, so nothing here
touches YAML, SSH or a container runtime.

Covers:
  1. test_engine_defaults_to_docker
        A docker: block with no engine key keeps the historical default.
  2. test_engine_can_be_selected
        engine: <name> reaches DockerConfig.engine (the field existed but no YAML key set it).
  3. test_engine_is_per_node_type
        A node type's engine is independent of the cluster-wide block.
  4. test_unsupported_engine_is_rejected_at_parse_time
        A runtime CHIA cannot drive fails during parsing, naming the offending block.
  5. test_no_docker_block_means_no_container
        Omitting docker: is the supported "run directly over SSH" path.
"""
import pytest

from chia.cluster.config import (
    SUPPORTED_CONTAINER_ENGINES, ConfigError, build_config,
)


def _raw(docker_block=None, node_docker_block=None):
    """Smallest config build_config accepts, with optional container blocks."""
    raw = {
        "cluster_name": "test",
        "provider": {"head_ip": "10.0.0.1"},
        "auth": {"ssh_user": "ray"},
        "available_node_types": {
            "worker_type": {
                "resources": {"cpu_work": 1},
                "num_workers": 1,
                "compatible_ips": ["10.0.0.2"],
            },
        },
    }
    if docker_block is not None:
        raw["docker"] = docker_block
    if node_docker_block is not None:
        raw["available_node_types"]["worker_type"]["docker"] = node_docker_block
    return raw


def test_engine_defaults_to_docker():
    cfg = build_config(_raw(docker_block={"image": "img:latest"}))

    assert cfg.global_docker is not None
    assert cfg.global_docker.engine == "docker"


@pytest.mark.parametrize("engine", SUPPORTED_CONTAINER_ENGINES)
def test_engine_can_be_selected(engine):
    """Every supported engine is reachable from YAML.

    DockerConfig.engine has always existed and DockerManager already interpolates it, but the
    parser hardcoded "docker", so the field could not be set from a cluster file at all.
    """
    cfg = build_config(_raw(docker_block={"image": "img:latest", "engine": engine}))

    assert cfg.global_docker.engine == engine


def test_engine_is_per_node_type():
    """A node type may use a different engine from the cluster-wide default."""
    cfg = build_config(_raw(
        docker_block={"image": "global:latest"},
        node_docker_block={"image": "nt:latest", "engine": "podman"},
    ))

    assert cfg.global_docker.engine == "docker"
    assert cfg.node_types["worker_type"].docker.engine == "podman"


@pytest.mark.parametrize("where_key,kwargs", [
    ("top-level config", {"docker_block": {"image": "i", "engine": "apptainer"}}),
    ("available_node_types.worker_type",
     {"node_docker_block": {"image": "i", "engine": "apptainer"}}),
])
def test_unsupported_engine_is_rejected_at_parse_time(where_key, kwargs):
    """Fail at parse time, not deep inside bring-up.

    Apptainer/bubblewrap have a different execution model (no long-lived container to exec into),
    so accepting the name would surface as an unrelated failure much later. The message must name
    the block that set it, since a cluster file can carry several.
    """
    with pytest.raises(ConfigError) as exc:
        build_config(_raw(**kwargs))

    assert "apptainer" in str(exc.value)
    assert where_key in str(exc.value)


def test_no_docker_block_means_no_container():
    """Omitting docker: is the documented bare-SSH path, not an error."""
    cfg = build_config(_raw())

    assert cfg.global_docker is None
    assert cfg.node_types["worker_type"].docker is None
