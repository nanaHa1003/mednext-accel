from mednext_accel.profiling.environment import collect_environment


class _Properties:
    name = "NVIDIA L40S"
    total_memory = 48 * 1024**3


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_device_properties(index: int) -> _Properties:
        assert index == 0
        return _Properties()

    @staticmethod
    def get_device_capability(index: int) -> tuple[int, int]:
        assert index == 0
        return (8, 9)


class _Cudnn:
    @staticmethod
    def version() -> int:
        return 90100


class _Backends:
    cudnn = _Cudnn()


class _Version:
    cuda = "12.8"


class _Torch:
    __version__ = "2.9.0"
    cuda = _Cuda()
    backends = _Backends()
    version = _Version()


def test_environment_collects_relevant_reproducibility_fields_without_host_identity() -> None:
    info = collect_environment(
        torch_module=_Torch(),
        package_version="0.1.0a0",
        triton_version="3.5.0",
        driver_version="580.65.06",
        timestamp="2026-09-21T12:00:00Z",
        python_version="3.11.13",
        platform_name="Linux-6.8-x86_64",
    )

    assert info["gpu"] == {
        "index": 0,
        "name": "NVIDIA L40S",
        "sm": [8, 9],
        "total_memory_bytes": 48 * 1024**3,
    }
    assert info["software"]["torch"] == "2.9.0"
    assert info["software"]["cuda"] == "12.8"
    assert info["software"]["cudnn"] == 90100
    assert info["software"]["triton"] == "3.5.0"
    assert info["driver"] == "580.65.06"
    assert "hostname" not in info
    assert "username" not in info
