import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks long tests (deselect with -m 'not slow')")


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    """Generate the synthetic sample dataset once per test session."""
    import make_fake_data
    out = tmp_path_factory.mktemp("fake")
    old = sys.argv
    sys.argv = ["make_fake_data.py", "--out", str(out), "--n", "8"]
    try:
        make_fake_data.main()
    finally:
        sys.argv = old
    return str(out)


@pytest.fixture(scope="session")
def tiny_cfg(fake_root):
    from tfgnet.config import Config
    cfg = Config()
    cfg.data.root = fake_root
    cfg.train.batch = 2
    return cfg
