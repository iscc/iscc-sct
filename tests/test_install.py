"""Tests for the install command."""

import pytest
from unittest.mock import patch, Mock
from typer.testing import CliRunner
from iscc_sct.cli import app
from iscc_sct.models_config import MODEL_REGISTRY


runner = CliRunner()


def test_install_all_models_already_installed(tmp_path, monkeypatch):
    """Test installing all models when they're already installed."""
    monkeypatch.setenv("ISCC_SCT_MODEL_DIR", str(tmp_path))

    # Create fake model files
    for version in MODEL_REGISTRY:
        config = MODEL_REGISTRY[version]
        version_dir = tmp_path / f"v{version}"
        version_dir.mkdir()
        for filename in config.filenames:
            (version_dir / filename).write_bytes(b"fake model data")

    # Mock check_integrity to succeed
    with patch("iscc_sct.cli.check_integrity"):
        result = runner.invoke(app, ["install"])

    assert result.exit_code == 0
    assert (
        "Already installed and verified" in result.stdout or "Integrity verified" in result.stdout
    )


def test_install_single_version(tmp_path, monkeypatch):
    """Test installing a single model version."""
    monkeypatch.setenv("ISCC_SCT_MODEL_DIR", str(tmp_path))

    # Create fake model files for v0 only
    config = MODEL_REGISTRY[0]
    version_dir = tmp_path / "v0"
    version_dir.mkdir()
    for filename in config.filenames:
        (version_dir / filename).write_bytes(b"fake model data")

    # Mock check_integrity to succeed
    with patch("iscc_sct.cli.check_integrity"):
        result = runner.invoke(app, ["install", "-m", "0"])

    assert result.exit_code == 0
    # Should only process v0
    assert "v0" in result.stdout


def test_install_force_redownload(tmp_path, monkeypatch):
    """Test force re-download even when files exist."""
    monkeypatch.setenv("ISCC_SCT_MODEL_DIR", str(tmp_path))

    # Create fake model files
    config = MODEL_REGISTRY[0]
    version_dir = tmp_path / "v0"
    version_dir.mkdir()
    for filename in config.filenames:
        (version_dir / filename).write_bytes(b"old model data")

    # Mock download_model_files to track if it was called
    with patch("iscc_sct.cli.download_model_files") as mock_download:
        result = runner.invoke(app, ["install", "-m", "0", "--force"])

    assert result.exit_code == 0
    assert mock_download.called


def test_install_verify_only_files_exist(tmp_path, monkeypatch):
    """Test verify-only mode when files exist."""
    monkeypatch.setenv("ISCC_SCT_MODEL_DIR", str(tmp_path))

    # Create fake model files
    config = MODEL_REGISTRY[0]
    version_dir = tmp_path / "v0"
    version_dir.mkdir()
    for filename in config.filenames:
        (version_dir / filename).write_bytes(b"fake model data")

    # Mock check_integrity to succeed
    with patch("iscc_sct.cli.check_integrity"):
        result = runner.invoke(app, ["install", "-m", "0", "--verify-only"])

    assert result.exit_code == 0
    assert "Integrity verified" in result.stdout


def test_install_verify_only_files_missing(tmp_path, monkeypatch):
    """Test verify-only mode when files are missing."""
    monkeypatch.setenv("ISCC_SCT_MODEL_DIR", str(tmp_path))

    # Mock get_model_path to return a directory without any files
    def mock_get_model_path(version):
        model_dir = tmp_path / f"v{version}"
        model_dir.mkdir(exist_ok=True)
        return model_dir

    with patch("iscc_sct.cli.get_model_path", side_effect=mock_get_model_path):
        result = runner.invoke(app, ["install", "-m", "0", "--verify-only"])

    assert result.exit_code == 0
    assert "Files missing" in result.stdout or "Missing" in result.stdout


def test_install_verify_only_integrity_failed(tmp_path, monkeypatch):
    """Test verify-only mode when integrity check fails."""
    monkeypatch.setenv("ISCC_SCT_MODEL_DIR", str(tmp_path))

    # Create fake model files
    config = MODEL_REGISTRY[0]
    version_dir = tmp_path / "v0"
    version_dir.mkdir()
    for filename in config.filenames:
        (version_dir / filename).write_bytes(b"corrupted data")

    # Mock check_integrity to fail
    with patch("iscc_sct.cli.check_integrity", side_effect=RuntimeError("Integrity check failed")):
        result = runner.invoke(app, ["install", "-m", "0", "--verify-only"])

    assert result.exit_code == 0
    assert "Integrity check failed" in result.stdout or "Failed" in result.stdout


def test_install_corrupted_files_redownload(tmp_path, monkeypatch):
    """Test re-download when existing files fail integrity check."""
    monkeypatch.setenv("ISCC_SCT_MODEL_DIR", str(tmp_path))

    # Create fake model files
    config = MODEL_REGISTRY[0]
    version_dir = tmp_path / "v0"
    version_dir.mkdir()
    for filename in config.filenames:
        (version_dir / filename).write_bytes(b"corrupted data")

    # Mock check_integrity to fail once, then succeed
    check_integrity_mock = Mock(side_effect=[RuntimeError("Bad checksum"), None])

    with patch("iscc_sct.cli.check_integrity", check_integrity_mock):
        with patch("iscc_sct.cli.download_model_files") as mock_download:
            result = runner.invoke(app, ["install", "-m", "0"])

    assert result.exit_code == 0
    assert mock_download.called


def test_install_quiet_mode(tmp_path, monkeypatch):
    """Test quiet mode suppresses output."""
    monkeypatch.setenv("ISCC_SCT_MODEL_DIR", str(tmp_path))

    # Create fake model files
    config = MODEL_REGISTRY[0]
    version_dir = tmp_path / "v0"
    version_dir.mkdir()
    for filename in config.filenames:
        (version_dir / filename).write_bytes(b"fake model data")

    with patch("iscc_sct.cli.check_integrity"):
        result = runner.invoke(app, ["install", "-m", "0", "--quiet"])

    assert result.exit_code == 0
    # Quiet mode should have minimal output
    assert "Installing ISCC-SCT Models" not in result.stdout


def test_install_invalid_version():
    """Test installing an invalid model version."""
    result = runner.invoke(app, ["install", "-m", "99"])

    assert result.exit_code == 1
    assert "not found" in result.stdout or "Error" in result.stdout


def test_install_multiple_versions(tmp_path, monkeypatch):
    """Test installing multiple specific versions."""
    monkeypatch.setenv("ISCC_SCT_MODEL_DIR", str(tmp_path))

    # Create fake model files for both versions
    for version in [0, 1]:
        config = MODEL_REGISTRY[version]
        version_dir = tmp_path / f"v{version}"
        version_dir.mkdir()
        for filename in config.filenames:
            (version_dir / filename).write_bytes(b"fake model data")

    with patch("iscc_sct.cli.check_integrity"):
        result = runner.invoke(app, ["install", "-m", "0", "-m", "1"])

    assert result.exit_code == 0
    assert "v0" in result.stdout
    assert "v1" in result.stdout


def test_download_model_files_with_progress(tmp_path, monkeypatch):
    """Test download_model_files function with progress tracking."""
    from iscc_sct.cli import download_model_files
    from rich.console import Console

    config = MODEL_REGISTRY[0]
    model_dir = tmp_path / "v0"
    model_dir.mkdir()
    console = Console()

    # Mock download_file to simulate download
    with patch("iscc_sct.cli.download_file") as mock_download:
        download_model_files(config, model_dir, 600, False, console)

    # Should have been called once per file
    assert mock_download.call_count == len(config.filenames)


def test_download_model_files_quiet(tmp_path, monkeypatch):
    """Test download_model_files in quiet mode."""
    from iscc_sct.cli import download_model_files
    from rich.console import Console

    config = MODEL_REGISTRY[0]
    model_dir = tmp_path / "v0"
    model_dir.mkdir()
    console = Console()

    # Mock download_file
    with patch("iscc_sct.cli.download_file") as mock_download:
        download_model_files(config, model_dir, 600, True, console)

    # Should have been called without progress arguments
    assert mock_download.call_count == len(config.filenames)
    # Verify it was called without progress/task_id
    for call in mock_download.call_args_list:
        assert len(call[0]) == 4  # url, dest_path, checksum, timeout (no progress args)


def test_download_model_files_with_exception(tmp_path):
    """Test download_model_files handles exceptions correctly."""
    from iscc_sct.cli import download_model_files
    from rich.console import Console

    config = MODEL_REGISTRY[0]
    model_dir = tmp_path / "v0"
    model_dir.mkdir()
    console = Console()

    # Mock download_file to raise an exception
    with patch("iscc_sct.cli.download_file", side_effect=Exception("Network error")):
        with pytest.raises(Exception, match="Network error"):
            download_model_files(config, model_dir, 600, False, console)
