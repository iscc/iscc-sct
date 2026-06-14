"""Tests for the iscc-sct doctor command (ONNX runtime diagnostics and opt-in repair)."""

import builtins

from iscc_sct import doctor


def _report(providers, gpu_dist=False, gpu_present=False, cpu_dist=False):
    """Build a diagnosis report from raw facts."""
    return doctor._diagnose(providers, gpu_dist, gpu_present, cpu_dist)


# --- detection helpers ---


def test_dist_installed_true():
    # onnxruntime is present in the test environment via the `test` dependency group
    assert doctor._dist_installed("onnxruntime") is True


def test_dist_installed_false():
    assert doctor._dist_installed("definitely-not-a-real-distribution-xyz") is False


def test_onnx_providers_returns_list():
    providers = doctor._onnx_providers()
    assert isinstance(providers, list)
    assert "CPUExecutionProvider" in providers


def test_onnx_providers_missing(monkeypatch):
    def _raise(name):
        raise ImportError(name)

    monkeypatch.setattr(doctor.importlib, "import_module", _raise)
    assert doctor._onnx_providers() is None


def test_gpu_present_returns_bool():
    assert isinstance(doctor._gpu_present(), bool)


def test_doctor_report_returns_dict():
    report = doctor.doctor_report()
    assert report["status"] in {"ok", "missing", "shadowed"}


# --- pure diagnosis ---


def test_diagnose_missing_cpu():
    report = _report(None)
    assert report["status"] == "missing"
    assert report["extra"] == "cpu"
    assert report["fix"] == [["install", "iscc-sct[cpu]"]]
    assert report["hint"] is False


def test_diagnose_missing_gpu():
    report = _report(None, gpu_present=True)
    assert report["status"] == "missing"
    assert report["extra"] == "gpu"
    assert report["fix"] == [["install", "iscc-sct[gpu]"]]


def test_diagnose_shadowed():
    # Genuine clobber: BOTH onnxruntime and onnxruntime-gpu installed, CUDA disabled.
    report = _report(["CPUExecutionProvider"], gpu_dist=True, gpu_present=True, cpu_dist=True)
    assert report["status"] == "shadowed"
    assert report["fix"][0][0] == "uninstall"
    assert report["fix"][1] == ["install", "--force-reinstall", "iscc-sct[gpu]"]


def test_diagnose_no_cuda_cpu_only_host():
    # onnxruntime-gpu correctly installed, no CUDA, nothing shadowing it (CPU-only host).
    # Not "shadowed": reinstalling cannot help an environment problem, so no fix is offered.
    report = _report(["CPUExecutionProvider"], gpu_dist=True, gpu_present=False)
    assert report["status"] == "no_cuda"
    assert report["fix"] == []


def test_diagnose_no_cuda_with_gpu_present():
    # NVIDIA GPU detected but CUDA unavailable (driver/library issue), gpu package not shadowed.
    report = _report(["CPUExecutionProvider"], gpu_dist=True, gpu_present=True)
    assert report["status"] == "no_cuda"
    assert report["fix"] == []


def test_diagnose_broken_gpu_import_is_missing():
    # onnxruntime-gpu installed but un-importable (providers=None) is "missing", not "shadowed".
    report = _report(None, gpu_dist=True)
    assert report["status"] == "missing"
    assert report["fix"] == [["install", "iscc-sct[cpu]"]]


def test_diagnose_ok_cuda():
    report = _report(["CUDAExecutionProvider", "CPUExecutionProvider"], gpu_dist=True)
    assert report["status"] == "ok"
    assert report["cuda"] is True
    assert report["hint"] is False


def test_diagnose_ok_cpu_no_gpu():
    report = _report(["CPUExecutionProvider"])
    assert report["status"] == "ok"
    assert report["hint"] is False


def test_diagnose_ok_hint_for_gpu_without_runtime():
    report = _report(["CPUExecutionProvider"], gpu_present=True)
    assert report["status"] == "ok"
    assert report["hint"] is True


# --- formatting ---


def test_format_report_ok():
    text = doctor.format_report(_report(["CPUExecutionProvider"]))
    assert "OK" in text
    assert "onnxruntime" in text


def test_format_report_hint():
    text = doctor.format_report(_report(["CPUExecutionProvider"], gpu_present=True))
    assert "CUDA acceleration" in text


def test_format_report_missing():
    text = doctor.format_report(_report(None))
    assert "no ONNX runtime" in text
    assert "not installed" in text
    assert "iscc-sct[cpu]" in text


def test_format_report_shadowed():
    text = doctor.format_report(
        _report(["CPUExecutionProvider"], gpu_dist=True, gpu_present=True, cpu_dist=True)
    )
    assert "shadowed" in text
    assert "onnxruntime-gpu" in text
    assert "conflicting" in text  # _runtime_label names both installed packages


def test_format_report_no_cuda():
    text = doctor.format_report(_report(["CPUExecutionProvider"], gpu_dist=True))
    assert "CUDA is unavailable" in text
    assert "environment issue" in text
    assert "onnxruntime-gpu" in text


# --- confirmation ---


def test_confirm_assume_yes():
    assert doctor._confirm(True) is True


def test_confirm_yes(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "Y")
    assert doctor._confirm(False) is True


def test_confirm_no(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")
    assert doctor._confirm(False) is False


def test_confirm_eof(monkeypatch):
    def _raise(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", _raise)
    assert doctor._confirm(False) is False


# --- run_doctor orchestration ---


class _Completed:
    def __init__(self, returncode):
        self.returncode = returncode


def _fake_runner(calls, returncode=0):
    def _run(cmd, *args, **kwargs):
        calls.append(cmd)
        return _Completed(returncode)

    return _run


def test_run_doctor_ok(monkeypatch):
    monkeypatch.setattr(doctor, "doctor_report", lambda: _report(["CPUExecutionProvider"]))
    calls = []
    monkeypatch.setattr(doctor.subprocess, "run", _fake_runner(calls))
    assert doctor.run_doctor() == 0
    assert calls == []  # no pip call for a healthy runtime


def test_run_doctor_missing_assume_yes(monkeypatch):
    monkeypatch.setattr(doctor, "doctor_report", lambda: _report(None))
    calls = []
    monkeypatch.setattr(doctor.subprocess, "run", _fake_runner(calls, returncode=0))
    assert doctor.run_doctor(assume_yes=True) == 0
    assert len(calls) == 1
    assert calls[0][-1] == "iscc-sct[cpu]"


def test_run_doctor_decline(monkeypatch):
    monkeypatch.setattr(doctor, "doctor_report", lambda: _report(None))
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    calls = []
    monkeypatch.setattr(doctor.subprocess, "run", _fake_runner(calls))
    assert doctor.run_doctor() == 1
    assert calls == []


def test_run_doctor_shadowed_runs_two_commands(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "doctor_report",
        lambda: _report(["CPUExecutionProvider"], gpu_dist=True, cpu_dist=True),
    )
    calls = []
    monkeypatch.setattr(doctor.subprocess, "run", _fake_runner(calls, returncode=0))
    assert doctor.run_doctor(assume_yes=True) == 0
    assert len(calls) == 2


def test_run_doctor_no_cuda(monkeypatch):
    # no_cuda has no actionable fix: print and exit 0 without prompting or pip calls.
    monkeypatch.setattr(
        doctor, "doctor_report", lambda: _report(["CPUExecutionProvider"], gpu_dist=True)
    )
    calls = []
    monkeypatch.setattr(doctor.subprocess, "run", _fake_runner(calls))
    assert doctor.run_doctor() == 0
    assert calls == []


def test_run_doctor_fix_failure(monkeypatch):
    monkeypatch.setattr(doctor, "doctor_report", lambda: _report(None))
    calls = []
    monkeypatch.setattr(doctor.subprocess, "run", _fake_runner(calls, returncode=1))
    assert doctor.run_doctor(assume_yes=True) == 1
