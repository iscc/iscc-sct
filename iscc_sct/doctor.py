"""ONNX runtime diagnostics and opt-in repair for the `iscc-sct doctor` command.

Inspects the installed ONNX runtime and classifies the environment: no runtime, a CPU
`onnxruntime` package shadowing `onnxruntime-gpu` (both installed, CUDA disabled),
`onnxruntime-gpu` installed without usable CUDA (driver/hardware issue, not fixable by
reinstalling), or healthy. Recommends the right install extra and - only with explicit
confirmation - runs the matching pip command in a subprocess.
"""

import importlib
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution


__all__ = ["doctor_report", "format_report", "run_doctor"]


def _dist_installed(name):
    # type: (str) -> bool
    """
    Check whether a distribution is installed.

    :param name: Distribution name to look up.
    :return: True if the distribution is installed.
    """
    try:
        distribution(name)
        return True
    except PackageNotFoundError:
        return False


def _onnx_providers():
    # type: () -> list[str] | None
    """
    Return the available onnxruntime execution providers.

    :return: Provider names reported by onnxruntime, or None if no runtime is importable.
    """
    try:
        rt = importlib.import_module("onnxruntime")
    except ImportError:
        return None
    return rt.get_available_providers()


def _gpu_present():
    # type: () -> bool
    """
    Detect an NVIDIA GPU by probing for the nvidia-smi tool on PATH.

    :return: True if nvidia-smi is found on PATH.
    """
    return shutil.which("nvidia-smi") is not None


def _diagnose(providers, gpu_dist, gpu_present, cpu_dist):
    # type: (list[str]|None, bool, bool, bool) -> dict
    """
    Derive runtime status, recommended extra, and pip fix from environment facts.

    :param providers: onnxruntime.get_available_providers() result, or None if not importable.
    :param gpu_dist: Whether the onnxruntime-gpu distribution is installed.
    :param gpu_present: Whether an NVIDIA GPU was detected.
    :param cpu_dist: Whether the CPU onnxruntime distribution is installed.
    :return: Report dict with status, runtime, cuda, gpu_present, gpu_dist, cpu_dist, extra,
        fix, hint.
    """
    runtime = providers is not None
    cuda = runtime and "CUDAExecutionProvider" in providers
    extra = "gpu" if gpu_present else "cpu"

    if runtime and gpu_dist and cpu_dist and not cuda:
        # Both runtime distributions are installed: their wheels unpacked into the same
        # directory and the CPU build clobbered onnxruntime-gpu, disabling CUDA (issue #23).
        status = "shadowed"
        fix = [
            ["uninstall", "-y", "onnxruntime", "onnxruntime-gpu"],
            ["install", "--force-reinstall", "iscc-sct[gpu]"],
        ]
    elif not runtime:
        status = "missing"
        fix = [["install", f"iscc-sct[{extra}]"]]
    elif gpu_dist and not cuda:
        # onnxruntime-gpu is installed and importable but CUDA is unavailable, and nothing
        # shadowed it - the NVIDIA driver/GPU is missing or the CUDA libraries failed to load.
        # Reinstalling cannot fix an environment problem, so offer no pip fix.
        status = "no_cuda"
        fix = []
    else:
        status = "ok"
        fix = []

    hint = status == "ok" and gpu_present and not cuda and not gpu_dist
    return {
        "status": status,
        "runtime": runtime,
        "cuda": cuda,
        "gpu_present": gpu_present,
        "gpu_dist": gpu_dist,
        "cpu_dist": cpu_dist,
        "extra": extra,
        "fix": fix,
        "hint": hint,
    }


def doctor_report():
    # type: () -> dict
    """
    Diagnose the ONNX runtime in the current environment.

    :return: A report dict (see `_diagnose`).
    """
    return _diagnose(
        _onnx_providers(),
        _dist_installed("onnxruntime-gpu"),
        _gpu_present(),
        _dist_installed("onnxruntime"),
    )


def _yn(flag):
    # type: (bool) -> str
    """Render a boolean as yes/no."""
    return "yes" if flag else "no"


def _runtime_label(report):
    # type: (dict) -> str
    """Describe which ONNX runtime package(s) are installed."""
    if report["gpu_dist"] and report["cpu_dist"]:
        return "onnxruntime + onnxruntime-gpu (conflicting)"
    if report["gpu_dist"]:
        return "onnxruntime-gpu"
    if report["runtime"]:
        return "onnxruntime"
    return "not installed"


def format_report(report):
    # type: (dict) -> str
    """
    Render a human-readable diagnosis.

    :param report: Report dict from `doctor_report`.
    :return: Multi-line report string.
    """
    lines = [
        "iscc-sct ONNX runtime check",
        "",
        f"  ONNX runtime:   {_runtime_label(report)}",
        f"  CUDA provider:  {_yn(report['cuda'])}",
        f"  NVIDIA GPU:     {_yn(report['gpu_present'])}",
        "",
    ]
    if report["status"] == "ok":
        lines.append("Status: OK - a working ONNX runtime is installed.")
        if report["hint"]:
            lines.append('GPU detected. For CUDA acceleration: pip install "iscc-sct[gpu]"')
    elif report["status"] == "missing":
        lines.append("Status: no ONNX runtime installed.")
        lines.append(f'Recommended: pip install "iscc-sct[{report["extra"]}]"')
    elif report["status"] == "no_cuda":
        lines.append(
            "Status: onnxruntime-gpu is installed and runs on CPU, but CUDA is unavailable - no "
            "NVIDIA driver/GPU was detected or the CUDA libraries failed to load. This is an "
            "environment issue, not a packaging problem; reinstalling will not help."
        )
    else:
        lines.append(
            "Status: onnxruntime-gpu is installed but CUDA is unavailable - the CPU onnxruntime "
            "package shadowed the GPU build. The fix reinstalls only the GPU build."
        )
    return "\n".join(lines)


def _confirm(assume_yes):
    # type: (bool) -> bool
    """
    Confirm the fix with the user unless auto-confirmed.

    :param assume_yes: Skip the prompt and proceed when True.
    :return: True to proceed with the fix.
    """
    if assume_yes:
        return True
    try:
        answer = input("Install it now? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def run_doctor(assume_yes=False):
    # type: (bool) -> int
    """
    Print the ONNX runtime diagnosis and, on confirmation, run the recommended pip fix.

    :param assume_yes: Auto-confirm the fix without prompting.
    :return: Exit code: 0 when healthy or the fix ran cleanly, 1 otherwise.
    """
    report = doctor_report()
    print(format_report(report))
    if not report["fix"]:
        return 0
    if not _confirm(assume_yes):
        print("No changes made.")
        return 1
    failed = False
    for args in report["fix"]:
        print(f"  running: pip {' '.join(args)}")
        completed = subprocess.run([sys.executable, "-m", "pip", *args])
        failed = failed or completed.returncode != 0
    return 1 if failed else 0
