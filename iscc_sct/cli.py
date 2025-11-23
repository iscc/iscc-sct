"""Command-line interface for ISCC-SCT."""

import json
import sys
from pathlib import Path
from typing import Optional, List

import typer
from charset_normalizer import from_bytes
from loguru import logger
from rich.console import Console
from rich.json import JSON
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from iscc_sct.main import create
from iscc_sct.models_config import MODEL_REGISTRY, get_model_config
from iscc_sct.utils import get_model_path, download_file, check_integrity
from iscc_sct.options import sct_opts

# Get version from package metadata
try:
    from importlib.metadata import version

    __version__ = version("iscc-sct")
except Exception:  # pragma: no cover
    __version__ = "unknown"

app = typer.Typer(
    name="iscc-sct",
    help="ISCC - Semantic Code Text: Generate semantic similarity preserving text codes.",
    add_completion=False,
    invoke_without_command=True,  # Allow default command
    no_args_is_help=False,  # Allow running without args for version
    context_settings={"help_option_names": ["-h", "--help"]},
)


def version_callback(value: bool):
    # type: (bool) -> None
    """Show version and exit."""
    if value:
        typer.echo(f"iscc-sct version {__version__}")
        raise typer.Exit()


def expand_path(path_pattern):
    # type: (str) -> str
    """Expand user directory in path pattern."""
    return Path(path_pattern).expanduser().as_posix()


def find_files_from_directory(directory):
    # type: (Path) -> list[Path]
    """Recursively find all files in a directory."""
    files = list(directory.rglob("*"))
    return sorted([f for f in files if f.is_file()])


def find_files_from_glob(pattern):
    # type: (str) -> list[Path]
    """Find files matching a glob pattern."""
    import glob

    matches = glob.glob(pattern, recursive=True)
    return sorted([Path(m) for m in matches if Path(m).is_file()])


def resolve_path(path_pattern):
    # type: (str) -> list[Path]
    """Resolve a path pattern to a list of file paths.

    Handles single files, directories (recursively), and glob patterns.

    Args:
        path_pattern: A file path, directory path, or glob pattern

    Returns:
        Sorted list of resolved file paths
    """
    expanded_pattern = expand_path(path_pattern)
    path = Path(expanded_pattern)

    # Case 1: Existing file
    if path.exists() and path.is_file():
        return [path]

    # Case 2: Existing directory - recursively find all files
    if path.exists() and path.is_dir():
        return find_files_from_directory(path)

    # Case 3: Glob pattern
    glob_chars = {"*", "?", "[", "]", "{", "}"}
    if any(char in expanded_pattern for char in glob_chars):
        return find_files_from_glob(expanded_pattern)

    # Case 4: Non-existent file or pattern with no matches
    return []


def collect_files(paths):
    # type: (list[str]) -> list[Path]
    """Collect and deduplicate files from multiple path patterns."""
    files = []
    for path in paths:
        matching_files = resolve_path(path)
        if not matching_files:
            logger.warning(f"No files found for pattern: {path}")
        files.extend(matching_files)

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


def validate_options(format, content, granular):
    # type: (str, bool, bool) -> None
    """Validate command-line options."""
    if format not in ["text", "json"]:
        typer.echo(f"Error: Invalid format '{format}'. Choose 'text' or 'json'.", err=True)
        raise typer.Exit(code=1)

    if content and not granular:
        typer.echo("Error: --content requires --granular to be enabled.", err=True)
        raise typer.Exit(code=1)


def read_text_from_file(file_path):
    # type: (Path) -> str | None
    """Read and decode text from a file.

    Returns:
        Decoded text or None if file cannot be processed
    """
    with file_path.open("rb") as file:
        data = file.read()

        # Try UTF-8 first
        try:
            text = data.decode("utf-8")
            if not text.strip():
                logger.warning(f"SKIPPED empty: {file_path}")
                return None
            return text
        except UnicodeDecodeError:
            # Fall back to charset detection
            logger.debug(f"Could not decode {file_path.name} as UTF-8.")
            charset_match = from_bytes(data).best()
            if not charset_match:
                logger.error(f"SKIPPING {file_path.name} - failed to detect text encoding")
                return None
            logger.debug(f"Decode {file_path.name} with {charset_match.encoding}.")
            return str(charset_match)


def process_single_file(file_path, unit_bits, simprint_bits, granular, content, model_version):
    # type: (Path, int, int, bool, bool, int) -> dict | None
    """Process a single file and generate ISCC.

    Returns:
        Dictionary with ISCC result or None if processing failed
    """
    logger.debug(f"Processing {file_path.name}")

    try:
        text = read_text_from_file(file_path)
        if text is None:
            return None

        # Build options for create() function
        options = {
            "bits": unit_bits,
            "bits_granular": simprint_bits,
            "model_version": model_version,
        }

        # Add content flag if requested
        if content:
            options["contents"] = True

        # Generate ISCC
        sct_meta = create(text, granular=granular, **options)

        return {"iscc": sct_meta.iscc, "filename": file_path.as_posix(), "meta": sct_meta}

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def format_index_features(feature_set):
    # type: (object) -> list[dict]
    """Format features in index format to dictionaries."""
    features_list = []
    for i in range(len(feature_set.simprints)):
        feature_dict = {"simprint": feature_set.simprints[i]}
        if feature_set.offsets:
            feature_dict["offset"] = feature_set.offsets[i]
        if feature_set.sizes:
            feature_dict["size"] = feature_set.sizes[i]
        if feature_set.contents:
            feature_dict["content"] = feature_set.contents[i]
        features_list.append(feature_dict)
    return features_list


def format_object_features(feature_set):
    # type: (object) -> list[dict]
    """Format features in object format to dictionaries."""
    return [feature.model_dump() for feature in feature_set.simprints]


def format_json_features(features):
    # type: (list) -> list[dict]
    """Format features for JSON output."""
    features_list = []
    for feature_set in features:
        # Check if we're in index format
        if feature_set.simprints and isinstance(feature_set.simprints[0], str):
            features_list.extend(format_index_features(feature_set))
        else:
            features_list.extend(format_object_features(feature_set))
    return features_list


def output_json(result, json_indent, console, progress):
    # type: (dict, int | None, Console | None, Progress | None) -> None
    """Output result in JSON format (NDJSON by default, pretty with --pretty flag)."""
    output = {
        "iscc": result["iscc"],
        "filename": result["filename"],
    }

    # Add features if available
    if result.get("meta") and result["meta"].features:
        output["features"] = format_json_features(result["meta"].features)

    # Use rich.json for pretty output, compact JSON for NDJSON
    if json_indent is not None:
        # Pretty output using rich.json
        json_str = json.dumps(output, indent=json_indent)
        if console:
            console.print(JSON(json_str))
        else:
            # Fallback to regular indented JSON if no console
            typer.echo(json_str)
    else:
        # Compact NDJSON format
        output_text = json.dumps(output)
        if console and progress:
            console.print(output_text, markup=False, highlight=False)
        else:
            typer.echo(output_text)


def escape_content(content, max_length=50):
    # type: (str, int) -> str
    """Escape content for single-line output and truncate if needed.

    Args:
        content: The content to escape
        max_length: Maximum length before truncation (0 = no limit)

    Returns:
        Escaped and optionally truncated content
    """
    escaped = (
        content.replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    if max_length > 0 and len(escaped) > max_length:
        return escaped[:max_length] + "..."
    return escaped


def print_line(line, console, progress):
    # type: (str, Console | None, Progress | None) -> None
    """Print a line of output."""
    if console and progress:
        console.print(line, markup=False, highlight=False)
    else:
        typer.echo(line)


def format_feature_parts_index(feature_set, index, include_content, truncate):
    # type: (object, int, bool, int) -> list[str]
    """Format parts for a feature in index format."""
    parts = [f"  {feature_set.simprints[index]}"]
    if feature_set.offsets:
        parts.append(str(feature_set.offsets[index]))
    if feature_set.sizes:
        parts.append(str(feature_set.sizes[index]))
    if include_content and feature_set.contents:
        parts.append(escape_content(feature_set.contents[index], truncate))
    return parts


def format_feature_parts_object(feature, include_content, truncate):
    # type: (object, bool, int) -> list[str]
    """Format parts for a feature in object format."""
    parts = [f"  {feature.simprint}"]
    if feature.offset is not None:
        parts.append(str(feature.offset))
    if feature.size is not None:
        parts.append(str(feature.size))
    if include_content and hasattr(feature, "content") and feature.content is not None:
        parts.append(escape_content(feature.content, truncate))
    return parts


def output_index_format_features(feature_set, console, progress, include_content, truncate):
    # type: (object, Console | None, Progress | None, bool, int) -> None
    """Output features in index format."""
    for i in range(len(feature_set.simprints)):
        parts = format_feature_parts_index(feature_set, i, include_content, truncate)
        detail_line = " ".join(parts)
        print_line(detail_line, console, progress)


def output_object_format_features(feature_set, console, progress, include_content, truncate):
    # type: (object, Console | None, Progress | None, bool, int) -> None
    """Output features in object format."""
    for feature in feature_set.simprints:
        parts = format_feature_parts_object(feature, include_content, truncate)
        detail_line = " ".join(parts)
        print_line(detail_line, console, progress)


def output_text_features(features, console, progress, include_content, truncate):
    # type: (list, Console | None, Progress | None, bool, int) -> None
    """Output granular features in text format."""
    for feature_set in features:
        # Check format and output simprints with details
        if feature_set.simprints and isinstance(feature_set.simprints[0], str):
            output_index_format_features(feature_set, console, progress, include_content, truncate)
        else:
            output_object_format_features(feature_set, console, progress, include_content, truncate)


def output_text(result, granular, console, progress, include_content, truncate):
    # type: (dict, bool, Console | None, Progress | None, bool, int) -> None
    """Output result in text format."""
    line = f"{result['iscc']} {result['filename']}"
    print_line(line, console, progress)

    # Add granular details if requested
    if granular and result.get("meta") and result["meta"].features:
        output_text_features(result["meta"].features, console, progress, include_content, truncate)


def create_progress(show_progress, total_files, need_console):
    # type: (bool, int, bool) -> tuple[Console | None, Progress | None]
    """Create progress bar components if needed.

    Args:
        show_progress: Whether to show progress bar
        total_files: Total number of files to process
        need_console: Whether a console is needed (e.g., for pretty JSON)
    """
    if not show_progress and not need_console:
        return None, None

    console = Console()

    if not show_progress:
        # Console needed but no progress bar
        return console, None

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
        refresh_per_second=10,
    )
    return console, progress


def setup_processing_environment(format, content, granular, paths):
    # type: (str, bool, bool, list[str]) -> tuple[list[Path], bool, int | None]
    """Setup the processing environment.

    Returns:
        Tuple of (files, show_progress, json_indent)
    """
    # Disable loguru by default
    logger.remove()

    # Validate options
    validate_options(format, content, granular)

    # Collect files
    files = collect_files(paths)
    if not files:
        typer.echo("Error: No files found matching any of the provided patterns.", err=True)
        raise typer.Exit(code=1)

    # Detect if output is being redirected (not a TTY)
    is_tty = sys.stdout.isatty()
    show_progress = len(files) > 1 and is_tty

    # Default to NDJSON (no indentation)
    json_indent = None

    return files, show_progress, json_indent


def setup_pretty_json(pretty, format, json_indent):
    # type: (bool, str, int | None) -> int | None
    """Setup JSON indentation based on pretty flag."""
    if pretty and format == "json":
        return 2
    return json_indent


def process_and_output_file(file_path, options, console, progress):
    # type: (Path, dict, Console | None, Progress | None) -> None
    """Process a single file and output the result."""
    result = process_single_file(
        file_path,
        options["unit_bits"],
        options["simprint_bits"],
        options["granular"],
        options["content"],
        options["model_version"],
    )

    if result:
        # Output result based on format
        if options["format"] == "json":
            output_json(result, options["json_indent"], console, progress)
        else:
            output_text(
                result,
                options["granular"],
                console,
                progress,
                options["content"],
                options["truncate"],
            )


def run_processing_loop(files, options, console, progress):
    # type: (list[Path], dict, Console | None, Progress | None) -> None
    """Run the main processing loop."""
    task_id = None

    try:
        if progress:
            progress.start()
            task_id = progress.add_task("Processing files...", total=len(files))

        for file_path in files:
            process_and_output_file(file_path, options, console, progress)

            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1)

    finally:
        if progress:
            progress.stop()


def process_files(
    paths,
    format,
    unit_bits,
    simprint_bits,
    granular,
    pretty,
    content,
    truncate,
    model_version,
):
    # type: (list[str], str, int, int, bool, bool, bool, int, int) -> None
    """Process files and generate ISCC codes."""

    # Setup environment
    files, show_progress, json_indent = setup_processing_environment(
        format, content, granular, paths
    )

    # Apply pretty flag to JSON indentation
    json_indent = setup_pretty_json(pretty, format, json_indent)

    # Create progress components (need console if pretty JSON is requested)
    need_console = json_indent is not None and format == "json"
    console, progress = create_progress(show_progress, len(files), need_console)

    # Prepare options dictionary
    options = {
        "unit_bits": unit_bits,
        "simprint_bits": simprint_bits,
        "granular": granular,
        "content": content,
        "format": format,
        "json_indent": json_indent,
        "truncate": truncate,
        "model_version": model_version,
    }

    # Run processing loop
    run_processing_loop(files, options, console, progress)


@app.callback()
def callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    # type: (typer.Context, Optional[bool]) -> None
    """Generate ISCC codes for files or use subcommands for advanced operations."""
    # Show help if no subcommand and no arguments
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command(name="create")
def create_command(
    paths: list[str] = typer.Argument(
        ...,
        help="Path(s) to text files (supports glob patterns)",
    ),
    format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: text or json",
    ),
    unit_bits: int = typer.Option(
        256,
        "--unit-bits",
        "-u",
        help="Bit-length of ISCC-UNIT",
    ),
    simprint_bits: int = typer.Option(
        256,
        "--simprint-bits",
        "-s",
        help="Bit-length of ISCC-SIMPRINTs",
    ),
    granular: bool = typer.Option(
        False,
        "--granular",
        "-g",
        help="Activate granular simprint processing",
    ),
    pretty: bool = typer.Option(
        False,
        "--pretty",
        "-p",
        help="Output pretty JSON",
    ),
    content: bool = typer.Option(
        False,
        "--content",
        "-c",
        help="Include chunked text content in output (requires --granular)",
    ),
    truncate: int = typer.Option(
        50,
        "--truncate",
        "-t",
        help="Max length for content output (0 = no limit)",
    ),
    model_version: Optional[int] = typer.Option(
        None,
        "--model-version",
        "-m",
        help="Model version (0=minilm-l12, 1=embeddinggemma-300m)",
    ),
):
    # type: (list[str], str, int, int, bool, bool, bool, int, int|None) -> None
    """Generate Semantic Text-Codes for text files."""
    # Use environment-configured default if not specified
    if model_version is None:
        model_version = sct_opts.model_version
    process_files(
        paths, format, unit_bits, simprint_bits, granular, pretty, content, truncate, model_version
    )


@app.command()
def demo():
    # type: () -> None
    """Launch Gradio web interface."""
    try:
        from iscc_sct.demo import demo as gradio_demo

        typer.echo("Launching Gradio demo...")
        gradio_demo.launch(inbrowser=True)
    except ImportError:
        typer.echo(
            "Error: Gradio is not installed. Install with: uv sync --extra demo",
            err=True,
        )
        raise typer.Exit(code=1)


def _determine_versions_to_install(model_version, console):
    # type: (Optional[List[int]], Console) -> list[int]
    """Determine and validate which model versions to install."""
    if model_version is None:
        return sorted(MODEL_REGISTRY.keys())

    versions_to_install = []
    for v in model_version:
        if v not in MODEL_REGISTRY:
            available = ", ".join(str(ver) for ver in sorted(MODEL_REGISTRY.keys()))
            console.print(
                f"[red]Error: Model version {v} not found. Available versions: {available}[/red]"
            )
            raise typer.Exit(code=1)
        versions_to_install.append(v)
    return sorted(set(versions_to_install))


def _verify_model_integrity(config, model_dir, quiet, console):
    # type: (object, Path, bool, Console) -> bool
    """Verify integrity of model files. Returns True if valid, False otherwise."""
    try:
        for filename, checksum in zip(config.filenames, config.checksums):
            file_path = model_dir / filename
            check_integrity(file_path, checksum)
        return True
    except RuntimeError:
        return False


def _process_verify_only(version, config, model_dir, quiet, console):
    # type: (int, object, Path, bool, Console) -> tuple[int, str, str]
    """Process a model in verify-only mode."""
    all_files_exist = all((model_dir / fname).exists() for fname in config.filenames)

    if not all_files_exist:
        if not quiet:
            console.print("  [red]✗ Files missing[/red]")
        return (version, config.name, "Missing")

    if _verify_model_integrity(config, model_dir, quiet, console):
        if not quiet:
            console.print("  [green]✓ Integrity verified[/green]")
        return (version, config.name, "OK")
    else:
        if not quiet:
            console.print("  [red]✗ Integrity check failed[/red]")
        return (version, config.name, "Failed")


def _process_install_mode(version, config, model_dir, force, timeout, quiet, console):
    # type: (int, object, Path, bool, int, bool, Console) -> tuple[int, str, str]
    """Process a model in install mode."""
    all_files_exist = all((model_dir / fname).exists() for fname in config.filenames)

    if all_files_exist and not force:
        if _verify_model_integrity(config, model_dir, quiet, console):
            if not quiet:
                console.print("  [green]✓ Already installed and verified[/green]")
            return (version, config.name, "OK")
        else:
            if not quiet:
                console.print("  [yellow]⚠ Integrity check failed - re-downloading[/yellow]")

    download_model_files(config, model_dir, timeout, quiet, console)
    return (version, config.name, "OK")


def _process_model_version(version, verify_only, force, quiet, console):
    # type: (int, bool, bool, bool, Console) -> tuple[int, str, str]
    """Process a single model version."""
    config = get_model_config(version)
    model_dir = get_model_path(version)
    timeout = sct_opts.download_timeout

    if not quiet:
        console.print(f"[bold cyan]Model v{version}:[/bold cyan] {config.name}")

    if verify_only:
        return _process_verify_only(version, config, model_dir, quiet, console)
    else:
        return _process_install_mode(version, config, model_dir, force, timeout, quiet, console)


def _display_install_summary(results, quiet, console):
    # type: (list[tuple[int, str, str]], bool, Console) -> None
    """Display installation summary table."""
    if quiet:
        return

    console.print("[bold]Summary[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Version", style="cyan")
    table.add_column("Model Name")
    table.add_column("Status")
    table.add_column("Location")

    for version, name, status in results:
        model_dir = get_model_path(version)
        status_icon = "[green]✓ OK[/green]" if status == "OK" else f"[red]✗ {status}[/red]"
        display_name = name if len(name) <= 30 else name[:27] + "..."
        location_str = str(model_dir)
        if len(location_str) > 40:
            location_str = "..." + location_str[-37:]
        table.add_row(f"v{version}", display_name, status_icon, location_str)

    console.print(table)
    console.print()


@app.command()
def install(
    model_version: Optional[List[int]] = typer.Option(
        None,
        "--model-version",
        "-m",
        help="Model version(s) to install (0, 1, or both if omitted)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download even if files exist and are valid",
    ),
    verify_only: bool = typer.Option(
        False,
        "--verify-only",
        "-v",
        help="Only verify existing models without downloading",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress bars and detailed output",
    ),
):
    # type: (Optional[List[int]], bool, bool, bool) -> None
    """Download and verify ISCC-SCT embedding models.

    Examples:
        iscc-sct install              # Install both v0 and v1
        iscc-sct install -m 0         # Install only v0
        iscc-sct install -m 0 -m 1    # Install both (explicit)
        iscc-sct install --force      # Re-download all models
        iscc-sct install --verify-only # Only check existing models
    """
    console = Console()
    versions_to_install = _determine_versions_to_install(model_version, console)

    if not quiet:
        console.print("\n[bold]Installing ISCC-SCT Models[/bold]")
        console.print("━" * console.width)
        console.print()

    results = []
    for version in versions_to_install:
        result = _process_model_version(version, verify_only, force, quiet, console)
        results.append(result)
        if not quiet:
            console.print()

    _display_install_summary(results, quiet, console)


def download_model_files(config, model_dir, timeout, quiet, console):
    # type: (object, Path, int, bool, Console) -> None
    """Download model files with progress tracking."""
    if quiet:
        # No progress bars in quiet mode
        for filename, url, checksum in zip(config.filenames, config.urls, config.checksums):
            dest_path = model_dir / filename
            download_file(url, dest_path, checksum, timeout)
    else:
        # Create progress bar for downloads
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            for filename, url, checksum in zip(config.filenames, config.urls, config.checksums):
                dest_path = model_dir / filename
                task_id = progress.add_task(f"  {filename}", total=None)

                try:
                    download_file(url, dest_path, checksum, timeout, progress, task_id)
                    progress.update(task_id, completed=True)
                except Exception as e:
                    progress.stop()
                    console.print(f"  [red]✗ Download failed: {e}[/red]")
                    raise

        console.print("  [green]✓ Integrity verified[/green]")


def main():
    # type: () -> None
    """Entry point for the CLI."""
    import sys

    # Check if we should insert 'create' as default command
    if len(sys.argv) > 1:
        known_commands = ["create", "demo", "install"]

        # Find first non-option argument
        first_non_option_idx = None
        for i in range(1, len(sys.argv)):
            if not sys.argv[i].startswith("-"):
                # Also skip the value after an option that takes a parameter
                if i > 1 and sys.argv[i - 1] in [
                    "-f",
                    "--format",
                    "-u",
                    "--unit-bits",
                    "-s",
                    "--simprint-bits",
                    "-t",
                    "--truncate",
                    "-m",
                    "--model-version",
                ]:
                    continue
                first_non_option_idx = i
                break

        # If we found a non-option argument that's not a known command
        if first_non_option_idx is not None:
            first_non_option = sys.argv[first_non_option_idx]
            if first_non_option not in known_commands:
                # Insert 'create' command at the beginning (after script name)
                sys.argv.insert(1, "create")

    app()


if __name__ == "__main__":  # pragma: no cover
    app()
