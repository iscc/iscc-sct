"""Command-line interface for ISCC-SCT."""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from charset_normalizer import from_bytes
from loguru import logger
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

from iscc_sct.main import create

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


def resolve_path(path_pattern):
    # type: (str) -> list[Path]
    """Resolve a path pattern to a list of file paths.

    Handles single files, directories (recursively), and glob patterns.

    Args:
        path_pattern: A file path, directory path, or glob pattern

    Returns:
        Sorted list of resolved file paths
    """
    # Expand user directory (~) if present
    expanded_pattern = Path(path_pattern).expanduser().as_posix()
    path = Path(expanded_pattern)

    # Case 1: Existing file
    if path.exists() and path.is_file():
        return [path]

    # Case 2: Existing directory - recursively find all files
    if path.exists() and path.is_dir():
        files = list(path.rglob("*"))
        return sorted([f for f in files if f.is_file()])

    # Case 3: Glob pattern
    # Check if pattern contains glob characters
    glob_chars = {"*", "?", "[", "]", "{", "}"}
    if any(char in expanded_pattern for char in glob_chars):
        # Use glob.glob for better pattern support
        import glob

        matches = glob.glob(expanded_pattern, recursive=True)
        return sorted([Path(m) for m in matches if Path(m).is_file()])

    # Case 4: Non-existent file or pattern with no matches
    return []


def process_files(
    paths,
    format,
    unit_bits,
    simprint_bits,
    granular,
    pretty,
    content,
):
    # type: (list[str], str, int, int, bool, bool, bool) -> None
    """Process files and generate ISCC codes."""

    # Disable loguru by default
    logger.remove()

    # Validate format
    if format not in ["text", "json"]:
        typer.echo(f"Error: Invalid format '{format}'. Choose 'text' or 'json'.", err=True)
        raise typer.Exit(code=1)

    # Validate content flag
    if content and not granular:
        typer.echo("Error: --content requires --granular to be enabled.", err=True)
        raise typer.Exit(code=1)

    # Find matching files from all provided paths
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
    files = unique_files

    if not files:
        typer.echo("Error: No files found matching any of the provided patterns.", err=True)
        raise typer.Exit(code=1)

    # Detect if output is being redirected (not a TTY)
    is_tty = sys.stdout.isatty()

    # Process files with optional progress bar
    # Show progress bar only if: multiple files and outputting to terminal
    show_progress = len(files) > 1 and is_tty

    # For JSON format, determine indentation (pretty formatting for single file or if --pretty is set)
    json_indent = 2 if (pretty or len(files) == 1) and format == "json" else None

    # Setup console and progress bar if needed
    console = Console() if show_progress else None
    progress = None
    task_id = None

    if show_progress:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=False,  # Keep progress bar visible
            refresh_per_second=10,
        )

    try:
        if progress:
            progress.start()
            task_id = progress.add_task("Processing files...", total=len(files))

        for file_path in files:
            logger.debug(f"Processing {file_path.name}")

            try:
                with file_path.open("rb") as file:
                    data = file.read()

                    # Try UTF-8 first
                    try:
                        text = data.decode("utf-8")
                        if not text.strip():
                            logger.warning(f"SKIPPED empty: {file_path}")
                            continue
                    except UnicodeDecodeError:
                        # Fall back to charset detection
                        logger.debug(f"Could not decode {file_path.name} as UTF-8.")
                        charset_match = from_bytes(data).best()
                        if not charset_match:
                            logger.error(
                                f"SKIPPING {file_path.name} - failed to detect text encoding"
                            )
                            continue
                        logger.debug(f"Decode {file_path.name} with {charset_match.encoding}.")
                        text = str(charset_match)

                    # Build options for create() function
                    options = {
                        "bits": unit_bits,
                        "bits_granular": simprint_bits,
                    }

                    # Add content flag if requested
                    if content:
                        options["contents"] = True

                    # Generate ISCC
                    sct_meta = create(text, granular=granular, **options)

                    # Format and output
                    if format == "json":
                        result = {
                            "iscc": sct_meta.iscc,
                            "filename": file_path.as_posix(),
                        }
                        if granular and sct_meta.features:
                            # Convert to object format for cleaner JSON
                            features_list = []
                            for feature_set in sct_meta.features:
                                # Check if we're in index format
                                if feature_set.simprints and isinstance(
                                    feature_set.simprints[0], str
                                ):
                                    # Index format - convert to list of dicts
                                    for i in range(len(feature_set.simprints)):
                                        feature_dict = {"simprint": feature_set.simprints[i]}
                                        if feature_set.offsets:
                                            feature_dict["offset"] = feature_set.offsets[i]
                                        if feature_set.sizes:
                                            feature_dict["size"] = feature_set.sizes[i]
                                        if feature_set.contents:
                                            feature_dict["content"] = feature_set.contents[i]
                                        features_list.append(feature_dict)
                                else:
                                    # Object format - use as is
                                    for feature in feature_set.simprints:
                                        features_list.append(feature.model_dump())
                            result["features"] = features_list

                        # Output JSON immediately as file is processed
                        output_text = json.dumps(result, indent=json_indent)
                        if console and progress:
                            console.print(output_text, markup=False, highlight=False)
                        else:
                            typer.echo(output_text)
                    else:
                        # Text format output - always include filename with forward slashes
                        line = f"{sct_meta.iscc} {file_path.as_posix()}"

                        # Output the main line
                        if console and progress:
                            console.print(line, markup=False, highlight=False)
                        else:
                            typer.echo(line)

                        # Add granular details if requested
                        if granular and sct_meta.features:
                            for feature_set in sct_meta.features:
                                # Check format and output simprints with details
                                if feature_set.simprints and isinstance(
                                    feature_set.simprints[0], str
                                ):
                                    # Index format
                                    for i in range(len(feature_set.simprints)):
                                        parts = [f"  {feature_set.simprints[i]}"]
                                        if feature_set.offsets:
                                            parts.append(str(feature_set.offsets[i]))
                                        if feature_set.sizes:
                                            parts.append(str(feature_set.sizes[i]))
                                        detail_line = " ".join(parts)
                                        if console and progress:
                                            console.print(
                                                detail_line, markup=False, highlight=False
                                            )
                                        else:
                                            typer.echo(detail_line)
                                else:
                                    # Object format
                                    for feature in feature_set.simprints:
                                        parts = [f"  {feature.simprint}"]
                                        if feature.offset is not None:
                                            parts.append(str(feature.offset))
                                        if feature.size is not None:
                                            parts.append(str(feature.size))
                                        detail_line = " ".join(parts)
                                        if console and progress:
                                            console.print(
                                                detail_line, markup=False, highlight=False
                                            )
                                        else:
                                            typer.echo(detail_line)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

            finally:
                # Update progress bar if active
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

    finally:
        if progress:
            progress.stop()


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    paths: list[str] = typer.Argument(
        default=None,
        help="Path(s) to text files (supports glob patterns)",
        show_default=False,
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
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    # type: (typer.Context, list[str]|None, str, int, int, bool, bool, bool, Optional[bool]) -> None
    """Default callback that runs create command when no subcommand is specified."""
    if ctx.invoked_subcommand is None:
        # No subcommand was invoked, run create as default
        # If no paths provided, show help
        if not paths:
            typer.echo(ctx.get_help())
            raise typer.Exit()

        process_files(paths, format, unit_bits, simprint_bits, granular, pretty, content)


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
):
    # type: (list[str], str, int, int, bool, bool, bool) -> None
    """Generate Semantic Text-Codes for text files."""
    process_files(paths, format, unit_bits, simprint_bits, granular, pretty, content)


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


def main():
    # type: () -> None
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
