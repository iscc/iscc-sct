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
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def version_callback(value: bool):
    # type: (bool) -> None
    """Show version and exit."""
    if value:
        typer.echo(f"iscc-sct version {__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    # type: (Optional[bool]) -> None
    """ISCC-SCT: Semantic Code Text for cross-lingual similarity detection."""
    pass


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


def main():
    # type: () -> None
    """Entry point for the CLI."""
    app()


@app.command(name="create")
def create_command(
    paths: list[str] = typer.Argument(..., help="Path(s) to text files (supports glob patterns)"),
    bits: int = typer.Option(256, "--bits", "-b", help="Bit-length of Code"),
    granular: bool = typer.Option(False, "--granular", "-g", help="Activate granular processing"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Show debugging messages"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress informational messages"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write output to file instead of stdout"
    ),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text or json"),
):
    # type: (list[str], int, bool, bool, bool, Optional[Path], str) -> None
    """Generate Semantic Text-Codes for text files."""
    # Configure logging
    if not debug:
        logger.remove()

    # Validate format
    if format not in ["text", "json"]:
        typer.echo(f"Error: Invalid format '{format}'. Choose 'text' or 'json'.", err=True)
        raise typer.Exit(code=1)

    # Find matching files from all provided paths
    files = []
    for path in paths:
        matching_files = resolve_path(path)
        if not matching_files and not quiet:
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

    # Prepare output storage for JSON format
    results = [] if format == "json" else None

    # Detect if output is being redirected (not a TTY)
    is_tty = sys.stdout.isatty() if output is None else False

    # Process files with optional progress bar
    # Show progress bar only if: not quiet, multiple files, and outputting to terminal
    show_progress = not quiet and len(files) > 1 and is_tty

    # Open output file if specified (for streaming writes)
    output_file = None
    if output and format != "json":  # JSON needs to be complete before writing
        output_file = output.open("w", encoding="utf-8")

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

        for idx, file_path in enumerate(files):
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

                    # Generate ISCC
                    sct_meta = create(text, granular=granular, bits=bits)

                    # Format and output immediately
                    if format == "json":
                        result = {
                            "file": str(file_path),
                            "iscc": sct_meta.iscc,
                        }
                        if granular:
                            result["metadata"] = sct_meta.model_dump()
                        results.append(result)
                    else:
                        # Format line
                        if granular:
                            # For granular output, keep the detailed format
                            line = f"{repr(sct_meta)} {file_path}"
                        else:
                            if len(files) > 1:
                                # Multiple files: use checksum-style format (iscc  file)
                                line = f"{sct_meta.iscc}  {file_path}"
                            else:
                                # Single file: output only the ISCC code
                                line = sct_meta.iscc

                        # Stream output immediately
                        if output_file:
                            output_file.write(line + "\n")
                            output_file.flush()  # Ensure immediate write
                        else:
                            # Use console.print if progress bar is active, otherwise typer.echo
                            if console and progress:
                                # Disable markup to prevent ISCC codes from being interpreted as styles
                                console.print(line, markup=False, highlight=False)
                            else:
                                typer.echo(line)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                if debug:
                    raise
                continue

            finally:
                # Update progress bar if active
                if progress and task_id is not None:
                    progress.update(task_id, advance=1)

    finally:
        if progress:
            progress.stop()
        if output_file:
            output_file.close()

    # Handle JSON output (needs all results before writing)
    if format == "json":
        output_text = json.dumps(results, indent=2)
        if output:
            output.write_text(output_text, encoding="utf-8")
            if not quiet:
                typer.echo(f"Output written to {output}")
        else:
            typer.echo(output_text)
    elif output and not quiet:
        typer.echo(f"Output written to {output}")


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


if __name__ == "__main__":  # pragma: no cover
    app()
