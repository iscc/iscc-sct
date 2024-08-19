import pathlib
import yaml


HERE = pathlib.Path(__file__).parent.absolute()


def convert_lf():  # pragma: no cover
    """Convert line endings to LF"""
    crlf = b"\r\n"
    lf = b"\n"
    extensions = {".py", ".toml", ".lock", ".txt", ".yml", ".sh", ".md"}
    n = 0
    for fp in HERE.parent.glob("**/*"):
        if fp.suffix in extensions:
            with open(fp, "rb") as infile:
                content = infile.read()
            if crlf in content:
                content = content.replace(crlf, lf)
                with open(fp, "wb") as outfile:
                    outfile.write(content)
                n += 1
    print(f"{n} files converted to LF")


def format_yml():
    for f in HERE.glob("**\*.yml"):
        with open(f, "rt", encoding="utf-8") as infile:
            data = yaml.safe_load(infile)
        with open(f, "wt", encoding="utf-8", newline="\n") as outf:
            yaml.safe_dump(
                data,
                outf,
                indent=2,
                width=80,
                encoding="utf-8",
                sort_keys=False,
                default_flow_style=False,
                default_style=">",
                allow_unicode=True,
                line_break="\n",
            )
