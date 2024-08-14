import pathlib


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
