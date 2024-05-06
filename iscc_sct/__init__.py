"""ISCC - Semantic-Code Text."""

import os
from platformdirs import PlatformDirs

__version__ = "0.1.0"
APP_NAME = "iscc-sct"
APP_AUTHOR = "iscc"
dirs = PlatformDirs(appname=APP_NAME, appauthor=APP_AUTHOR)
os.makedirs(dirs.user_data_dir, exist_ok=True)


from iscc_sct.utils import *
from iscc_sct.code_semantic_text import *
