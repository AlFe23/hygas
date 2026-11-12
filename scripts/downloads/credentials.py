"""Local Earthdata token storage; no network calls or secret logging."""
from __future__ import annotations

import os
from pathlib import Path


DEFAULT_TOKEN_FILE = Path.home() / ".config" / "hygas" / "earthdata-token"


def validate_token(token):
    token = token.strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("Il file deve contenere soltanto il token Earthdata, su una riga.")
    return token


def read_token(path=DEFAULT_TOKEN_FILE):
    return validate_token(Path(path).expanduser().read_text(encoding="utf8"))


def save_token(token, path=DEFAULT_TOKEN_FILE, *, replace=False):
    """Store the token outside the repository with owner-only file permissions."""
    token = validate_token(token)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if replace else os.O_EXCL)
    fd = os.open(path, flags, 0o600)
    try:
        if os.name == "posix":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf8") as stream:
            fd = None
            stream.write(token + "\n")
    finally:
        if fd is not None:
            os.close(fd)
    return path
