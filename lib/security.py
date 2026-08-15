"""Security helpers for untrusted Director paths and cache keys."""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit

import folder_paths


_CACHE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def safe_cache_key(value: object) -> str | None:
    """Return a filesystem-safe cache key, or ``None`` for untrusted values."""
    key = str(value or "").strip()
    return key if _CACHE_KEY_RE.fullmatch(key) else None


def resolve_input_file(name: str, *, subfolder: str = "") -> Path:
    """Resolve an existing file while containing symlinks under ComfyUI input/."""
    raw_name = str(name or "").strip().replace("\\", "/")
    raw_subfolder = str(subfolder or "").strip().replace("\\", "/")
    if not raw_name:
        raise ValueError("Missing input filename")

    input_dir = Path(folder_paths.get_input_directory())
    relative = Path(raw_subfolder) / Path(raw_name) if raw_subfolder else Path(raw_name)
    target = input_dir / relative
    if not folder_paths.is_within_directory(str(input_dir), str(target)):
        raise ValueError(f"Invalid input file path: {raw_name!r}")
    try:
        is_file = target.is_file()
    except OSError as exc:
        raise ValueError(f"Invalid input file path: {raw_name!r}") from exc
    if not is_file:
        raise FileNotFoundError(f"Input file not found: {raw_name}")
    return target.resolve()


def validate_llm_url(url: str) -> str:
    """Reject unsafe plaintext and special-address LLM destinations."""
    value = str(url or "").strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LLM URL must use http:// or https://")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("LLM URL must not contain embedded authentication")

    host = parsed.hostname.rstrip(".").lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None

    labels = host.split(".")
    ambiguous_numeric_host = address is None and all(
        label.isdigit()
        or (
            label.lower().startswith("0x")
            and len(label) > 2
            and all(char in "0123456789abcdef" for char in label[2:].lower())
        )
        for label in labels
    )
    if ambiguous_numeric_host:
        raise ValueError("LLM URL targets an ambiguous numeric address")

    if address is not None and not address.is_loopback and (
        address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise ValueError("LLM URL targets a restricted network address")

    if parsed.scheme == "http":
        is_local_name = (
            host == "localhost"
            or host.endswith(".localhost")
            or host.endswith(".local")
            or host in {"host.docker.internal", "gateway.docker.internal"}
            or "." not in host
        )
        is_local_address = address is not None and (address.is_loopback or address.is_private)
        if not is_local_name and not is_local_address:
            raise ValueError(
                "Plain HTTP LLM URLs must target a local service; use HTTPS for remote endpoints"
            )
    return value
