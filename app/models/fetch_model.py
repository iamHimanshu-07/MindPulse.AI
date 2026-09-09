"""
Download the trained emotion-detection artefacts on first run.

This module is invoked by ``app/app.py`` at startup when the on-disk
``mental_health_model.pkl`` is missing. It is designed for Streamlit
Community Cloud, where the 268 MB model is too large to commit to git.

Configuration
-------------
The download *base URL* is resolved in this order:

  1. Environment variable ``MODEL_URL`` (set as a Streamlit Cloud secret
     via the dashboard's advanced settings, or exported locally).
  2. ``st.secrets["MODEL_URL"]`` if running under Streamlit.
  3. The baked-in default (``HF_DEFAULT_BASE_URL`` below).

``MODEL_URL`` may point at either a *base* (no filename) or directly at
``mental_health_model.pkl`` — if the latter, the filename is stripped and
the directory is used as the base. Two artefacts are fetched:

  - ``<base>/mental_health_model.pkl``   (~268 MB, the trained pipeline)
  - ``<base>/mlb.pkl``                   (~644 B, the label binariser)

The downloads are streamed, progress-logged, and idempotent — already
present files are skipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Callable

try:
    import requests  # available in Streamlit's default image
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:
    import streamlit as st
    _HAVE_ST = True
except ImportError:  # pragma: no cover
    _HAVE_ST = False

# Default base URL of the artefacts on Hugging Face Hub. Override via
# the MODEL_URL env var (or st.secrets["MODEL_URL"]) to point at your
# own release / bucket.
HF_DEFAULT_BASE_URL = (
    "https://huggingface.co/iamHimanshu-07/MindPulse.AI/resolve/main"
)

# Artefact filenames (relative to the base URL).
MODEL_FILENAME = "mental_health_model.pkl"
MLB_FILENAME = "mlb.pkl"

# Paths where the artefacts are expected by app.py.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / MODEL_FILENAME
MLB_PATH = APP_DIR / MLB_FILENAME

# Stream chunk size for the download (1 MiB).
CHUNK = 1024 * 1024


# --------------------------------------------------------------------------- #
# URL resolution
# --------------------------------------------------------------------------- #
def _resolve_base_url() -> Optional[str]:
    """Pick the best available base URL from env, secrets, or default.

    Accepts either a *base* (no trailing slash, no filename) or a direct
    URL ending in ``MODEL_FILENAME`` — in the latter case the filename is
    stripped and the directory used as the base.
    """
    def _normalise(value: str) -> str:
        if value.endswith(MODEL_FILENAME):
            value = value[: -len(MODEL_FILENAME)]
        return value.rstrip("/")

    env_url = os.environ.get("MODEL_URL")
    if env_url:
        return _normalise(env_url)
    if _HAVE_ST:
        try:
            secrets_url = st.secrets.get("MODEL_URL")
            if secrets_url:
                return _normalise(str(secrets_url))
        except Exception:
            # No secrets file / no permission / not running under streamlit
            pass
    return HF_DEFAULT_BASE_URL


# --------------------------------------------------------------------------- #
# Streaming download
# --------------------------------------------------------------------------- #
def _stream_download(
    url: str,
    dest: Path,
    progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
) -> None:
    """Stream a remote file to ``dest`` with atomic .part → rename.

    ``progress_cb`` receives (bytes_done, total_bytes_or_None).
    """
    if requests is None:
        raise RuntimeError(
            "The 'requests' package is required to download artefacts. "
            "Add `requests` to requirements.txt."
        )

    def log(msg: str) -> None:
        if _HAVE_ST:
            st.write(msg)
        else:
            print(msg, file=sys.stderr)

    log(f"[fetch_model] {url} → {dest.name}")
    sess = requests.Session()
    token = os.environ.get("HF_TOKEN") or (st.secrets.get("HF_TOKEN") if _HAVE_ST else None)
    if token:
        sess.headers.update({"Authorization": f"Bearer {token}"})
    sess.headers.update({"User-Agent": "MindPulse.AI/1.0 (+streamlit)"})

    partial = dest.with_suffix(dest.suffix + ".part")
    with sess.get(url, stream=True, timeout=600, allow_redirects=True) as r:
        if r.status_code == 404:
            raise FileNotFoundError(
                f"Artefact not found at {url}. "
                "Set MODEL_URL to a reachable base URL where both "
                f"{MODEL_FILENAME} and {MLB_FILENAME} live side-by-side."
            )
        r.raise_for_status()

        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with partial.open("wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if progress_cb is not None:
                    try:
                        progress_cb(done, total or None)
                    except Exception:
                        # Never let a progress UI break the download.
                        pass

        partial.replace(dest)

    log(f"[fetch_model]    wrote {done / (1024 * 1024):.1f} MiB")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def ensure_model(
    progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
    force: bool = False,
) -> Path:
    """Make sure both artefacts exist on disk; download missing ones.

    Returns the path to the model. Raises ``FileNotFoundError`` if the
    remote download fails.
    """
    base = _resolve_base_url()
    if not base:
        raise FileNotFoundError(
            "No MODEL_URL configured and no baked-in default available. "
            "Set the MODEL_URL environment variable / Streamlit secret."
        )

    # MLB is tiny; only fetch it if missing. We pass a no-op progress
    # callback so it doesn't compete with the model-download progress UI.
    if force or not (MLB_PATH.exists() and MLB_PATH.stat().st_size > 0):
        _stream_download(f"{base}/{MLB_FILENAME}", MLB_PATH, progress_cb=None)

    # The model is the big one — only fetched if missing (unless forced).
    if force or not (MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0):
        _stream_download(f"{base}/{MODEL_FILENAME}", MODEL_PATH, progress_cb=progress_cb)

    return MODEL_PATH


# --------------------------------------------------------------------------- #
# CLI: `python app/models/fetch_model.py` for local warm-up / testing.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    ensure_model()
    print(f"OK: {MODEL_PATH} ({MODEL_PATH.stat().st_size / (1024 * 1024):.1f} MiB)")
    print(f"OK: {MLB_PATH} ({MLB_PATH.stat().st_size} B)")
