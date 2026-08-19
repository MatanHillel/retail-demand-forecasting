"""Capture the Streamlit screens as PNG files for the business presentation (US-38, PRD §53.2).

Slide 10 of the deck is "product demo screenshots". A screenshot has to be a picture of the real
running application — a redrawn mock-up would be a claim about the product rather than evidence of
it — so this script boots the app exactly as a user would (``streamlit run src/app/Home.py``),
drives a headless Chrome over it and saves one full-page PNG per screen under
``docs/img/screens/``.

**This is a developer tool, not a pipeline step.** It is run by hand when the screens change; its
output is committed, and ``scripts/build_presentation.py`` only ever reads the committed PNGs. That
matters for two reasons:

* it opens no :class:`~pipeline.run_context.RunContext` and writes nothing under ``artifacts/``, so
  it cannot disturb a run's bookkeeping (§39);
* Selenium and a real Chrome are needed *here* and nowhere else, so they are imported inside the
  function that needs them and are deliberately absent from ``requirements.txt`` — CI installs that
  file and must not be asked to download a browser to run the test suite.

Usage::

    python scripts/capture_screens.py                 # all screens
    python scripts/capture_screens.py --screens 1 4   # a subset, by screen number
    python scripts/capture_screens.py --keep-open     # leave the server up for a manual look

The app refuses to render anything when ``run_log.json`` does not say ``success`` (US-27's status
banner calls ``st.stop()``), so the capture would silently produce six pictures of an error box.
The run status is therefore checked up front and the script exits non-zero rather than writing
misleading screenshots.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:  # the script may run without an editable install
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from pipeline import paths  # noqa: E402  (must follow the sys.path bootstrap)

#: Where the committed screenshots live. ``build_presentation.py`` reads this directory.
SCREENS_DIR: Path = paths.DOCS_DIR / "img" / "screens"

#: Browser window size. Wide enough that Streamlit lays out its columns side by side (below
#: ~1000 px it stacks them and the screenshot stops looking like the product), tall enough that the
#: first screenful is the interesting one.
WINDOW_SIZE: tuple[int, int] = (1600, 1200)

#: Seconds to wait for the server to answer, and for a page to finish its first render. Streamlit
#: streams a page in over a websocket after the HTML arrives, so "document ready" is far too early:
#: the browser has the page long before the app has run. The real signal is Streamlit's own status
#: widget, which shows "RUNNING…" while a script is executing — :func:`_wait_for_render` waits for
#: it to disappear, and these are only the bounds around that wait.
SERVER_TIMEOUT_S: float = 90.0
RENDER_TIMEOUT_S: float = 120.0
RENDER_SETTLE_S: float = 3.0


@dataclass(frozen=True)
class Screen:
    """One capture target: a screen number, the file stem to write and the app path to visit."""

    number: int
    stem: str
    #: Streamlit's multi-page URL path. ``""`` is the entry point (``Home.py``); the others are the
    #: page file name without its ``N_`` ordering prefix and ``.py`` suffix.
    url_path: str

    @property
    def output(self) -> Path:
        return SCREENS_DIR / f"S{self.number}_{self.stem}.png"


#: The seven screens of PRD §33, in the order the deck and the demo walk them.
SCREENS: tuple[Screen, ...] = (
    Screen(1, "executive_dashboard", ""),
    Screen(2, "product_forecasts", "Product_Forecasts"),
    Screen(3, "product_detail", "Product_Detail"),
    Screen(4, "model_evaluation", "Model_Evaluation"),
    Screen(5, "inventory_policy", "Inventory_Policy"),
    Screen(6, "pipeline_data_quality", "Pipeline_Data_Quality"),
    Screen(7, "data_insights", "Data_Insights"),
)


def _free_port() -> int:
    """Ask the OS for a port nobody is using, so a second capture never collides with a first."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_status() -> str:
    """The audited run's status, or a marker string when there is no run log at all."""
    if not paths.RUN_LOG.is_file():
        return "missing"
    return str(json.loads(paths.RUN_LOG.read_text(encoding="utf-8")).get("status", "missing"))


def _wait_for_server(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=5) as response:  # noqa: S310 (a localhost URL we just built)
                if response.status == 200:
                    return
        except (URLError, OSError) as error:  # not up yet
            last_error = error
        time.sleep(1.0)
    raise TimeoutError(f"streamlit did not answer {url} within {timeout_s:.0f}s ({last_error})")


def _start_streamlit(port: int) -> subprocess.Popen[bytes]:
    """Launch the app headless on ``port``, with the usage prompt and telemetry disabled."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(paths.PROJECT_ROOT / "src" / "app" / "Home.py"),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.fileWatcherType",
        "none",
        # Hide the developer toolbar ("Deploy", the running-man menu). It belongs to the editing
        # session, not to the product, and a slide is a picture of the product.
        "--client.toolbarMode",
        "viewer",
    ]
    return subprocess.Popen(
        command,
        cwd=paths.PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def _make_driver():  # type: ignore[no-untyped-def]  # a selenium WebDriver; imported lazily
    """Build a headless Chrome. Selenium Manager resolves the driver for the installed Chrome."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError as error:  # pragma: no cover - developer machines only
        raise SystemExit(
            "selenium is not installed. This capture tool is a developer utility and is "
            "deliberately not part of requirements.txt; install it with "
            "`uv pip install selenium` and re-run."
        ) from error

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--force-device-scale-factor=2")  # a crisp picture on a projector
    options.add_argument(f"--window-size={WINDOW_SIZE[0]},{WINDOW_SIZE[1]}")
    return webdriver.Chrome(options=options)


def _wait_for_render(driver, timeout_s: float = RENDER_TIMEOUT_S) -> None:  # type: ignore[no-untyped-def]
    """Block until Streamlit has finished running the page's script.

    Streamlit serves the HTML shell immediately and then streams the widgets in over a websocket,
    so a screenshot taken on page load catches grey placeholder blocks. While the script runs, the
    app mounts a status widget (``data-testid="stStatusWidget"``) reading "RUNNING…"; when it
    unmounts, the page is drawn. A short settle pause afterwards lets Matplotlib images decode.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        still_running = driver.execute_script(
            "return document.querySelectorAll('[data-testid=\"stStatusWidget\"]').length > 0;"
        )
        if not still_running:
            time.sleep(RENDER_SETTLE_S)
            return
        time.sleep(0.5)
    raise TimeoutError(f"the page was still running after {timeout_s:.0f}s")


def capture(screens: tuple[Screen, ...], port: int) -> list[Path]:
    """Screenshot every screen in ``screens`` and return the files written."""
    base_url = f"http://127.0.0.1:{port}"
    server = _start_streamlit(port)
    written: list[Path] = []
    try:
        _wait_for_server(f"{base_url}/_stcore/health", SERVER_TIMEOUT_S)
        driver = _make_driver()
        try:
            for screen in screens:
                driver.get(f"{base_url}/{screen.url_path}")
                _wait_for_render(driver)
                screen.output.parent.mkdir(parents=True, exist_ok=True)
                driver.save_screenshot(str(screen.output))
                size_kb = screen.output.stat().st_size / 1024
                print(f"  screen {screen.number}: {screen.output.name} ({size_kb:,.0f} kB)")
                written.append(screen.output)
        finally:
            driver.quit()
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover - a wedged server
            server.kill()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--screens",
        nargs="*",
        type=int,
        choices=[screen.number for screen in SCREENS],
        help="screen numbers to capture (default: all seven)",
    )
    parser.add_argument("--port", type=int, default=None, help="port for the app (default: free)")
    args = parser.parse_args(argv)

    status = _run_status()
    if status != "success":
        print(
            f"run_log.json status is {status!r}, not 'success' — every screen would render the "
            "status banner instead of the product. Re-run the pipeline before capturing.",
            file=sys.stderr,
        )
        return 2

    wanted = tuple(s for s in SCREENS if args.screens is None or s.number in args.screens)
    port = args.port or _free_port()
    print(f"capturing {len(wanted)} screen(s) from http://127.0.0.1:{port}")
    written = capture(wanted, port)
    print(f"wrote {len(written)} screenshot(s) to {SCREENS_DIR.relative_to(paths.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
