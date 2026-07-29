from pathlib import Path
import subprocess
import time


def test_process_inbox_copies_markdown_and_leaves_pdf(tmp_path: Path):
    app = tmp_path / "app"
    data = tmp_path / "data"
    scripts = app / "scripts"
    inbox = data / "inbox"
    scripts.mkdir(parents=True)
    inbox.mkdir(parents=True)
    (inbox / "report.md").write_text("# Report", encoding="utf-8")
    (inbox / "deferred.pdf").write_bytes(b"pdf")
    (scripts / "compile_and_enrich.py").write_text(
        'print("compiled=0 skipped=1 failed=0")\n', encoding="utf-8"
    )

    result = subprocess.run(
        ["bash", str(Path(__file__).parents[1] / "scripts/process_inbox.sh")],
        env={"APP_DIR": str(app), "DATA_DIR": str(data), "PATH": "/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert (data / "sources/report.md").read_text(encoding="utf-8") == "# Report"
    assert (inbox / "deferred.pdf").read_bytes() == b"pdf"
    assert "MinerU deferred: deferred.pdf" in result.stdout
    assert "No wiki changes; index and site build skipped" in result.stdout


def test_process_inbox_streams_compiler_progress_before_completion(tmp_path: Path):
    app = tmp_path / "app"
    data = tmp_path / "data"
    scripts = app / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "compile_and_enrich.py").write_text(
        "import time\n"
        'print("compile.start source=report.md", flush=True)\n'
        "time.sleep(1)\n"
        'print("compiled=0 skipped=1 failed=0", flush=True)\n',
        encoding="utf-8",
    )
    process = subprocess.Popen(
        ["bash", str(Path(__file__).parents[1] / "scripts/process_inbox.sh")],
        env={"APP_DIR": str(app), "DATA_DIR": str(data), "PATH": "/usr/bin:/bin"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None

    started = time.monotonic()
    first_line = process.stdout.readline()
    elapsed = time.monotonic() - started
    process.wait(timeout=3)

    assert first_line == "compile.start source=report.md\n"
    assert elapsed < 0.5
