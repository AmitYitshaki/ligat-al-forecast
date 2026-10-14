"""Exercise Plotly in the project interpreter's kernel and strip its notebook outputs."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import nbformat
import pytest
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def test_venv_kernel_plotly_and_output_stripping(tmp_path: Path) -> None:
    kernels = tmp_path / "kernels"
    spec = kernels / "ligat-test"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Ligat test interpreter",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    manager = KernelManager(
        kernel_name="ligat-test", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)])
    )
    code = (
        "import sys\nprint(sys.executable)\n"
        "import plotly.graph_objects as go\n"
        "go.Figure(go.Scatter(x=[1, 2], y=[2, 3])).show(renderer='plotly_mimetype')"
    )
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)])
    manager.start_kernel(
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    client = manager.blocking_client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=30)
        message_id = client.execute(code)
        outputs = []
        interpreter_output = ""
        found_plotly = False
        while True:
            message = client.get_iopub_msg(timeout=30)
            if message["parent_header"].get("msg_id") != message_id:
                continue
            content = message["content"]
            kind = message["header"]["msg_type"]
            if kind == "error":
                raise AssertionError(content["traceback"])
            if kind == "stream":
                interpreter_output += content["text"]
                outputs.append(
                    nbformat.v4.new_output("stream", name=content["name"], text=content["text"])
                )
            elif kind == "display_data":
                found_plotly |= "application/vnd.plotly.v1+json" in content["data"]
                outputs.append(
                    nbformat.v4.new_output(
                        "display_data", data=content["data"], metadata=content["metadata"]
                    )
                )
            elif kind == "status" and content["execution_state"] == "idle":
                break
        assert sys.executable in interpreter_output
        assert found_plotly
        notebook.cells[0].outputs = outputs
        notebook.cells[0].execution_count = 1
        result = subprocess.run(
            [sys.executable, "-m", "nbstripout"],
            input=nbformat.writes(notebook),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        stripped = nbformat.reads(result.stdout, as_version=4)
        assert stripped.cells[0].outputs == []
        assert stripped.cells[0].execution_count is None
        assert stripped.cells[0].source == code
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)


main = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/check_notebook_outputs.py")
)["main"]


@pytest.mark.parametrize("staged_clean", [True, False])
def test_hook_preserves_local_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    staged_clean: bool,
) -> None:
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1 + 1")])
    notebook.cells[0].execution_count = 1
    notebook.cells[0].outputs = [
        nbformat.v4.new_output("execute_result", data={"text/plain": "2"}, execution_count=1)
    ]
    path = tmp_path / "executed.ipynb"
    nbformat.write(notebook, path)
    original = path.read_bytes()
    run = subprocess.run
    clean = run(
        [sys.executable, "-m", "nbstripout"], input=original, capture_output=True, check=True
    ).stdout
    staged = clean if staged_clean else original

    def fake_git(args: list[str], **kwargs: Any) -> Any:
        if args[0] == "git":
            return subprocess.CompletedProcess(args, 0, stdout=staged)
        return run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_git)
    assert main([str(path)]) == (0 if staged_clean else 1)
    assert path.read_bytes() == original
