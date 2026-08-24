import os
import shutil
import subprocess
from pathlib import Path


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def option_value(arguments: list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def test_start_script_hides_vllm_behind_gateway(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "start.sh"
    service_dir = tmp_path / "service"
    bin_dir = service_dir / "venv" / "bin"
    model_dir = service_dir / "artifacts" / "models" / "bge-reranker-v2-m3"
    output_dir = tmp_path / "output"
    bin_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    output_dir.mkdir()
    shutil.copy2(source, service_dir / "start.sh")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (bin_dir / "activate").write_text(
        'export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PATH"\n',
        encoding="utf-8",
    )
    write_executable(
        bin_dir / "vllm",
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${TEST_OUTPUT}/vllm"\nsleep 0.1\n',
    )
    write_executable(
        bin_dir / "uvicorn",
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${TEST_OUTPUT}/gateway"\nsleep 0.1\n',
    )

    environment = os.environ.copy()
    environment["TEST_OUTPUT"] = str(output_dir)
    result = subprocess.run(
        ["bash", str(service_dir / "start.sh")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    vllm_arguments = (output_dir / "vllm").read_text().splitlines()
    gateway_arguments = (output_dir / "gateway").read_text().splitlines()
    assert vllm_arguments[:2] == [
        "serve",
        "./artifacts/models/bge-reranker-v2-m3",
    ]
    assert option_value(vllm_arguments, "--host") == "127.0.0.1"
    assert option_value(vllm_arguments, "--served-model-name") == (
        "survey-cross-encoder"
    )
    assert option_value(vllm_arguments, "--gpu-memory-utilization") == "0.90"
    assert option_value(vllm_arguments, "--api-key")
    assert gateway_arguments[0] == "app.main:app"
    assert option_value(gateway_arguments, "--host") == "0.0.0.0"
    assert option_value(gateway_arguments, "--port") == "8080"


def test_start_script_pulls_dvc_model_before_serving(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "start.sh"
    service_dir = tmp_path / "service"
    bin_dir = service_dir / "venv" / "bin"
    artifacts_dir = service_dir / "artifacts"
    models_dir = artifacts_dir / "models"
    output_dir = tmp_path / "output"
    bin_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    (artifacts_dir / ".dvc").mkdir()
    output_dir.mkdir()
    shutil.copy2(source, service_dir / "start.sh")
    (models_dir / "bge-reranker-v2-m3.dvc").write_text(
        "outs:\n- md5: test.dir\n  path: bge-reranker-v2-m3\n",
        encoding="utf-8",
    )
    (bin_dir / "activate").write_text(
        'export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PATH"\n',
        encoding="utf-8",
    )
    write_executable(
        bin_dir / "dvc",
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$PWD" "$@" > "${TEST_OUTPUT}/dvc"\n'
        "mkdir -p models/bge-reranker-v2-m3\n"
        "printf '{}' > models/bge-reranker-v2-m3/config.json\n",
    )
    write_executable(
        bin_dir / "vllm",
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${TEST_OUTPUT}/vllm"\nsleep 0.1\n',
    )
    write_executable(
        bin_dir / "uvicorn",
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${TEST_OUTPUT}/gateway"\nsleep 0.1\n',
    )

    environment = os.environ.copy()
    environment.update(
        TEST_OUTPUT=str(output_dir),
        DVC_REMOTE="production-models",
        DVC_JOBS="4",
    )
    result = subprocess.run(
        ["bash", str(service_dir / "start.sh")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    dvc_arguments = (output_dir / "dvc").read_text().splitlines()
    assert dvc_arguments == [
        str(artifacts_dir),
        "pull",
        "models/bge-reranker-v2-m3.dvc",
        "--remote",
        "production-models",
        "--jobs",
        "4",
    ]
    assert (models_dir / "bge-reranker-v2-m3" / "config.json").is_file()
    assert (output_dir / "vllm").is_file()
    assert (output_dir / "gateway").is_file()
