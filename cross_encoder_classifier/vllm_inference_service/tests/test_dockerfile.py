from pathlib import Path


def test_dockerfile_uses_corporate_vllm_and_dvc_s3() -> None:
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert (
        "FROM mlops-docker-local.artifactory.corp.ingos.ru/ai-platform/"
        "vllm:0.17.1-cp310-2026_03_31_14_51_59" in dockerfile
    )
    assert "dvc[s3]==3.1.0" in dockerfile
    assert "uv==0.7.21" in dockerfile
    assert "EXPOSE 8080" in dockerfile
    assert "EXPOSE 8000" not in dockerfile
    assert 'ENTRYPOINT ["/bin/bash"]' in dockerfile
    assert 'CMD ["./start.sh"]' in dockerfile
