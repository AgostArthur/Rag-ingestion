from rag_ingestion.config import rewrite_loopback_url


def test_rewrite_loopback_local_maps_docker_hostname():
    assert (
        rewrite_loopback_url(
            "http://host.docker.internal:11434/v1", in_docker=False
        )
        == "http://localhost:11434/v1"
    )


def test_rewrite_loopback_docker_maps_localhost():
    assert (
        rewrite_loopback_url("http://localhost:11434/v1", in_docker=True)
        == "http://host.docker.internal:11434/v1"
    )
    assert (
        rewrite_loopback_url("http://127.0.0.1:11434", in_docker=True)
        == "http://host.docker.internal:11434"
    )


def test_rewrite_loopback_leaves_other_hosts():
    assert rewrite_loopback_url("http://qdrant:6333", in_docker=True) == "http://qdrant:6333"
    assert (
        rewrite_loopback_url("http://ollama:11434", in_docker=True) == "http://ollama:11434"
    )
    assert (
        rewrite_loopback_url("https://api.openai.com/v1", in_docker=True)
        == "https://api.openai.com/v1"
    )
    assert (
        rewrite_loopback_url("http://localhost:11434", in_docker=False)
        == "http://localhost:11434"
    )
