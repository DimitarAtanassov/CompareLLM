from __future__ import annotations

from httpx import AsyncClient

DATASET = "books"
EMBEDDING_KEY = "fake:embed"
STORE_ID = f"{DATASET}::{EMBEDDING_KEY}"


async def _create_and_index(client: AsyncClient, store_id: str, embedding_key: str) -> None:
    created = await client.post(
        "/embeddings/stores", json={"store_id": store_id, "embedding_key": embedding_key}
    )
    assert created.status_code == 200

    indexed = await client.post(
        "/embeddings/index/docs",
        json={
            "store_id": store_id,
            "docs": [
                {"page_content": "the cat sat on the mat", "metadata": {"id": "a"}},
                {"page_content": "deep learning transforms language", "metadata": {"id": "b"}},
            ],
        },
    )
    assert indexed.status_code == 200
    assert len(indexed.json()["ids"]) == 2


async def test_index_and_query(client: AsyncClient) -> None:
    await _create_and_index(client, STORE_ID, EMBEDDING_KEY)

    response = await client.post(
        "/embeddings/query",
        json={"store_id": STORE_ID, "query": "the cat sat on the mat", "k": 1, "with_scores": True},
    )
    assert response.status_code == 200
    matches = response.json()["matches"]
    assert len(matches) == 1
    assert matches[0]["page_content"] == "the cat sat on the mat"
    assert matches[0]["score"] is not None


async def test_query_missing_store(client: AsyncClient) -> None:
    response = await client.post(
        "/embeddings/query", json={"store_id": "nope::fake:embed", "query": "x"}
    )
    assert response.status_code == 404


async def test_stores_listing(client: AsyncClient) -> None:
    await _create_and_index(client, STORE_ID, EMBEDDING_KEY)
    response = await client.get("/embeddings/stores")
    assert response.json()["stores"][STORE_ID] == EMBEDDING_KEY


async def test_compare_across_models(client: AsyncClient) -> None:
    await _create_and_index(client, f"{DATASET}::fake:embed", "fake:embed")
    await _create_and_index(client, f"{DATASET}::fake:embed2", "fake:embed2")

    response = await client.post(
        "/embeddings/compare",
        json={
            "dataset_id": DATASET,
            "embedding_models": ["fake:embed", "fake:embed2"],
            "query": "deep learning transforms language",
            "k": 2,
            "with_scores": True,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert set(results) == {"fake:embed", "fake:embed2"}
    for bucket in results.values():
        assert bucket["items"]
        assert bucket["items"][0]["page_content"] == "deep learning transforms language"


async def test_mmr_search(client: AsyncClient) -> None:
    await _create_and_index(client, STORE_ID, EMBEDDING_KEY)
    response = await client.post(
        "/embeddings/query",
        json={
            "store_id": STORE_ID,
            "query": "deep learning transforms language",
            "k": 2,
            "search_type": "mmr",
            "fetch_k": 2,
        },
    )
    assert response.status_code == 200
    assert len(response.json()["matches"]) == 2
