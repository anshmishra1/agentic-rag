from agentic_rag.ingestion.contracts import (
    Artifact,
    ArtifactKind,
    Derivation,
    GroundingBasis,
    RepresentationKind,
    SearchRecord,
    SourceLocator,
)
from agentic_rag.ingestion.parsers.base import ParseRun
from agentic_rag.ingestion.record_segmentation import segment_parse_run_records


class _WhitespaceTokenizer:
    def encode(self, text, **_kwargs):
        return list(range(len(text.split())))


class _ThreeWordSplitter:
    def split_text(self, text):
        words = text.split()
        return [" ".join(words[start:start + 3]) for start in range(0, len(words), 3)]


def _run(text: str) -> ParseRun:
    document_id = "d" * 64
    derivation = Derivation(producer="parser", revision="1")
    artifact = Artifact.from_text(
        document_id=document_id,
        kind=ArtifactKind.TEXT_BLOCK,
        text=text,
        derivation=derivation,
        locator=SourceLocator(page_start=1, page_end=1, page_label="1"),
    )
    record = SearchRecord.from_artifact(
        artifact,
        representation_kind=RepresentationKind.NATIVE_TEXT,
        search_text=text,
        grounding_basis=GroundingBasis.DIRECT_TEXT,
        retrieval_schema_id="multimodal_v3:test",
    )
    return ParseRun(
        parser_name="parser",
        parser_revision="1",
        document_id=document_id,
        retrieval_schema_id="multimodal_v3:test",
        selected_pages=(1,),
        artifacts=(artifact,),
        records=(record,),
        elapsed_seconds=0.1,
        peak_memory_mib=1.0,
    )


def test_oversized_proxy_splits_without_changing_its_artifact() -> None:
    original = _run("one two three four five six seven")

    segmented = segment_parse_run_records(
        original,
        embedding_model="unused",
        embedding_revision="unused",
        local_files_only=True,
        max_tokens=3,
        tokenizer=_WhitespaceTokenizer(),
        splitter=_ThreeWordSplitter(),
    )

    assert segmented.artifacts == original.artifacts
    assert [record.search_text for record in segmented.records] == [
        "one two three",
        "four five six",
        "seven",
    ]
    assert {record.artifact_id for record in segmented.records} == {
        original.artifacts[0].artifact_id
    }
    assert len({record.record_id for record in segmented.records}) == 3
    assert [record.metadata["segment_index"] for record in segmented.records] == [0, 1, 2]
    assert all(record.metadata["segment_count"] == 3 for record in segmented.records)
    assert all(
        record.metadata["parent_record_id"] == original.records[0].record_id
        for record in segmented.records
    )


def test_model_safe_proxy_is_left_unchanged() -> None:
    original = _run("one two three")

    segmented = segment_parse_run_records(
        original,
        embedding_model="unused",
        embedding_revision="unused",
        local_files_only=True,
        max_tokens=3,
        tokenizer=_WhitespaceTokenizer(),
        splitter=_ThreeWordSplitter(),
    )

    assert segmented is original
