import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SUITE_PATH = Path("eval_set/v3_end_to_end_questions.json")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_suite() -> dict:
    return json.loads(
        SUITE_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )


def test_v3_end_to_end_suite_has_the_reviewed_scope() -> None:
    suite = _load_suite()
    cases = suite["cases"]

    assert suite["schema_version"] == 1
    assert suite["suite_id"] == "v3-end-to-end-acceptance-v1"
    assert suite["status"] == "source_reviewed_pending_v3_artifact_ids"
    assert len(suite["documents"]) == 7
    assert len(cases) == 49
    assert Counter(case["execution_tier"] for case in cases) == {
        "e2e_core": 18,
        "e2e_extended": 31,
    }

    document_counts = Counter(case["document_key"] for case in cases)
    assert set(document_counts) == {
        document["document_key"] for document in suite["documents"]
    }
    assert min(document_counts.values()) >= suite["coverage_requirements"][
        "document_cases_per_document"
    ]
    assert {
        case["document_key"]
        for case in cases
        if case["execution_tier"] == "e2e_core"
    } == set(document_counts)


def test_v3_end_to_end_case_contracts_are_complete_and_unique() -> None:
    suite = _load_suite()
    document_keys = {document["document_key"] for document in suite["documents"]}
    case_ids = [case["case_id"] for case in suite["cases"]]

    assert len(case_ids) == len(set(case_ids))
    assert all(case_id.strip() == case_id and case_id for case_id in case_ids)

    for case in suite["cases"]:
        expected = case["expected"]
        assert case["document_key"] in document_keys
        assert case["question"].strip()
        assert case["review_status"] == suite["status"]
        assert expected["answer_source"] in {"document", "general", "clarify", "control"}
        assert expected["allowed_answer_statuses"]
        assert isinstance(expected["should_retrieve"], bool)
        assert isinstance(expected["citations_required"], bool)
        assert expected["max_retrievals"] >= 0
        assert expected["max_rewrites"] >= 0
        assert expected["max_corrections"] >= 0
        assert expected["expected_answer"].strip()


def test_controlled_and_natural_questions_form_matched_pairs() -> None:
    cases_by_pair: dict[str, list[dict]] = defaultdict(list)
    for case in _load_suite()["cases"]:
        if pair_id := case.get("pair_id"):
            cases_by_pair[pair_id].append(case)

    assert len(cases_by_pair) == 7
    for cases in cases_by_pair.values():
        assert len(cases) == 2
        assert {case["interaction_style"] for case in cases} == {
            "controlled",
            "natural_ambiguous",
        }
        assert len({case["document_key"] for case in cases}) == 1
        first, second = (case["expected"] for case in cases)
        assert first["answer_source"] == second["answer_source"] == "document"
        assert set(first["source_pages"]) & set(second["source_pages"])
        assert set(first["artifact_types"]) & set(second["artifact_types"])


def test_declared_coverage_is_present_in_the_cases() -> None:
    suite = _load_suite()
    cases = suite["cases"]
    requirements = suite["coverage_requirements"]

    assert set(requirements["required_interaction_styles"]).issubset(
        {case["interaction_style"] for case in cases}
    )
    assert set(requirements["required_response_formats"]).issubset(
        {case["expected"]["response_format"] for case in cases}
    )
    assert set(requirements["required_artifact_types"]).issubset(
        {
            artifact_type
            for case in cases
            for artifact_type in case["expected"]["artifact_types"]
        }
    )
    assert set(requirements["required_grounding_bases"]).issubset(
        {
            grounding_basis
            for case in cases
            for grounding_basis in case["expected"]["grounding_bases"]
        }
    )


def test_negative_and_non_retrieval_cases_define_safe_outcomes() -> None:
    cases = _load_suite()["cases"]

    unanswerable = [case for case in cases if case["interaction_style"] == "unanswerable"]
    assert len(unanswerable) == 7
    for case in unanswerable:
        expected = case["expected"]
        assert expected["allowed_answer_statuses"] == ["insufficient_evidence"]
        assert expected["citations_required"] is False
        assert expected["source_pages"] == []
        assert expected["max_corrections"] == 0

    no_retrieval = [case for case in cases if not case["expected"]["should_retrieve"]]
    assert {case["case_id"] for case in no_retrieval} == {
        "route-auto-general",
        "route-auto-clarify",
        "route-control",
    }
    for case in no_retrieval:
        expected = case["expected"]
        assert expected["citations_required"] is False
        assert expected["source_pages"] == []
        assert expected["max_retrievals"] == 0
        assert expected["max_rewrites"] == 0
        assert expected["max_corrections"] == 0


def test_conversation_cases_reference_an_earlier_turn_in_the_same_thread() -> None:
    cases_by_id = {case["case_id"]: case for case in _load_suite()["cases"]}
    dependent_cases = [case for case in cases_by_id.values() if case.get("depends_on")]

    assert {case["interaction_style"] for case in dependent_cases} == {
        "follow_up",
        "verify_previous",
    }
    for case in dependent_cases:
        previous = cases_by_id[case["depends_on"]]
        assert case["conversation_id"] == previous["conversation_id"]
        assert previous["turn_index"] == 1
        assert case["turn_index"] == 2
        assert case["document_key"] == previous["document_key"]


def test_repository_fixture_hash_matches_the_reviewed_manifest() -> None:
    suite = _load_suite()
    document = next(
        document
        for document in suite["documents"]
        if document["source_group"] == "repository"
    )
    source_path = Path(document["filename"])

    assert source_path.is_file()
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == document["sha256"]
