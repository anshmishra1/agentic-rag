"""Artifact-aware evidence constraints for the V3 retrieval path."""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.policies.query_planning import EvidenceRequirement


_VISUAL_KINDS = frozenset({"figure", "page_image", "standalone_image"})
_STRUCTURED_KINDS = frozenset({*_VISUAL_KINDS, "table", "formula"})
_KINDS_BY_REQUIREMENT: dict[EvidenceRequirement, frozenset[str]] = {
    "text": frozenset(),
    "visual": _VISUAL_KINDS,
    "table": frozenset({"table"}),
    "formula": frozenset({"formula"}),
    "mixed": _STRUCTURED_KINDS,
}


def required_artifact_kinds(
    requirement: EvidenceRequirement | str | None,
) -> frozenset[str]:
    """Return artifact kinds that can satisfy one planned evidence lane."""

    return _KINDS_BY_REQUIREMENT.get(str(requirement or "text"), frozenset())


def document_matches_requirement(document, requirement: str | None) -> bool:
    required = required_artifact_kinds(requirement)
    if not required:
        return True
    metadata = getattr(document, "metadata", {}) or {}
    return str(metadata.get("artifact_kind") or "") in required


def evidence_requirement_is_satisfied(
    documents: Sequence,
    requirement: str | None,
) -> bool:
    """Check whether selected evidence contains the requested source form."""

    required = required_artifact_kinds(requirement)
    if not required:
        return bool(documents)
    return any(document_matches_requirement(document, requirement) for document in documents)


def _identity(document) -> tuple[str, str]:
    metadata = getattr(document, "metadata", {}) or {}
    return (
        str(metadata.get("record_id") or ""),
        str(metadata.get("artifact_id") or ""),
    )


def _location(document) -> tuple[str, str]:
    metadata = getattr(document, "metadata", {}) or {}
    return (
        str(metadata.get("parent_artifact_id") or ""),
        str(metadata.get("page_label") or metadata.get("page") or ""),
    )


def select_required_evidence(
    reranked: Sequence[tuple],
    rrf_ranked: Sequence[tuple],
    *,
    requirement: str | None,
    top_k: int,
    required_slots: int,
    companion_slots: int = 0,
) -> list[tuple]:
    """Retain bounded artifact-lane recall, then fill by reranker score.

    The incoming RRF order supplies the recall safeguard. The cross-encoder
    still orders every retained record and fills the remaining context slots.
    Ordinary text questions keep the reranker's existing top-k behavior.
    """

    if top_k < 1:
        return []
    if not required_artifact_kinds(requirement):
        return list(reranked[:top_k])

    budget = min(max(0, required_slots), top_k)
    required_ids = [
        _identity(document)
        for document, _ in rrf_ranked
        if document_matches_requirement(document, requirement)
    ][:budget]
    required_id_set = set(required_ids)

    selected = [
        item for item in reranked
        if _identity(item[0]) in required_id_set
    ]
    selected_ids = {_identity(document) for document, _ in selected}

    # Structured artifacts often need a caption or nearby explanatory block.
    # Retain the strongest same-page/parent companion already present in the
    # reranked pool rather than issuing another broad expansion query.
    anchor_locations = {
        _location(document)
        for document, _ in selected
    }
    if companion_slots > 0:
        companions_added = 0
        for item in reranked:
            if len(selected) >= top_k:
                break
            document = item[0]
            identity = _identity(document)
            if identity in selected_ids:
                continue
            parent_id, page = _location(document)
            if not any(
                (anchor_parent and anchor_parent == parent_id)
                or (anchor_page and anchor_page == page)
                for anchor_parent, anchor_page in anchor_locations
            ):
                continue
            selected.append(item)
            selected_ids.add(identity)
            companions_added += 1
            if companions_added >= companion_slots:
                break

    for item in reranked:
        if len(selected) >= top_k:
            break
        identity = _identity(item[0])
        if identity in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(identity)

    selected = selected[:top_k]
    selected.sort(key=lambda item: float(item[1]), reverse=True)
    return selected
