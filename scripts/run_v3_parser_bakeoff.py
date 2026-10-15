"""Run one V3 PDF parser through the fixed offline comparison corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

from agentic_rag.evaluation.parser_bakeoff import (
    audit_parser_corpus,
    evaluate_parser_case,
    load_parser_cases,
)
from agentic_rag.ingestion.asset_store import LocalAssetStore
from agentic_rag.ingestion.contracts import RetrievalSchema
from agentic_rag.ingestion.parsers.base import ParserAdapter


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parser",
        choices=("docling", "pymupdf", "pymupdf_layout"),
        default="pymupdf_layout",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval_set/v3_parser_bakeoff_cases.json"),
    )
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifacts-path",
        type=Path,
        default=Path("/home/appuser/.cache/docling/models"),
    )
    parser.add_argument("--image-digest", default="unrecorded")
    parser.add_argument("--render-dpi", type=int, default=144)
    parser.add_argument("--ocr-dpi", type=int, default=150)
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="Disable conditional OCR for parsers that support it.",
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Run only this manifest case; repeat the option to select more than one.",
    )
    return parser.parse_args()


def _schema(adapter: ParserAdapter) -> RetrievalSchema:
    return RetrievalSchema(
        parser_name=adapter.name,
        parser_revision=adapter.revision,
        parser_options=adapter.parser_options,
        chunking_strategy="artifact_structure_v3",
        text_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        text_embedding_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        text_embedding_dimension=384,
    )


def _adapter(args: argparse.Namespace) -> ParserAdapter:
    if args.parser == "docling":
        from agentic_rag.ingestion.parsers.docling_adapter import DoclingParserAdapter

        return DoclingParserAdapter(
            artifacts_path=args.artifacts_path,
            accelerator_device=args.accelerator,
        )

    if args.parser == "pymupdf_layout":
        from agentic_rag.ingestion.parsers.pymupdf_layout_adapter import (
            PyMuPDFLayoutParserAdapter,
        )

        return PyMuPDFLayoutParserAdapter(
            render_dpi=args.render_dpi,
            ocr_dpi=args.ocr_dpi,
            enable_ocr=not args.disable_ocr,
        )

    from agentic_rag.ingestion.parsers.pymupdf_adapter import PyMuPDFParserAdapter

    return PyMuPDFParserAdapter(render_dpi=args.render_dpi)


def _runtime_metadata(*, collect_cuda: bool) -> dict[str, Any]:
    if not collect_cuda:
        return {
            "cuda_metrics_collected": False,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "cuda_total_memory_mib": 0.0,
        }
    import torch

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    total_memory_mib = (
        round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)
        if cuda_available
        else 0.0
    )
    return {
        "cuda_metrics_collected": True,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": device_name,
        "cuda_total_memory_mib": total_memory_mib,
    }


def _reset_accelerator_metrics(*, collect_cuda: bool) -> None:
    if not collect_cuda:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _accelerator_metrics(*, collect_cuda: bool) -> dict[str, float]:
    if not collect_cuda:
        return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    import torch

    if not torch.cuda.is_available():
        return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    try:
        torch.cuda.synchronize()
        return {
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated() / 1024**2,
                1,
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved() / 1024**2,
                1,
            ),
        }
    except RuntimeError:
        return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}


def _copy_review_assets(
    run: Any,
    store: LocalAssetStore,
    output_dir: Path,
    case_id: str,
) -> list[str]:
    review_dir = output_dir / "review" / case_id
    review_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for artifact in run.artifacts:
        if artifact.asset_key is None:
            continue
        filename = f"{artifact.kind.value}-{artifact.artifact_id[:20]}.png"
        target = review_dir / filename
        with store.open(artifact.asset_key) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        paths.append(target.relative_to(output_dir).as_posix())
    return sorted(paths)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# V3 {payload['parser']['name']} parser bakeoff",
        "",
        f"Image digest: `{payload['image_digest']}`",
        "",
        f"Corpus: {payload['corpus']['case_count']} cases across "
        f"{payload['corpus']['unique_file_count']} exact files.",
        "",
        "The report contains counts, stable identities, performance measurements,",
        "and review-image paths. Extracted passage text is intentionally excluded.",
        "",
        "| Case | Status | Expectations | Artifacts | Seconds | Peak MiB | Warnings |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for entry in payload["cases"]:
        if entry["status"] != "success":
            lines.append(
                f"| {entry['case_id']} | error: {entry['error_type']} | - | - | - | - | - |"
            )
            continue
        evaluation = entry["evaluation"]
        summary = entry["run"]
        lines.append(
            "| {case} | success | {met}/{count} | {artifacts} | {seconds:.2f} | "
            "{memory:.1f} | {warnings} |".format(
                case=entry["case_id"],
                met=evaluation["expectations_met"],
                count=evaluation["expectation_count"],
                artifacts=sum(summary["artifact_counts"].values()),
                seconds=summary["elapsed_seconds"],
                memory=summary["peak_memory_mib"],
                warnings=len(summary["warnings"]),
            )
        )
    totals = payload["totals"]
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Successful cases: {totals['successful_cases']}/{totals['case_count']}",
            f"- Expectations met: {totals['expectations_met']}/{totals['expectation_count']} "
            f"({totals['expectation_rate']:.1%})",
            f"- Mean successful-case time: {totals['mean_elapsed_seconds']:.2f} seconds",
            f"- Maximum reported process peak: {totals['maximum_peak_memory_mib']:.1f} MiB",
            f"- Maximum CUDA memory reserved: {totals['maximum_cuda_reserved_mib']:.1f} MiB",
            "",
            "Automated expectations are screening checks. Parser selection still requires",
            "human review of reading order, bounding boxes, tables, formulas, and image crops.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _arguments()
    cases = load_parser_cases(args.manifest)
    if args.case_ids:
        requested = set(args.case_ids)
        available = {case.case_id for case in cases}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"unknown parser case_id: {', '.join(missing)}")
        cases = [case for case in cases if case.case_id in requested]
    roots = {
        "research_articles": args.research_root.resolve(),
        "repository": args.repository_root.resolve(),
    }
    corpus = audit_parser_corpus(cases, roots)
    if not corpus["valid"]:
        print(json.dumps(corpus, indent=2))
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    store = LocalAssetStore(output_dir / "assets")
    adapter = _adapter(args)
    schema = _schema(adapter)
    collect_cuda = adapter.name == "docling" and args.accelerator in {"auto", "cuda"}
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for case in cases:
        source = roots[case.source_group] / case.source_file
        print(f"[{len(results) + 1}/{len(cases)}] {case.case_id}", flush=True)
        _reset_accelerator_metrics(collect_cuda=collect_cuda)
        try:
            run = adapter.parse(
                source,
                document_id=case.sha256,
                selected_pages=case.pages,
                schema=schema,
                asset_store=store,
            )
            evaluation = evaluate_parser_case(case, run)
            results.append(
                {
                    "case_id": case.case_id,
                    "status": "success",
                    "evaluation": evaluation,
                    "run": run.safe_summary(),
                    "accelerator_metrics": _accelerator_metrics(
                        collect_cuda=collect_cuda
                    ),
                    "review_assets": _copy_review_assets(
                        run,
                        store,
                        output_dir,
                        case.case_id,
                    ),
                }
            )
        except Exception as exc:  # continue so the report exposes every failing case
            results.append(
                {
                    "case_id": case.case_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "accelerator_metrics": _accelerator_metrics(
                        collect_cuda=collect_cuda
                    ),
                }
            )

    successful = [entry for entry in results if entry["status"] == "success"]
    met = sum(entry["evaluation"]["expectations_met"] for entry in successful)
    expectation_count = sum(
        entry["evaluation"]["expectation_count"] for entry in successful
    )
    payload = {
        "parser": {
            "name": adapter.name,
            "revision": adapter.revision,
            "capabilities": adapter.capabilities.to_dict(),
        },
        "accelerator_requested": (
            args.accelerator if adapter.name == "docling" else "cpu"
        ),
        "runtime": _runtime_metadata(collect_cuda=collect_cuda),
        "image_digest": args.image_digest,
        "retrieval_schema": {
            "schema_id": schema.schema_id,
            "definition": schema.to_dict(),
        },
        "corpus": corpus,
        "cases": results,
        "totals": {
            "case_count": len(cases),
            "successful_cases": len(successful),
            "expectations_met": met,
            "expectation_count": expectation_count,
            "expectation_rate": met / expectation_count if expectation_count else 0.0,
            "mean_elapsed_seconds": (
                statistics.mean(entry["run"]["elapsed_seconds"] for entry in successful)
                if successful
                else 0.0
            ),
            "maximum_peak_memory_mib": max(
                (entry["run"]["peak_memory_mib"] for entry in successful),
                default=0.0,
            ),
            "maximum_cuda_reserved_mib": max(
                (
                    entry["accelerator_metrics"]["peak_reserved_mib"]
                    for entry in results
                ),
                default=0.0,
            ),
            "wall_seconds": time.perf_counter() - started,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["totals"], indent=2))
    return 0 if len(successful) == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
