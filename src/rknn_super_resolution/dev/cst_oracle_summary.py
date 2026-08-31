"""Aggregate matched-seed CST Stage A0 oracle results and apply the frozen gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rknn_super_resolution.dev.cst_oracle import CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=("coarse", "half", "full", "attention", "retrieved"),
        default="coarse",
    )
    parser.add_argument("--coarse_psnr_gate", type=float, default=0.10)
    return parser.parse_args()


def _load_run(run_dir: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for condition in CONDITIONS:
        path = run_dir / condition / "result.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or payload.get("condition") != condition:
            raise ValueError(f"invalid oracle result: {path}")
        results[condition] = payload
    return results


def _describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(
    run_dirs: list[Path],
    *,
    candidate: str = "coarse",
    coarse_psnr_gate: float,
) -> dict:
    rows: list[dict] = []
    for run_dir in run_dirs:
        results = _load_run(run_dir)
        required = {"spatial", "center", "full", candidate}
        missing = required - results.keys()
        if missing:
            raise FileNotFoundError(f"{run_dir} missing conditions: {sorted(missing)}")
        seed = int(results["spatial"]["seed"])
        scores = {
            condition: float(results[condition]["evaluations"][condition]["psnr"])
            for condition in required
        }
        hf_scores = {
            condition: float(results[condition]["evaluations"][condition]["hf_psnr"])
            for condition in required
        }
        rows.append(
            {
                "seed": seed,
                "psnr": scores,
                "hf_psnr": hf_scores,
                "delta_psnr": {
                    "center_minus_spatial": scores["center"] - scores["spatial"],
                    "full_minus_spatial": scores["full"] - scores["spatial"],
                    f"{candidate}_minus_spatial": scores[candidate] - scores["spatial"],
                    f"{candidate}_minus_center": scores[candidate] - scores["center"],
                },
                f"{candidate}_history_dependency": results[candidate]["history_dependency"],
                "full_history_dependency": results["full"]["history_dependency"],
            }
        )
    rows.sort(key=lambda item: item["seed"])
    deltas = {
        key: _describe([float(row["delta_psnr"][key]) for row in rows])
        for key in (
            "center_minus_spatial",
            "full_minus_spatial",
            f"{candidate}_minus_spatial",
            f"{candidate}_minus_center",
        )
    }
    candidate_values = [row["delta_psnr"][f"{candidate}_minus_spatial"] for row in rows]
    passed = deltas[f"{candidate}_minus_spatial"]["mean"] >= coarse_psnr_gate and min(
        candidate_values
    ) > 0.0
    return {
        "schema_version": 1,
        "experiment": "CST/PST Stage A0 paired oracle",
        "candidate": candidate,
        "seeds": [row["seed"] for row in rows],
        "coarse_psnr_gate": coarse_psnr_gate,
        "gate_status": "PASS" if passed else "FAIL",
        "decision": (
            f"continue_to_{candidate}_temporal_training"
            if passed
            else f"terminate_{candidate}_transport"
        ),
        "paired_rows": rows,
        "paired_delta_summary": deltas,
    }


def _markdown(summary: dict) -> str:
    candidate = summary["candidate"]
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        "- Verification Status: ANALYZED",
        "- Version Label: cst_a0_summary_v1",
        "",
        "## CST Stage A0 Oracle Result",
        "",
        f"- Gate: **{summary['gate_status']}**",
        f"- Decision: `{summary['decision']}`",
        f"- Candidate PSNR threshold: {summary['coarse_psnr_gate']:.3f} dB",
        "",
    ]
    if candidate == "full":
        lines.extend(
            (
                "| Seed | Spatial | Center Δ | Full Δ | Full−Center |",
                "| ---: | ---: | ---: | ---: | ---: |",
            )
        )
    else:
        lines.extend(
            (
                f"| Seed | Spatial | Center Δ | Full Δ | {candidate.title()} Δ | "
                f"{candidate.title()}−Center |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
    for row in summary["paired_rows"]:
        delta = row["delta_psnr"]
        if candidate == "full":
            lines.append(
                f"| {row['seed']} | {row['psnr']['spatial']:.4f} | "
                f"{delta['center_minus_spatial']:+.4f} | "
                f"{delta['full_minus_spatial']:+.4f} | "
                f"{delta['full_minus_center']:+.4f} |"
            )
        else:
            lines.append(
                f"| {row['seed']} | {row['psnr']['spatial']:.4f} | "
                f"{delta['center_minus_spatial']:+.4f} | "
                f"{delta['full_minus_spatial']:+.4f} | "
                f"{delta[f'{candidate}_minus_spatial']:+.4f} | "
                f"{delta[f'{candidate}_minus_center']:+.4f} |"
            )
    lines.extend(("", "### Paired delta summary", ""))
    for name, values in summary["paired_delta_summary"].items():
        lines.append(f"- `{name}`: {values['mean']:+.4f} ± {values['std']:.4f} dB")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.coarse_psnr_gate <= 0:
        raise ValueError("coarse_psnr_gate must be positive")
    summary = summarize(
        args.run_dirs,
        candidate=args.candidate,
        coarse_psnr_gate=args.coarse_psnr_gate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    markdown_path = args.output_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
