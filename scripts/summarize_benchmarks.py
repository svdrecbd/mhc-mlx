import argparse
import csv
import json
import math
from collections import defaultdict


KEY_FIELDS = [
    "mode",
    "compiled",
    "B",
    "n",
    "C",
    "dtype",
    "threads_per_group",
    "sinkhorn_iters",
    "eps",
    "seed",
]


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _summarize(values: list[float]) -> dict:
    values = sorted(values)
    return {
        "count": len(values),
        "speedup_min": values[0] if values else float("nan"),
        "speedup_p10": _percentile(values, 10.0),
        "speedup_median": _percentile(values, 50.0),
        "speedup_p90": _percentile(values, 90.0),
        "speedup_max": values[-1] if values else float("nan"),
    }


def _get_time(record: dict) -> float | None:
    if "time_s_median" in record:
        return float(record["time_s_median"])
    if "time_s" in record:
        return float(record["time_s"])
    return None


def _parse_results(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _pair_speedups(records: list[dict]) -> list[dict]:
    cases: dict[tuple[str, tuple], dict] = defaultdict(dict)

    for record in records:
        bench = record.get("benchmark")
        if not bench or bench == "correctness":
            continue
        if bench.endswith("_ref"):
            base = bench[:-4]
            kind = "ref"
        elif bench.endswith("_metal"):
            base = bench[:-6]
            kind = "metal"
        else:
            continue

        time_s = _get_time(record)
        if time_s is None:
            continue

        key = tuple(record.get(field) for field in KEY_FIELDS)
        cases[(base, key)][kind] = time_s

    entries = []
    for (base, key), times in cases.items():
        if "ref" not in times or "metal" not in times:
            continue
        ref = times["ref"]
        metal = times["metal"]
        if metal <= 0:
            continue
        speedup = ref / metal
        key_map = dict(zip(KEY_FIELDS, key))
        entries.append(
            {
                "benchmark": base,
                "mode": key_map.get("mode"),
                "dtype": key_map.get("dtype"),
                "n": key_map.get("n"),
                "C": key_map.get("C"),
                "speedup": speedup,
            }
        )
    return entries


def _write_summary(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_by(entries: list[dict], fields: list[str]) -> list[dict]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for entry in entries:
        key = tuple(entry[field] for field in fields)
        buckets[key].append(entry["speedup"])

    rows = []
    for key, speeds in buckets.items():
        summary = _summarize(speeds)
        row = {field: value for field, value in zip(fields, key)}
        row.update(summary)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MHCLayer benchmark results.")
    parser.add_argument("--in", dest="in_paths", action="append", default=[])
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    paths = args.in_paths if args.in_paths else ["results.jsonl"]
    in_paths = []
    for path in paths:
        in_paths.extend([part.strip() for part in path.split(",") if part.strip()])

    records = []
    for path in in_paths:
        records.extend(_parse_results(path))
    entries = _pair_speedups(records)

    overall = _summarize_by(entries, ["benchmark", "mode"])
    by_dtype = _summarize_by(entries, ["benchmark", "mode", "dtype"])
    by_n = _summarize_by(entries, ["benchmark", "mode", "n"])
    by_C = _summarize_by(entries, ["benchmark", "mode", "C"])

    _write_summary(
        f"{args.out_dir}/summary_overall.csv",
        overall,
        ["benchmark", "mode", "count", "speedup_min", "speedup_p10", "speedup_median", "speedup_p90", "speedup_max"],
    )
    _write_summary(
        f"{args.out_dir}/summary_by_dtype.csv",
        by_dtype,
        ["benchmark", "mode", "dtype", "count", "speedup_min", "speedup_p10", "speedup_median", "speedup_p90", "speedup_max"],
    )
    _write_summary(
        f"{args.out_dir}/summary_by_n.csv",
        by_n,
        ["benchmark", "mode", "n", "count", "speedup_min", "speedup_p10", "speedup_median", "speedup_p90", "speedup_max"],
    )
    _write_summary(
        f"{args.out_dir}/summary_by_C.csv",
        by_C,
        ["benchmark", "mode", "C", "count", "speedup_min", "speedup_p10", "speedup_median", "speedup_p90", "speedup_max"],
    )

    print("Wrote summary_overall.csv, summary_by_dtype.csv, summary_by_n.csv, summary_by_C.csv")


if __name__ == "__main__":
    main()
