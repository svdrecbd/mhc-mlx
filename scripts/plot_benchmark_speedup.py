import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def _load_summary(path: str) -> dict[tuple[str, str], list[tuple[int, float]]]:
    data: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["mode"], row["benchmark"])
            data[key].append((int(row["C"]), float(row["speedup_median"])))
    for key in data:
        data[key].sort()
    return data


def _all_C_values(data: dict[tuple[str, str], list[tuple[int, float]]]) -> list[int]:
    values = sorted({C for series in data.values() for C, _ in series})
    return values


def plot_speedup_by_C(summary_path: str, out_png: str, out_svg: str | None) -> None:
    data = _load_summary(summary_path)

    modes = ["throughput", "latency"]
    benchmarks = ["sinkhorn", "fused", "layer"]
    colors = {
        "sinkhorn": "#1f77b4",
        "fused": "#ff7f0e",
        "layer": "#2ca02c",
    }

    Cs = _all_C_values(data)
    all_speedups = [speed for series in data.values() for _, speed in series]
    y_min = min(all_speedups) * 0.9 if all_speedups else 0.0
    y_max = max(all_speedups) * 1.1 if all_speedups else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, mode in zip(axes, modes):
        for bench in benchmarks:
            series = data.get((mode, bench))
            if not series:
                continue
            xs, ys = zip(*series)
            ax.plot(xs, ys, marker="o", color=colors.get(bench), label=bench)

        ax.axhline(1.0, color="0.5", linewidth=1, linestyle="--")
        ax.set_title(f"{mode} (median speedup)")
        ax.set_xlabel("C")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Cs)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.set_ylim(bottom=max(0.0, y_min), top=y_max)

    axes[0].set_ylabel("Speedup (reference / metal)")
    axes[1].legend(title="benchmark", loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    if out_svg:
        fig.savefig(out_svg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark speedups by C.")
    parser.add_argument("--summary", type=str, default="summary_by_C.csv")
    parser.add_argument("--out", type=str, default="benchmark_speedup_by_C.png")
    parser.add_argument("--out-svg", type=str, default="benchmark_speedup_by_C.svg")
    args = parser.parse_args()

    plot_speedup_by_C(args.summary, args.out, args.out_svg)
    print(f"Wrote {args.out}")
    if args.out_svg:
        print(f"Wrote {args.out_svg}")


if __name__ == "__main__":
    main()
