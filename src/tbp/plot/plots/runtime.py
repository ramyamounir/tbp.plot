# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from tbp.interactive.data import DataLocator, DataLocatorStep, ReadyDataParser
from tbp.plot.registry import attach_args, register

EPISODE_NUM = 2


class RunTime:
    def __init__(
        self,
        no_resampling_x_percent_parser,
        no_resampling_all_parser,
        resampling_all_parser,
    ):
        self.parsers = {
            "no_resampling_x_percent": no_resampling_x_percent_parser,
            "no_resampling_all": no_resampling_all_parser,
            "resampling_all": resampling_all_parser,
        }

        self.target_locator = DataLocator(
            path=[
                DataLocatorStep.key(name="episode", value=str(EPISODE_NUM)),
                DataLocatorStep.key(name="lm", value="target"),
                DataLocatorStep.key(name="telemetry", value="primary_target_object"),
            ],
        )

        self.time_locator = DataLocator(
            path=[
                DataLocatorStep.key(name="episode", value=str(EPISODE_NUM)),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="time"),
            ],
        )

        self.hyp_update_locator = DataLocator(
            path=[
                DataLocatorStep.key(name="episode", value=str(EPISODE_NUM)),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(name="obj"),
                DataLocatorStep.key(name="channel", value="patch"),
                DataLocatorStep.key(name="telemetry2", value="hypotheses_updater"),
                DataLocatorStep.key(name="result", value="num_hyp_updated"),
            ],
        )

        data_df = self._create_df()

        fig, ax, agg = self.plot_run_scatter(
            data_df,
            runs=[
                "no_resampling_x_percent",
                "no_resampling_all",
                "resampling_all",
            ],
        )
        plt.show()

        fig, (ax_top, ax_bottom) = self.plot_hyp_updates(
            data_df, "no_resampling_x_percent"
        )
        plt.show()

        fig, (ax_top, ax_bottom) = self.plot_hyp_updates(data_df, "no_resampling_all")
        plt.show()

        fig, (ax_top, ax_bottom) = self.plot_hyp_updates(data_df, "resampling_all")
        plt.show()

    def _create_df(self):
        rows = []
        for parser_name, parser in self.parsers.items():
            target = parser.extract(self.target_locator)
            time = parser.extract(self.time_locator)
            for step in parser.query(self.hyp_update_locator)[1:]:
                objects = parser.query(self.hyp_update_locator, step=step)
                for obj in objects:
                    num_hyp_updated = parser.extract(
                        self.hyp_update_locator, step=step, obj=obj
                    )
                    rows.append(
                        {
                            "run": str(parser_name),
                            "step": int(step),
                            "obj": str(obj),
                            "num_hyp_updated": int(num_hyp_updated),
                            "is_target": target == obj,
                            "episode_time": time[step] - time[step - 1],
                        }
                    )

        data_df = (
            pd.DataFrame(rows)
            .sort_values(["run", "step", "obj"])
            .reset_index(drop=True)
        )
        return data_df

    def plot_hyp_updates(self, df: pd.DataFrame, run: str, *, figsize=(11, 7)):
        sns.set_theme(style="whitegrid", context="talk")

        data = df[df["run"] == run].copy()
        if data.empty:
            raise ValueError(f"No rows found for run={run!r}")

        # Per-step totals for the right y-axis (top panel)
        totals = (
            data.groupby("step", as_index=False)["num_hyp_updated"]
            .sum()
            .rename(columns={"num_hyp_updated": "total_updated"})
        )

        # One time per step for the left y-axis (top panel)
        time_per_step = (
            data[["step", "episode_time"]]
            .dropna(subset=["episode_time"])
            .drop_duplicates(subset=["step"])
            .sort_values("step")
        )

        targets = data[data["is_target"]]
        non_targets = data[~data["is_target"]]

        # Figure and layout
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        fig.set_constrained_layout_pads(
            w_pad=0.08, h_pad=0.10, wspace=0.08, hspace=0.12
        )

        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 3])
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)

        # Twin y for the top panel
        ax_top_right = ax_top.twinx()

        # Left y: episode time
        if not time_per_step.empty:
            ax_top.plot(
                time_per_step["step"],
                time_per_step["episode_time"],
                linewidth=2.0,
                marker="o",
                markersize=5,
                label="Episode time (s)",
            )
            ax_top.set_ylabel("Episode time (s)", labelpad=8)

        # Right y: total updated across objects
        ax_top_right.plot(
            totals["step"],
            totals["total_updated"],
            linewidth=2.0,
            linestyle="--",
            label="Total updated",
        )
        ax_top_right.set_ylabel("Total updated", labelpad=10)

        # Title, grid, limits
        step_min, step_max = int(data["step"].min()), int(data["step"].max())
        ax_top.set_title(f"Hypotheses updates and timing — {run}", pad=10)
        ax_top.set_xlim(step_min, step_max)
        ax_top.grid(True, axis="both", alpha=0.25)

        # Top legend INSIDE
        h_left, l_left = ax_top.get_legend_handles_labels()
        h_right, l_right = ax_top_right.get_legend_handles_labels()
        ax_top.legend(
            h_left + h_right,
            l_left + l_right,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            bbox_transform=ax_top.transAxes,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            handlelength=2.0,
            labelspacing=0.4,
            borderaxespad=0.3,
        )
        ax_top.margins(y=0.06)

        # Bottom panel: per-object lines
        if not non_targets.empty:
            for obj, d in non_targets.groupby("obj"):
                d = d.sort_values("step")
                ax_bottom.plot(
                    d["step"],
                    d["num_hyp_updated"],
                    linewidth=1.2,
                    alpha=0.7,
                    label=str(obj),
                )

        if not targets.empty:
            for obj, d in targets.groupby("obj"):
                d = d.sort_values("step")
                ax_bottom.plot(
                    d["step"],
                    d["num_hyp_updated"],
                    linewidth=3.0,
                    marker="o",
                    markersize=6,
                    label=f"{obj} (target)",
                )

        ax_bottom.set_xlabel("Step", labelpad=8)
        ax_bottom.set_ylabel("Updated hypotheses per object", labelpad=8)
        ax_bottom.grid(True, axis="both", alpha=0.25)

        # Bottom legend INSIDE
        handles, labels = ax_bottom.get_legend_handles_labels()
        if len(labels) > 10 and not targets.empty:
            keep = [
                (h, l) for h, l in zip(handles, labels, strict=False) if "(target)" in l
            ]
            if keep:
                handles, labels = zip(*keep, strict=False)
        ax_bottom.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            bbox_transform=ax_bottom.transAxes,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            ncols=2 if len(labels) > 6 else 1,
            handlelength=2.0,
            labelspacing=0.4,
            borderaxespad=0.3,
        )
        ax_bottom.margins(y=0.08)

        # Shared x: show ticks at 5, 10, 15, ...
        # Use integer ticks, multiples of 5
        major5 = mticker.MultipleLocator(base=5)
        ax_bottom.xaxis.set_major_locator(major5)
        ax_top.xaxis.set_major_locator(major5)
        ax_bottom.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        ax_top.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))

        # Hide top x tick labels to reduce clutter
        plt.setp(ax_top.get_xticklabels(), visible=False)

        return fig, (ax_top, ax_bottom)

    def plot_run_scatter(
        self, df: pd.DataFrame, runs: list[str] | None = None, *, figsize=(7, 6)
    ):
        sns.set_theme(style="whitegrid", context="talk")

        data = df.copy()
        if runs is not None:
            data = data[data["run"].isin(runs)]
        if data.empty:
            raise ValueError("No rows to plot after filtering runs.")

        # Sum of updates is simple: sum across all rows in the run
        total_updates = data.groupby("run")["num_hyp_updated"].mean()

        # Total time needs dedup per step, since each step appears once per object
        step_time = (
            data[["run", "step", "episode_time"]]
            .dropna(subset=["episode_time"])
            .drop_duplicates(subset=["run", "step"])
            .groupby("run")["episode_time"]
            .mean()
            .rename("total_time_s")
        )

        agg = (
            pd.concat([total_updates.rename("total_updates"), step_time], axis=1)
            .reset_index()
            .sort_values("run")
        )

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(
            data=agg,
            x="total_updates",
            y="total_time_s",
            hue="run",
            s=140,
            ax=ax,
            legend=True,
        )

        # Labels and grid
        ax.set_xlabel("Average step updated hypotheses")
        ax.set_ylabel("Average step time (s)")
        ax.grid(True, alpha=0.3)

        # Nice integer ticks on x if values are integer-like
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))

        # Annotate each point with the run label (slightly offset)
        for _, row in agg.iterrows():
            ax.annotate(
                row["run"],
                xy=(row["total_updates"], row["total_time_s"]),
                xytext=(6, 6),
                textcoords="offset points",
            )

        # Keep legend inside but tidy
        ax.legend(title="Run", loc="upper left", frameon=True, framealpha=0.9)

        return fig, ax, agg


@register(
    "runtime",
    description="Detailed debugger",
)
def main(experiment_log_dir: str) -> int:
    """Interactive visualization for inspecting the terminal conditions.

    Args:
        experiment_log_dir: Path to the experiment directory containing the detailed
            stats file.

    Returns:
        Exit code.
    """
    base_path = Path("/home/ramy/tbp/results/monty/projects/evidence_eval_runs/logs")

    no_resampling_x_percent_path = (
        base_path
        / "no_resampling_x_percent"
        / "detailed_run_stats"
        / "episode_000002.json"
    )
    with open(no_resampling_x_percent_path, "r") as f:
        no_resampling_x_percent = json.load(f)
    no_resampling_x_percent_parser = ReadyDataParser(no_resampling_x_percent)

    no_resampling_all_path = (
        base_path / "no_resampling_all" / "detailed_run_stats" / "episode_000002.json"
    )
    with open(no_resampling_all_path, "r") as f:
        no_resampling_all = json.load(f)
    no_resampling_all_parser = ReadyDataParser(no_resampling_all)

    resampling_all_path = (
        base_path / "resampling_all" / "detailed_run_stats" / "episode_000002.json"
    )
    with open(resampling_all_path, "r") as f:
        resampling_all = json.load(f)
    resampling_all_parser = ReadyDataParser(resampling_all)

    runtime_debugger = RunTime(
        no_resampling_x_percent_parser, no_resampling_all_parser, resampling_all_parser
    )
    # runtime_debugger.check_hyp_space_size()

    return 0


@attach_args("runtime")
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
