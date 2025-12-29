# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from tbp.interactive.data import DataLocator, DataLocatorStep, DataParser
from tbp.plot.registry import attach_args, register

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)


# NOTE: Copied from tbp.monty.frameworks.environments.ycb.py.
# 10 objects that have little similarities in morphology.
DISTINCT_OBJECTS = [
    "mug",
    "bowl",
    "potted_meat_can",
    "spoon",
    "strawberry",
    "mustard_bottle",
    "dice",
    "golf_ball",
    "c_lego_duplo",
    "banana",
]


def create_locators() -> dict[str, DataLocator]:
    locators = {}

    base_loc = DataLocator(
        path=[
            DataLocatorStep.key(name="episode"),
            DataLocatorStep.key(name="lm", value="LM_0"),
            DataLocatorStep.key(name="telemetry", value="hypotheses_updater_telemetry"),
            DataLocatorStep.index(name="step"),
            DataLocatorStep.key(name="obj"),
            DataLocatorStep.key(name="channel", value="patch"),
        ]
    )

    locators["target"] = DataLocator(
        path=[
            DataLocatorStep.key(name="episode"),
            DataLocatorStep.key(name="lm", value="target"),
            DataLocatorStep.key(name="target_stat", value="primary_target_object"),
        ]
    )

    locators["evidence"] = base_loc.extend(
        [
            DataLocatorStep.key(name="telemetry2", value="evidence"),
        ]
    )

    locators["max_slope"] = base_loc.extend(
        [
            DataLocatorStep.key(name="telemetry2", value="hypotheses_updater"),
            DataLocatorStep.key(name="telemetry3", value="max_slope"),
        ]
    )

    locators["added_ids"] = base_loc.extend(
        [
            DataLocatorStep.key(name="telemetry2", value="hypotheses_updater"),
            DataLocatorStep.key(name="telemetry3", value="added_ids"),
        ]
    )

    return locators


def get_bursts(data_parser, locators, episode):
    bursts = []

    obj = data_parser.extract(locators["target"], episode=episode)
    for step in data_parser.query(locators["added_ids"], episode=episode):
        num_added = len(
            data_parser.extract(
                locators["added_ids"],
                episode=episode,
                step=step,
                obj=obj,
            )
        )
        bursts.append(num_added > 0)
    return bursts


def get_max_slopes(data_parser, locators, episode):
    max_slopes = []
    obj = data_parser.extract(locators["target"], episode=episode)
    for step in data_parser.query(locators["evidence"], episode=episode):
        slope = data_parser.extract(
            locators["max_slope"],
            episode=episode,
            step=step,
            obj=obj,
        )
        max_slopes.append(slope)

    return max_slopes


def get_stepwise_target(data_parser, episode):
    mapping_locator = DataLocator(
        path=[
            DataLocatorStep.key(name="episode", value=episode),
            DataLocatorStep.key(name="lm", value="LM_0"),
            DataLocatorStep.key(name="telemetry", value="lm_processed_steps"),
        ]
    )

    stepwise_locator = DataLocator(
        path=[
            DataLocatorStep.key(name="episode", value=episode),
            DataLocatorStep.key(name="lm", value="LM_0"),
            DataLocatorStep.key(name="telemetry", value="stepwise_targets_list"),
        ]
    )

    full_stepwise = np.array(data_parser.extract(stepwise_locator))
    steps_mask = data_parser.extract(mapping_locator)
    mapping = np.flatnonzero(steps_mask)

    return full_stepwise[mapping].tolist()


def get_max_evidence(data_parser, episode):
    max_evidence_locator = DataLocator(
        path=[
            DataLocatorStep.key(name="episode", value=episode),
            DataLocatorStep.key(name="lm", value="LM_0"),
            DataLocatorStep.key(name="telemetry", value="max_evidence"),
        ]
    )

    max_evs = data_parser.extract(max_evidence_locator)

    columns = []
    for max_ev in max_evs:
        columns.append(dict.fromkeys(DISTINCT_OBJECTS, 0) | max_ev)
    return pd.DataFrame(columns)


def create_df(data_parser: DataParser, episode) -> pd.DataFrame:
    locators = create_locators()

    bursts = get_bursts(data_parser, locators, episode)
    max_slopes = get_max_slopes(data_parser, locators, episode)
    stepwise = get_stepwise_target(data_parser, episode)

    data_df = pd.DataFrame(
        {
            "burst": bursts,
            "max_slope": max_slopes,
            "stepwise": stepwise,
        }
    )

    max_evidence_df = get_max_evidence(data_parser, episode)

    return data_df, max_evidence_df


def plot_evidence_and_bursts(
    data_df: pd.DataFrame,
    ev_df: pd.DataFrame,
    objects: list[str],
    obj_colors: dict[str, tuple],
    *,
    figsize: tuple[float, float] = (14, 8),
    burst_color: str = "red",
    burst_linestyle: str = "--",
    burst_linewidth: float = 1.25,
    evidence_linewidth: float = 1.8,
    rect_height_frac: float = 0.02,
    rect_alpha: float = 1.0,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    if not {"burst", "max_slope", "stepwise"}.issubset(data_df.columns):
        raise ValueError(
            "data_df must contain columns: 'burst', 'max_slope', 'stepwise'"
        )

    # Align lengths defensively
    n = min(len(data_df), len(ev_df))
    data_df = data_df.iloc[:n].reset_index(drop=True)
    ev_df = ev_df.iloc[:n].reset_index(drop=True)

    # Ensure all requested object columns exist (missing -> zeros)
    ev_df = ev_df.copy()
    for obj in objects:
        if obj not in ev_df.columns:
            ev_df[obj] = 0.0
    ev_df = ev_df[objects]

    sns.set_theme(style="darkgrid")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )

    x = np.arange(n)

    # -----------------
    # Top: max_slope + burst vlines
    # -----------------
    ax_top.plot(x, data_df["max_slope"].to_numpy(), linewidth=2.0, label="max_slope")

    burst_idx = np.flatnonzero(data_df["burst"].to_numpy().astype(bool))
    for i, bi in enumerate(burst_idx):
        ax_top.axvline(
            bi,
            color=burst_color,
            linestyle=burst_linestyle,
            linewidth=burst_linewidth,
            alpha=1.0,
            clip_on=False,
            label="burst" if i == 0 else None,
            zorder=5,
        )

    ax_top.set_ylabel("Max Slope")
    ax_top.legend(loc="upper right", frameon=True)

    ax_top.set_ylim([0.0, 2.0])

    # -----------------
    # Bottom: evidence lines + stepwise target band (rectangles)
    # -----------------
    for obj in objects:
        ax_bot.plot(
            x,
            ev_df[obj].to_numpy(),
            linewidth=evidence_linewidth,
            label=obj,
            color=obj_colors[obj],
        )

    ax_bot.set_xlabel("Timestep")
    ax_bot.set_ylabel("Max Evidence")
    ax_bot.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        title="object",
    )

    # Draw rectangles above the bottom axis showing stepwise target segments
    rect_y = 1.02  # a bit above axis
    rect_h = rect_height_frac
    stepwise = [t if t in obj_colors else None for t in data_df["stepwise"].tolist()]

    start = 0
    last_x = n - 1

    while start < n:
        target = stepwise[start]
        end = start + 1
        while end < n and stepwise[end] == target:
            end += 1

        if target is not None:
            # Cap rectangle so it never extends past the last timestep
            rect_end = min(end, last_x)
            width = rect_end - start

            if width > 0:
                ax_bot.add_patch(
                    Rectangle(
                        (start, rect_y),
                        width=width,
                        height=rect_h,
                        transform=ax_bot.get_xaxis_transform(),  # x in data, y in axes
                        facecolor=obj_colors[target],
                        edgecolor="none",
                        alpha=rect_alpha,
                        clip_on=False,
                    )
                )

        start = end

    ax_bot.margins(x=0)
    ax_bot.set_xlim(0, n - 1 if n > 0 else 1)

    fig.tight_layout()
    fig.subplots_adjust(right=0.82, hspace=0.08, top=0.92)

    return fig, (ax_top, ax_bot)


@register(
    "multi_objects_evidence_over_time",
    description="Maximum evidence score for each object over time",
)
def main(experiment_log_dir: str) -> int:
    """Plot evidence scores for each object over time.

    This function visualizes the evidence scores for each object. The plot is produced
    over a sequence of episodes, and overlays colored rectangles highlighting when a
    particular target object is active.

    Args:
        experiment_log_dir: Path to the experiment directory containing the detailed
            stats file.

    Returns:
        Exit code.
    """
    if not Path(experiment_log_dir).exists():
        logger.error(f"Experiment path not found: {experiment_log_dir}")
        return 1

    # seaborn darkgrid style
    sns.set_theme(style="darkgrid")
    cmap = plt.cm.tab10
    num_colors = len(DISTINCT_OBJECTS)
    ycb_colors = {obj: cmap(i / num_colors) for i, obj in enumerate(DISTINCT_OBJECTS)}

    # load detailed stats
    data_parser = DataParser(Path(experiment_log_dir))
    data_df, ev_df = create_df(data_parser, episode="9")

    _, _ = plot_evidence_and_bursts(
        data_df=data_df,
        ev_df=ev_df,
        objects=DISTINCT_OBJECTS,
        obj_colors=ycb_colors,
        figsize=(14, 8),
    )

    # Show plot
    plt.show()

    return 0


@attach_args("multi_objects_evidence_over_time")
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
