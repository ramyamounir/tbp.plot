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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tbp.interactive.data import (
    DataLocator,
    DataLocatorStep,
    DataParser,
)
from tbp.plot.registry import attach_args, register

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)


POSE_ERROR_LOC = DataLocator(
    path=[
        DataLocatorStep.key(name="episode", value="0"),
        DataLocatorStep.key(name="lm", value="LM_0"),
        DataLocatorStep.key(name="telemetry", value="target_object_pose_error"),
    ]
)

TH_LIMIT_LOC = DataLocator(
    path=[
        DataLocatorStep.key(name="episode", value="0"),
        DataLocatorStep.key(name="lm", value="LM_0"),
        DataLocatorStep.key(name="telemetry", value="target_object_theoretical_limit"),
    ]
)


MLH_LOC = DataLocator(
    path=[
        DataLocatorStep.key(name="episode", value="0"),
        DataLocatorStep.key(name="lm", value="LM_0"),
        DataLocatorStep.key(name="telemetry", value="current_mlh"),
        DataLocatorStep.index(name="step"),
        DataLocatorStep.key(name="mlh_id", value="graph_id"),
    ]
)

TARGET_LOC = DataLocator(
    path=[
        DataLocatorStep.key(name="episode", value="0"),
        DataLocatorStep.key(name="lm", value="target"),
        DataLocatorStep.key(name="target_stat", value="primary_target_object"),
    ]
)

AVG_SLOPES_LOC = DataLocator(
    path=[
        DataLocatorStep.key(name="episode", value="0"),
        DataLocatorStep.key(name="lm", value="LM_0"),
        DataLocatorStep.key(name="telemetry", value="hypotheses_updater_telemetry"),
        DataLocatorStep.index(name="step"),
        DataLocatorStep.key(name="object", value="mug"),
        DataLocatorStep.key(name="channel", value="patch"),
        DataLocatorStep.key(name="telemetry2", value="hypotheses_updater"),
        DataLocatorStep.key(name="telemetry3", value="avg_slopes"),
    ]
)

ADDED_IDS_LOC = DataLocator(
    path=[
        DataLocatorStep.key(name="episode", value="0"),
        DataLocatorStep.key(name="lm", value="LM_0"),
        DataLocatorStep.key(name="telemetry", value="hypotheses_updater_telemetry"),
        DataLocatorStep.index(name="step"),
        DataLocatorStep.key(name="obj"),
        DataLocatorStep.key(name="channel", value="patch"),
        DataLocatorStep.key(name="telemetry2", value="hypotheses_updater"),
        DataLocatorStep.key(name="telemetry3", value="added_ids"),
    ]
)


EVIDENCE_LOC = DataLocator(
    path=[
        DataLocatorStep.key(name="episode", value="0"),
        DataLocatorStep.key(name="lm", value="LM_0"),
        DataLocatorStep.key(name="telemetry", value="hypotheses_updater_telemetry"),
        DataLocatorStep.index(name="step"),
        DataLocatorStep.key(name="obj"),
        DataLocatorStep.key(name="channel", value="patch"),
        DataLocatorStep.key(name="telemetry2", value="evidence"),
    ]
)


def get_hyp_space_sizes(parser):
    hyp_space_sizes = []
    for s in parser.query(EVIDENCE_LOC):
        obj_space = 0
        for o in parser.query(EVIDENCE_LOC, step=s):
            obj_space += len(parser.extract(EVIDENCE_LOC, step=s, obj=o))
        hyp_space_sizes.append(obj_space)
    return hyp_space_sizes


def get_added_bool(parser):
    added = []
    for s in parser.query(ADDED_IDS_LOC):
        obj_added = 0
        for o in parser.query(ADDED_IDS_LOC, step=s):
            obj_added += len(parser.extract(ADDED_IDS_LOC, step=s, obj=o))
        added.append(obj_added > 0)
    return added


@register(
    "intelligent_sampling_debugger",
    description="Debugger for intelligent sampling",
)
def main(experiment_log_dir: str) -> int:
    if not Path(experiment_log_dir).exists():
        logger.error(f"Experiment path not found: {experiment_log_dir}")
        return 1

    # seaborn darkgrid style
    sns.set_theme(style="darkgrid")

    parser = DataParser(experiment_log_dir)

    correct_mlh = [
        (parser.extract(MLH_LOC, step=s) == parser.extract(TARGET_LOC))
        for s in parser.query(MLH_LOC)
    ]
    added_hyp = get_added_bool(parser)
    pose_errors = parser.extract(POSE_ERROR_LOC)
    th_limit = parser.extract(TH_LIMIT_LOC)
    avg_slopes = [
        parser.extract(AVG_SLOPES_LOC, step=s) for s in parser.query(AVG_SLOPES_LOC)
    ]
    hyp_space_sizes = get_hyp_space_sizes(parser)

    assert (
        len(correct_mlh)
        == len(added_hyp)
        == len(pose_errors)
        == len(th_limit)
        == len(avg_slopes)
        == len(hyp_space_sizes)
    )

    correct_mlh = np.asarray(correct_mlh, dtype=bool)
    added_hyp = np.asarray(added_hyp, dtype=bool)
    pose_errors = np.asarray(pose_errors, dtype=float)
    th_limit = np.asarray(th_limit, dtype=float)
    slopes = np.asarray(avg_slopes, dtype=float)
    hyp_space_sizes = np.asarray(hyp_space_sizes, dtype=float)

    pose_deg = np.rad2deg(pose_errors)
    th_deg = np.rad2deg(th_limit)

    x = np.arange(len(pose_deg))

    _, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # Top: theoretical limit (line, black) + pose error (scatter only when correct)
    ax_top.plot(x, th_deg, color="black", linewidth=1.5, label="Theoretical limit")
    mask = correct_mlh  # or: mask = np.isfinite(pose_deg) & correct_mlh
    if np.any(mask):
        ax_top.scatter(x[mask], pose_deg[mask], s=16, label="Pose error (correct)")
    ax_top.set_ylim(0, 50)
    ax_top.set_ylabel("Degrees")
    ax_top.legend(loc="best")

    # Bottom: twin y axes for slope (left) and hyp space size (right)
    ax_left = ax_bot
    ax_right = ax_bot.twinx()

    # If early slope values are NaN, start from first valid index
    valid_idx = np.where(~np.isnan(slopes))[0]
    start_idx = int(valid_idx[0]) if valid_idx.size > 0 else 0

    if start_idx < len(slopes):
        ax_left.plot(x[start_idx:], slopes[start_idx:], label="Average slope")
    else:
        # All slopes are NaN, plot nothing on left
        pass

    ax_right.plot(x, hyp_space_sizes, linestyle="--", label="Hypothesis space size")

    ax_left.set_ylim(-1, 2)
    ax_right.set_ylim(0, 40000)

    # Vertical markers for when hypotheses were added (bottom plot only)
    add_idx = np.flatnonzero(added_hyp)
    if add_idx.size > 0:
        # Use left axis limits so lines span the full height of the bottom plot
        ymin, ymax = ax_left.get_ylim()
        ax_left.vlines(
            add_idx,
            ymin,
            ymax,
            colors="red",
            linestyles="--",
            alpha=0.25,
            linewidth=1.0,
            zorder=0,
            label="Added hypotheses",
        )

    ax_left.set_xlabel("Step")
    ax_left.set_ylabel("Slope")
    ax_right.set_ylabel("Hyp space size")

    # Combine legends from both axes
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    if lines_left or lines_right:
        ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="best")

    plt.tight_layout()
    plt.title(
        # f"No Sampling\nTotal: {np.sum(hyp_space_sizes)}"
        f"Burst Sampling\nTotal: {np.sum(hyp_space_sizes)}"
        # f"Constant Sampling: 0.0\nTotal: {np.sum(hyp_space_sizes)}"
        # f"Intelligent Sampling: 0.1 + (-1 * 0.05 * slope)\nTotal: {np.sum(hyp_space_sizes)}"
    )
    plt.show()

    return 0


@attach_args("intelligent_sampling_debugger")
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
