# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
from pathlib import Path

import numpy as np

from tbp.interactive.data import DataLocator, DataLocatorStep, ReadyDataParser
from tbp.plot.plots.stats import deserialize_json_chunks
from tbp.plot.registry import attach_args, register

EPISODE_NUM = 4


class Debugger:
    def __init__(self, baseline_parser, proposed_parser):
        self.baseline_parser = baseline_parser
        self.proposed_parser = proposed_parser

        self.target_locator = DataLocator(
            path=[
                DataLocatorStep.key(name="episode", value=str(EPISODE_NUM)),
                DataLocatorStep.key(name="lm", value="target"),
                DataLocatorStep.key(name="telemetry", value="primary_target_object"),
            ],
        )

        self.evidence_locator = DataLocator(
            path=[
                DataLocatorStep.key(name="episode", value=str(EPISODE_NUM)),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(
                    name="obj", value=self.baseline_parser.extract(self.target_locator)
                ),
                DataLocatorStep.key(name="channel", value="patch"),
                DataLocatorStep.key(name="metric", value="evidence"),
            ],
        )

        self.age_locator = DataLocator(
            path=[
                DataLocatorStep.key(name="episode", value=str(EPISODE_NUM)),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(
                    name="obj", value=self.baseline_parser.extract(self.target_locator)
                ),
                DataLocatorStep.key(name="channel", value="patch"),
                DataLocatorStep.key(name="metric", value="hypotheses_updater"),
                DataLocatorStep.key(name="updater_metric", value="ages"),
            ],
        )

    def check_hyp_space_size(self):
        baseline_steps = len(self.baseline_parser.query(self.evidence_locator))
        proposed_steps = len(self.proposed_parser.query(self.evidence_locator))

        for step in range(min(baseline_steps, proposed_steps)):
            baseline_evidence = self.baseline_parser.extract(
                self.evidence_locator, step=step
            )
            baseline_normalized_evidence = (
                np.array(baseline_evidence) / float(step + 1)
            ).tolist()

            proposed_evidence = self.proposed_parser.extract(
                self.evidence_locator, step=step
            )

            ixs = np.where(
                ~np.isclose(
                    baseline_normalized_evidence, proposed_evidence, rtol=1e-9, atol=0.0
                )
            )[0]
            assert len(baseline_evidence) == len(proposed_evidence)
            assert len(ixs) == 0


@register(
    "debugger",
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
    baseline_stats = deserialize_json_chunks(
        Path(experiment_log_dir) / "detailed_run_stats.json",
        episodes=[EPISODE_NUM],
    )
    baseline_parser = ReadyDataParser(baseline_stats)

    proposed_stats = deserialize_json_chunks(
        Path("/home/ramy/tbp/results/monty/projects/evidence_eval_runs/logs/proposed")
        / "detailed_run_stats.json",
        episodes=[EPISODE_NUM],
    )
    proposed_parser = ReadyDataParser(proposed_stats)

    debugger = Debugger(baseline_parser, proposed_parser)
    debugger.check_hyp_space_size()

    return 0


@attach_args("debugger")
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
