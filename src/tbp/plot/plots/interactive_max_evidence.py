# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import logging
from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Series
from pubsub.core import Publisher
from scipy.spatial.transform import Rotation
from vedo import Button, Image, Plotter, Slider2D

from tbp.interactive.data import (
    DataLocator,
    DataLocatorStep,
    DataParser,
)
from tbp.interactive.topics import TopicMessage, TopicSpec
from tbp.interactive.widget_updaters import WidgetUpdater
from tbp.interactive.widgets import (
    VtkDebounceScheduler,
    Widget,
    extract_slider_state,
    set_button_state,
    set_slider_state,
)
from tbp.plot.registry import attach_args, register

logger = logging.getLogger(__name__)


HUE_PALETTE = {
    "Added": "#66c2a5",
    "Removed": "#fc8d62",
    "Maintained": "#8da0cb",
    "Evidence": "#1f77b4",
    "Slope": "#ff7f0e",
}


class StepSliderWidgetOps:
    """WidgetOps implementation for a Step slider.

    This class listens for the current episode selection and adjusts the step
    slider range to match the number of steps in that episode. It publishes
    changes as `TopicMessage` items under the "step_number" topic.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        data_parser: A parser that extracts or queries information from the
            JSON log file.
        updaters: A list with a single `WidgetUpdater` that reacts to the
            `"episode_number"` topic and calls `update_slider_range`.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
        _locators: Data accessors keyed by name that instruct the `DataParser`
            how to retrieve the required information.
    """

    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser

        self._add_kwargs = {
            "xmin": 0,
            "xmax": 10,
            "value": 0,
            "pos": [(0.1, 0.1), (0.9, 0.1)],
            "title": "Step",
        }
        self._locators = self.create_locators()

    def create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this widget.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}
        locators["step"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode", value="0"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="evidences"),
                DataLocatorStep.index(name="step"),
            ]
        )
        return locators

    def add(self, callback: Callable) -> Slider2D:
        """Create the slider widget.

        Args:
            callback: Function called with `(widget, event)` when the UI changes.

        Returns:
            The created `Slider2D` widget.
        """
        kwargs = deepcopy(self._add_kwargs)
        locator = self._locators["step"]
        kwargs.update({"xmax": len(self.data_parser.query(locator)) - 1})
        widget = self.plotter.add_slider(callback, **kwargs)
        self.plotter.render()
        return widget

    def extract_state(self, widget: Slider2D) -> int:
        """Read the current slider value.

        Args:
            widget: The slider widget.

        Returns:
            The current slider value rounded to the nearest integer.
        """
        return extract_slider_state(widget)

    def set_state(self, widget: Slider2D, value: int) -> None:
        """Set the slider's value.

        Args:
            widget: Slider widget object.
            value: Desired step index.
        """
        set_slider_state(widget, value)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        """Convert the slider state to pubsub messages.

        Args:
            state: Selected step index.

        Returns:
            A list with a single `TopicMessage` for the topic `"step_number"`.
        """
        messages = [TopicMessage(name="step_number", value=state)]
        return messages


class TopKWidgetOps:
    """WidgetOps implementation for a Top-K slider."""

    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter
        self.updaters = [
            WidgetUpdater(
                topics=[TopicSpec("visualization", required=True)],
                callback=self.toggle_slider,
            )
        ]

        self._add_kwargs = {
            "xmin": 1,
            "xmax": 1000,
            "value": 1,
            "pos": [(0.1, 0.1), (0.1, 0.85)],
            "title": "Top-K",
        }

    def add(self, callback: Callable) -> Slider2D:
        """Create the slider widget and re-render.

        Args:
            callback: Function called with `(widget, event)` when the UI changes.

        Returns:
            The created `Slider2D` widget.
        """
        self.callback = callback
        widget = self.plotter.add_slider(callback, **self._add_kwargs)
        self.plotter.render()
        return widget

    def remove(self, widget: Slider2D) -> None:
        """Remove the Button."""
        widget.off()
        self.plotter.remove(widget)
        self.plotter.render()

    def set_state(self, widget: Slider2D, value: int) -> None:
        """Set the slider value.

        Args:
            widget: The slider widget.
            value: Desired threshold (integer).
        """
        set_slider_state(widget, value)

    def extract_state(self, widget: Slider2D) -> int:
        """Read the current slider value.

        Args:
            widget: The slider widget.

        Returns:
            The current value as an integer.
        """
        return extract_slider_state(widget)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        """Convert the slider state to pubsub messages.

        Args:
            state: Current threshold value.

        Returns:
            A list with a single `TopicMessage` named `"age_threshold"`.
        """
        return [TopicMessage(name="topk", value=state)]

    def toggle_slider(
        self, widget: Slider2D, msgs: list[TopicMessage]
    ) -> tuple[Slider2D, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}
        msg = msgs_dict["visualization"]

        if msg == "Evidence":
            if widget is not None:
                return widget, False
            return self.add(self.callback), True
        elif msg == "Correlation":
            if widget is None:
                return None, False
            return self.remove(widget), False

        return widget, False


class AlphaWidgetOps:
    """WidgetOps implementation for a EMA alpha."""

    def __init__(self, plotter: Plotter) -> None:
        self.plotter = plotter
        self.updaters = [
            WidgetUpdater(
                topics=[TopicSpec("heuristic", required=True)],
                callback=self.toggle_slider,
            )
        ]

        self._add_kwargs = {
            "xmin": 0.1,
            "xmax": 1,
            "value": 0.5,
            "pos": [(0.85, 0.1), (0.85, 0.85)],
            "title": "Alpha",
        }

    def add(self, callback: Callable) -> Slider2D:
        """Create the slider widget and re-render.

        Args:
            callback: Function called with `(widget, event)` when the UI changes.

        Returns:
            The created `Slider2D` widget.
        """
        self.callback = callback
        widget = self.plotter.add_slider(callback, **self._add_kwargs)
        self.plotter.render()
        return widget

    def remove(self, widget: Slider2D) -> None:
        """Remove the Button."""
        widget.off()
        self.plotter.remove(widget)
        self.plotter.render()

    def set_state(self, widget: Slider2D, value: int) -> None:
        """Set the slider value.

        Args:
            widget: The slider widget.
            value: Desired threshold (integer).
        """
        set_slider_state(widget, value)

    def extract_state(self, widget: Slider2D) -> int:
        """Read the current slider value.

        Args:
            widget: The slider widget.

        Returns:
            The current value as an integer.
        """
        return round(widget.GetRepresentation().GetValue(), 1)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        """Convert the slider state to pubsub messages.

        Args:
            state: Current threshold value.

        Returns:
            A list with a single `TopicMessage` named `"age_threshold"`.
        """
        return [TopicMessage(name="alpha", value=state)]

    def toggle_slider(
        self, widget: Slider2D, msgs: list[TopicMessage]
    ) -> tuple[Slider2D, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}
        msg = msgs_dict["heuristic"]

        if msg == "EMA":
            if widget is not None:
                return widget, False
            return self.add(self.callback), True
        elif msg == "Max Evidence":
            if widget is None:
                return None, False
            return self.remove(widget), False

        return widget, False


class VisualizationWidgetOps:
    """WidgetOps implementation for a visualization type button."""

    def __init__(self, plotter: Plotter):
        self.plotter = plotter

        self._add_kwargs = {
            "pos": (0.9, 0.8),
            "states": ["Evidence", "Correlation"],
            "c": ["w", "w"],
            "bc": ["dg", "db"],
            "size": 30,
            "font": "Calco",
            "bold": True,
        }

    def add(self, callback: Callable) -> Button:
        """Create the button widget and re-render.

        Args:
            callback: Function called with `(widget, event)` on UI interaction.

        Returns:
            The created `vedo.Button`.
        """
        widget = self.plotter.add_button(callback, **self._add_kwargs)
        self.plotter.render()
        self.extract_state(widget)
        return widget

    def extract_state(self, widget: Button) -> str:
        """Read the current button value.

        Args:
            widget: The Button widget.

        Returns:
            The current button state.
        """
        return widget.status()

    def set_state(self, widget: Slider2D, value: str) -> None:
        """Set the slider's value.

        Args:
            widget: Slider widget object.
            value: Desired step index.
        """
        set_button_state(widget, value)

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        """Convert the button state to pubsub messages.

        Args:
            state: Current button state.

        Returns:
            A list with a single `TopicMessage` with the topic "primary_button" .
        """
        messages = [
            TopicMessage(name="visualization", value=state),
        ]
        return messages


class HeuristicWidgetOps:
    """WidgetOps implementation for a Heuristic button."""

    def __init__(self, plotter: Plotter):
        self.plotter = plotter

        self._add_kwargs = {
            "pos": (0.9, 0.9),
            "states": ["Max Evidence", "EMA"],
            "c": ["w", "w"],
            "bc": ["dg", "db"],
            "size": 30,
            "font": "Calco",
            "bold": True,
        }

    def add(self, callback: Callable) -> Button:
        """Create the button widget and re-render.

        Args:
            callback: Function called with `(widget, event)` on UI interaction.

        Returns:
            The created `vedo.Button`.
        """
        widget = self.plotter.add_button(callback, **self._add_kwargs)
        self.plotter.render()
        self.extract_state(widget)
        return widget

    def extract_state(self, widget: Button) -> str:
        """Read the current button value.

        Args:
            widget: The Button widget.

        Returns:
            The current button state.
        """
        return widget.status()

    def set_state(self, widget: Slider2D, value: str) -> None:
        """Set the slider's value.

        Args:
            widget: Slider widget object.
            value: Desired step index.
        """
        set_button_state(widget, value)

    def state_to_messages(self, state: str) -> Iterable[TopicMessage]:
        """Convert the button state to pubsub messages.

        Args:
            state: Current button state.

        Returns:
            A list with a single `TopicMessage` with the topic "primary_button" .
        """
        messages = [
            TopicMessage(name="heuristic", value=state),
        ]
        return messages


class PlotWidgetOps:
    """WidgetOps for a correlation scatter plot with selection highlighting."""

    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser
        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("step_number", required=True),
                    TopicSpec("topk", required=True),
                    TopicSpec("visualization", required=True),
                    TopicSpec("heuristic", required=True),
                    TopicSpec("alpha", required=True),
                ],
                callback=self.update_plot,
            )
        ]
        self._locators = self.create_locators()

        self.df: DataFrame = self.generate_df()

    def create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this widget.

        Returns:
            A dictionary with entries for `"target"` and `"evidence"`.
        """
        locators = {}

        locators["target"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode", value="0"),
                DataLocatorStep.key(name="system", value="target"),
            ]
        )

        locators["evidence"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode", value="0"),
                DataLocatorStep.key(name="system", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="evidences"),
            ]
        )

        locators["rotations"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode", value="0"),
                DataLocatorStep.key(name="system", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="possible_rotations"),
            ]
        )
        return locators

    def generate_df(self) -> DataFrame:
        """Return a DataFrame of hypotheses and their stats."""
        target = self.data_parser.extract(self._locators["target"])
        target_object, target_rotation = (
            target["primary_target_object"],
            target["primary_target_rotation_quat"],
        )
        target_rotation = Rotation.from_quat(target_rotation)

        # Get evidence values
        evidence = self.data_parser.extract(self._locators["evidence"])
        evidence = [x[target_object] for x in evidence]
        df = (
            pd.DataFrame(evidence)
            .rename_axis("step")
            .reset_index()
            .melt(id_vars="step", var_name="hypothesis", value_name="evidence")
        )

        # Calculate error values
        rotations = self.data_parser.extract(self._locators["rotations"])
        rotations = [x[target_object] for x in rotations]
        rotations = Rotation.from_matrix(rotations[0])  # only one step stored
        er = (rotations * target_rotation.inv()).magnitude()
        df["error"] = df["hypothesis"].map(dict(enumerate(er)))

        return df

    def add_evidence_plot(
        self,
        step: int,
        k: int,
    ) -> list[int]:
        """Plot the top-k hypotheses at a specific step, then show those hypotheses.

        Args:
            step: Step index to rank hypotheses by evidence.
            k: Number of top hypotheses to plot.

        Returns:
            List of selected hypothesis IDs (length may be < k if fewer available).
        """
        # Pick top-k by evidence at the given step
        top_hyps = (
            self.df.loc[self.df["step"] == step]
            .sort_values("evidence", ascending=False)
            .head(k)["hypothesis"]
            .tolist()
        )

        # Filter to those hypotheses across all steps
        plot_df = self.df[self.df["hypothesis"].isin(top_hyps)].sort_values(
            ["hypothesis", "step"]
        )

        _, ax = plt.subplots(figsize=(9, 4.5))
        created_fig = True

        # Line plot: each hypothesis is a separate line
        g = sns.lineplot(
            data=plot_df,
            x="step",
            y="evidence",
            hue="hypothesis",
            hue_order=top_hyps,
            palette=dict.fromkeys(top_hyps, "C0"),
            marker="o",
            errorbar=None,
            ax=ax,
        )

        # Vertical marker at the selection step
        ax.axvline(step, linestyle="--", color="black", linewidth=1)

        ax.set_xlabel("step")
        ax.set_ylabel("evidence")
        ax.set_title(f"Top-{len(top_hyps)} hypotheses at step {step}")
        ax.get_legend().remove()
        plt.tight_layout()

        widget = Image(g.figure)
        plt.close(g.figure)
        self.plotter.add(widget)
        return widget

    def add_correlation_plot(self, step: int) -> Image:
        """Create a seaborn joint scatter (evidence vs error) for a given step.

        Args:
            step: Step index to filter the dataframe.

        Returns:
            The Image widget for the correlation plot.
        """
        # Filter dataframe to just this step
        df_step = self.df[self.df["step"] == step][["evidence", "error"]].dropna()

        # Build JointGrid
        g = sns.JointGrid(data=df_step, x="evidence", y="error", height=6, space=0)

        base_color = "C0"

        # Joint scatter
        sns.scatterplot(
            data=df_step,
            x="evidence",
            y="error",
            ax=g.ax_joint,
            s=12,
            alpha=0.7,
            edgecolor="none",
            color=base_color,
            legend=False,
        )

        # Marginal KDEs
        sns.kdeplot(
            data=df_step,
            x="evidence",
            ax=g.ax_marg_x,
            fill=True,
            alpha=0.2,
            linewidth=0,
            color=base_color,
            legend=False,
        )
        sns.kdeplot(
            data=df_step,
            y="error",
            ax=g.ax_marg_y,
            fill=True,
            alpha=0.2,
            linewidth=0,
            color=base_color,
            legend=False,
        )

        # Remove any auto legends if present
        for ax in (g.ax_joint, g.ax_marg_x, g.ax_marg_y):
            leg = ax.get_legend()
            if leg:
                leg.remove()

        # Labels/layout
        g.ax_joint.set_xlabel("Evidence", labelpad=10)
        g.ax_joint.set_ylabel("Pose Error", labelpad=10)
        g.ax_joint.set_title(f"Correlation at step {step}")
        g.figure.tight_layout()

        widget = Image(g.figure).shift((150, 0, 0))
        plt.close(g.figure)
        self.plotter.add(widget)
        return widget

    def add_evidence_ema_plot(self, step: int, k: int, alpha: float) -> list[int]:
        """Plot the top-k hypotheses at a specific step, then show those hypotheses.

        Args:
            step: Step index to rank hypotheses by evidence.
            k: Number of top hypotheses to plot.
            alpha: EMA alpha value.

        Returns:
            List of selected hypothesis IDs (length may be < k if fewer available).
        """
        # Ensure proper ordering for diffs/EMA
        df_sorted = self.df.sort_values(["hypothesis", "step"]).copy()

        # 1) Per-step evidence increase per hypothesis
        #    For the first step of each hypothesis, delta = evidence (baseline gain).
        df_sorted["evidence_delta"] = (
            df_sorted.groupby("hypothesis")["evidence"]
            .apply(lambda s: s.diff().fillna(s))
            .reset_index(level=0, drop=True)
        )

        # 2) EMA over deltas per hypothesis
        #    adjust=False gives the standard recursive EMA:
        #      ema[t] = alpha * delta[t] + (1-alpha) * ema[t-1]
        df_sorted["evidence_ema"] = (
            df_sorted.groupby("hypothesis")["evidence_delta"]
            .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
            .reset_index(level=0, drop=True)
        )

        # 3) Select top-k hypotheses by EMA at the requested step
        top_hyps = (
            df_sorted.loc[df_sorted["step"] == step]
            .sort_values("evidence_ema", ascending=False)
            .head(k)["hypothesis"]
            .tolist()
        )

        # 4) Plot only those hypotheses across all steps
        plot_df = df_sorted[df_sorted["hypothesis"].isin(top_hyps)].sort_values(
            ["hypothesis", "step"]
        )

        fig, ax = plt.subplots(figsize=(9, 4.5))

        # Single-color palette for all lines but keep hue to get separate line objects
        palette = dict.fromkeys(top_hyps, "C0")
        g = sns.lineplot(
            data=plot_df,
            x="step",
            y="evidence_ema",
            hue="hypothesis",
            hue_order=top_hyps,
            palette=palette,
            marker="o",
            errorbar=None,
            ax=ax,
        )

        # Vertical marker at the selection step
        ax.axvline(step, linestyle="--", color="black", linewidth=1)

        ax.set_xlabel("step")
        ax.set_ylabel(f"EMA with (alpha={alpha:g})")
        ax.set_title(f"Top-{len(top_hyps)} hypotheses at step {step} by EMA")
        leg = ax.get_legend()
        if leg:
            leg.remove()
        plt.tight_layout()

        widget = Image(g.figure)
        plt.close(g.figure)
        self.plotter.add(widget)
        return widget

    def add_correlation_ema_plot(self, step: int, alpha: float) -> Image:
        """Create a seaborn joint scatter of EMA vs error for a given step.

        The EMA is computed per hypothesis over per-step evidence increments:
          delta[t] = evidence[t] - evidence[t-1], with t=0 using evidence[0].
          ema[t]   = alpha * delta[t] + (1 - alpha) * ema[t-1]

        Args:
            step: Step index to filter the dataframe.
            alpha: EMA smoothing factor in (0, 1].

        Returns:
            The Image widget for the correlation plot.
        """
        # Compute EMA(Δevidence) per hypothesis across steps
        df_sorted = self.df.sort_values(["hypothesis", "step"]).copy()

        # Per-step increase; first step's delta is the initial evidence
        df_sorted["evidence_delta"] = (
            df_sorted.groupby("hypothesis")["evidence"]
            .apply(lambda s: s.diff().fillna(s))
            .reset_index(level=0, drop=True)
        )

        # Recursive EMA over deltas
        df_sorted["evidence_ema"] = (
            df_sorted.groupby("hypothesis")["evidence_delta"]
            .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
            .reset_index(level=0, drop=True)
        )

        # Data at the requested step
        df_step = df_sorted[df_sorted["step"] == step][
            ["evidence_ema", "error"]
        ].dropna()

        # Build JointGrid
        g = sns.JointGrid(data=df_step, x="evidence_ema", y="error", height=6, space=0)

        base_color = "C0"

        # Joint scatter
        sns.scatterplot(
            data=df_step,
            x="evidence_ema",
            y="error",
            ax=g.ax_joint,
            s=12,
            alpha=0.7,
            edgecolor="none",
            color=base_color,
            legend=False,
        )

        # Marginal KDEs
        sns.kdeplot(
            data=df_step,
            x="evidence_ema",
            ax=g.ax_marg_x,
            fill=True,
            alpha=0.2,
            linewidth=0,
            color=base_color,
            legend=False,
        )
        sns.kdeplot(
            data=df_step,
            y="error",
            ax=g.ax_marg_y,
            fill=True,
            alpha=0.2,
            linewidth=0,
            color=base_color,
            legend=False,
        )

        # Remove any auto legends if present
        for ax in (g.ax_joint, g.ax_marg_x, g.ax_marg_y):
            leg = ax.get_legend()
            if leg:
                leg.remove()

        # Labels/layout
        g.ax_joint.set_xlabel(f"EMA, alpha={alpha:g}", labelpad=10)
        g.ax_joint.set_ylabel("Pose Error", labelpad=10)
        g.ax_joint.set_title(f"Correlation at step {step}")
        g.figure.tight_layout()

        widget = Image(g.figure).shift((150, 0, 0))
        plt.close(g.figure)
        self.plotter.add(widget)
        return widget

    def update_plot(
        self, widget: Image, msgs: list[TopicMessage]
    ) -> tuple[Image, bool]:
        """Rebuild the plot for the selected step, visualization, and topk.

        Removes previous plot, generates the new DataFrame, creates a
        new plot, places it in the scene.

        Args:
            widget: The previous figure, if any.
            msgs: Messages received, containing `"step_number"`,
                `"visualization"`, and `"topk"`.

        Returns:
            `(new_widget, True)` where `new_widget` is the new image actor.
        """
        # Build DataFrame and filter by age
        msgs_dict = {msg.name: msg.value for msg in msgs}

        # Add the scatter correlation plot to the scene
        if widget is not None:
            self.plotter.remove(widget)

        if msgs_dict["heuristic"] == "Max Evidence":
            if msgs_dict["visualization"] == "Evidence":
                widget = self.add_evidence_plot(
                    step=msgs_dict["step_number"], k=msgs_dict["topk"]
                )
            elif msgs_dict["visualization"] == "Correlation":
                widget = self.add_correlation_plot(step=msgs_dict["step_number"])

        elif msgs_dict["heuristic"] == "EMA":
            if msgs_dict["visualization"] == "Evidence":
                widget = self.add_evidence_ema_plot(
                    step=msgs_dict["step_number"],
                    k=msgs_dict["topk"],
                    alpha=msgs_dict["alpha"],
                )
            elif msgs_dict["visualization"] == "Correlation":
                widget = self.add_correlation_ema_plot(
                    step=msgs_dict["step_number"], alpha=msgs_dict["alpha"]
                )

        self.plotter.render()
        return widget, False


class ClickWidgetOps:
    """Captures 3D click positions and publish them on the bus.

    This class registers plotter-level mouse callbacks. A left-click picks a 3D
    point (if available) and triggers the widget callback; a right-click
    resets the camera pose. There is no visual widget created by this class.

    Attributes:
        plotter: The `vedo.Plotter` where callbacks are installed.
        cam_dict: Dictionary for camera default specs.
        click_location: Last picked 3D location, if any.
    """

    def __init__(self, plotter: Plotter, cam_dict: dict[str, Any]) -> None:
        self.plotter = plotter
        self.cam_dict = cam_dict

    def add(self, callback: Callable) -> None:
        """Register mouse callbacks on the plotter.

        Note that this callback makes use of the `VtkDebounceScheduler`
        to publish messages. Storing the callback and triggering it, will
        simulate a UI change on e.g., a button or a slider, which schedules
        a publish. We use this callback because this event is not triggered
        by receiving topics from a `WidgetUpdater`.


        Args:
            callback: Function invoked like `(widget, event)` when a left-click
                captures a 3D location.
        """
        self.plotter.add_callback("RightButtonPress", self.on_left_click)

    def on_left_click(self, event):
        """Handle right mouse press (reset camera pose and render).

        Notes:
            Bound to the "RightButtonPress" event in `self.add()`.
        """
        renderer = self.plotter.renderer
        if renderer is not None:
            cam = renderer.GetActiveCamera()
            cam.SetPosition(self.cam_dict["pos"])
            cam.SetFocalPoint(self.cam_dict["focal_point"])
            cam.SetViewUp((0, 1, 0))
            cam.SetClippingRange((0.01, 1000.01))
            self.plotter.render()


class InteractivePlot:
    """An interactive plot for visualizing max evidence update."""

    def __init__(
        self,
        exp_path: str,
    ):
        self.data_parser = DataParser(exp_path)
        self.event_bus = Publisher()
        self.plotter = Plotter().render()
        self.scheduler = VtkDebounceScheduler(self.plotter.interactor, period_ms=33)

        # create and add the widgets to the plotter
        self._widgets = self.create_widgets()
        for w in self._widgets.values():
            w.add()
        self._widgets["step_slider"].set_state(0)
        self._widgets["topk_slider"].set_state(1)
        self._widgets["alpha_slider"].set_state(0.5)
        self._widgets["visualization_button"].set_state("Evidence")
        self._widgets["heuristic_button"].set_state("Max Evidence")

        self.plotter.show(interactive=True, resetcam=False, camera=self.cam_dict())

    def cam_dict(self) -> dict[str, tuple[float, float, float]]:
        """Returns camera parameters for an overhead view of the plot.

        Returns:
            Dictionary with camera position and focal point.
        """
        x_val = 500
        y_val = 250
        z_val = 1500
        return {"pos": (x_val, y_val, z_val), "focal_point": (x_val, y_val, 0)}

    def create_widgets(self):
        widgets = {}

        widgets["step_slider"] = Widget[Slider2D, int](
            widget_ops=StepSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["topk_slider"] = Widget[Slider2D, int](
            widget_ops=TopKWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["visualization_button"] = Widget[Button, str](
            widget_ops=VisualizationWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["heuristic_button"] = Widget[Button, str](
            widget_ops=HeuristicWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["alpha_slider"] = Widget[Slider2D, float](
            widget_ops=AlphaWidgetOps(plotter=self.plotter),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["plot"] = Widget[Image, Series](
            widget_ops=PlotWidgetOps(
                plotter=self.plotter, data_parser=self.data_parser
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.0,
            dedupe=False,
        )

        widgets["click_widget"] = Widget[None, None](
            widget_ops=ClickWidgetOps(plotter=self.plotter, cam_dict=self.cam_dict()),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.1,
            dedupe=True,
        )

        return widgets


@register(
    "interactive_max_evidence",
    description="Max evidence plot",
)
def main(experiment_log_dir: str) -> int:
    """Interactive visualization for inspecting the max evidence update.

    Args:
        experiment_log_dir: Path to the experiment directory containing the detailed
            stats file.

    Returns:
        Exit code.
    """
    if not Path(experiment_log_dir).exists():
        logger.error(f"Experiment path not found: {experiment_log_dir}")
        return 1

    InteractivePlot(experiment_log_dir)

    return 0


@attach_args("interactive_max_evidence")
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
