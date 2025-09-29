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

import numpy as np
from pubsub.core import Publisher
from vedo import Circle, Line, Mesh, Plotter, Slider2D, Sphere, Text2D

from tbp.interactive.data import (
    DataLocator,
    DataLocatorStep,
    DataParser,
    YCBMeshLoader,
)
from tbp.interactive.topics import TopicMessage, TopicSpec
from tbp.interactive.widget_updaters import WidgetUpdater
from tbp.interactive.widgets import (
    VtkDebounceScheduler,
    Widget,
    extract_slider_state,
    set_slider_state,
)
from tbp.plot.registry import attach_args, register

logger = logging.getLogger(__name__)


class EpisodeSliderWidgetOps:
    """WidgetOps implementation for an Episode slider.

    This class sets the slider's range based on the number of
    available episodes and publishes changes as `TopicMessage` items
    under the "episode_number" topic.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        data_parser: A parser that extracts or queries information from the
            json log file.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
        _locators: Data accessors keyed by name. These instruct the DataParser
            on how to retrieve the required information.
    """

    def __init__(self, plotter: Plotter, data_parser: DataParser) -> None:
        self.plotter = plotter
        self.data_parser = data_parser

        self._add_kwargs = {
            "xmin": 0,
            "xmax": 10,
            "value": 0,
            "pos": [(0.1, 0.2), (0.9, 0.2)],
            "title": "Episode",
        }

        self._locators = self.create_locators()

    def create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this widget.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}

        locators["episode"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
            ],
        )
        return locators

    def add(self, callback: Callable[[Slider2D, str], None]) -> Slider2D:
        """Create the slider widget and set its range from the data.

        The slider's `xmax` is set to the number of episodes.

        Args:
            callback: Function called with the arguments `(widget, event)` when
                the slider changes in the UI.

        Returns:
            The created widget as returned by the plotter.
        """
        kwargs = deepcopy(self._add_kwargs)
        locator = self._locators["episode"]
        kwargs.update({"xmax": len(self.data_parser.query(locator)) - 1})
        widget = self.plotter.at(0).add_slider(callback, **kwargs)
        self.plotter.at(0).render()
        return widget

    def extract_state(self, widget: Slider2D) -> int:
        """Read the current slider value from its VTK representation.

        Args:
            widget: The widget object.

        Returns:
            The current slider value rounded to the nearest integer.
        """
        return extract_slider_state(widget)

    def set_state(self, widget: Slider2D, value: int) -> None:
        """Set the slider's value.

        Args:
            widget: Slider widget object.
            value: Desired episode index.
        """
        set_slider_state(widget, value)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        """Convert the slider state to pubsub messages.

        Args:
            state: Selected episode index.

        Returns:
            A list with a single `TopicMessage` named `"episode_number"`.
        """
        messages = [TopicMessage(name="episode_number", value=state)]
        return messages


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
        self.updaters = [
            WidgetUpdater(
                topics=[TopicSpec("episode_number", required=True)],
                callback=self.update_slider_range,
            )
        ]

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
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
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
        widget = self.plotter.at(0).add_slider(callback, **self._add_kwargs)
        self.plotter.at(0).render()
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

    def update_slider_range(
        self, widget: Slider2D, msgs: list[TopicMessage]
    ) -> tuple[Slider2D, bool]:
        """Adjust slider range based on the selected episode and reset to 0.

        Looks up the `"episode_number"` message, queries the number of steps for
        that episode, sets the slider range to `[0, num_steps - 1]`, resets the
        value to 0, and re-renders.

        Args:
            widget: The slider widget to update.
            msgs: Messages from the `WidgetUpdater`.

        Returns:
            A tuple `(widget, True)` indicating the updated widget and whether
            a publish should occur.
        """
        msgs_dict = {msg.name: msg.value for msg in msgs}

        # set widget range to the correct step number
        widget.range = [
            0,
            len(
                self.data_parser.query(
                    self._locators["step"], episode=str(msgs_dict["episode_number"])
                )
            )
            - 1,
        ]

        # set slider value back to zero
        self.set_state(widget, 0)
        self.plotter.at(0).render()

        return widget, True


class GtMeshWidgetOps:
    """WidgetOps implementation for rendering the ground-truth target mesh.

    This widget is display-only. It listens for `"episode_number"` updates,
    loads the target object's YCB mesh, applies the episode-specific rotations,
    scales and positions it, and adds it to the plotter. It does not publish
    any messages.

    Attributes:
        plotter: A `vedo.Plotter` used to add and remove actors.
        data_parser: A parser that extracts entries from the JSON log.
        ycb_loader: Loader that returns a textured `vedo.Mesh` for a YCB object.
        updaters: A single `WidgetUpdater` that reacts to `"episode_number"`.
        _locators: Data accessors keyed by name for the parser.
    """

    def __init__(
        self, plotter: Plotter, data_parser: DataParser, ycb_loader: YCBMeshLoader
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.ycb_loader = ycb_loader
        self.updaters = [
            WidgetUpdater(
                topics=[TopicSpec("episode_number", required=True)],
                callback=self.update_mesh,
            ),
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                ],
                callback=self.update_sensor,
            ),
        ]
        self._locators = self.create_locators()

        self.gaze_line: Line | None = None
        self.sensor_circle: Circle | None = None

        self.plotter.at(1).add(Text2D(txt="Ground Truth", pos="top-center"))

    def create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this widget.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}
        locators["target"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="target"),
            ]
        )

        locators["steps_mask"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="system", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="lm_processed_steps"),
            ]
        )

        locators["sensor_location"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="system", value="motor_system"),
                DataLocatorStep.key(name="telemetry", value="action_sequence"),
                DataLocatorStep.index(name="sm_step"),
                DataLocatorStep.index(name="telemetry_type", value=1),
                DataLocatorStep.key(name="agent", value="agent_id_0"),
                DataLocatorStep.key(name="pose", value="position"),
            ]
        )

        locators["patch_location"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="system", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="locations"),
                DataLocatorStep.key(name="sm", value="patch"),
                DataLocatorStep.index(name="step"),
            ]
        )

        return locators

    def remove(self, widget: Mesh) -> None:
        """Remove the mesh widget and re-render.

        Args:
            widget: The mesh widget to remove. If `None`, no action is taken.
        """
        if widget is not None:
            self.plotter.at(1).remove(widget)
            self.plotter.at(1).render()

    def update_mesh(self, widget: Mesh, msgs: list[TopicMessage]) -> tuple[Mesh, bool]:
        """Update the target mesh when the episode changes.

        Removes any existing mesh, loads the episode's primary target object,
        applies its Euler rotations, scales and positions it, then adds it to
        the plotter.

        Args:
            widget: The currently displayed mesh, if any.
            msgs: Messages received from the `WidgetUpdater`.

        Returns:
            A tuple `(mesh, False)`. The second value is `False` to indicate
            that no publish should occur.
        """
        self.remove(widget)
        msgs_dict = {msg.name: msg.value for msg in msgs}

        locator = self._locators["target"]
        target = self.data_parser.extract(
            locator, episode=str(msgs_dict["episode_number"])
        )
        target_id = target["primary_target_object"]
        target_rot = target["primary_target_rotation_euler"]
        target_pos = target["primary_target_position"]
        widget = self.ycb_loader.create_mesh(target_id).clone(deep=True)
        widget.rotate_x(target_rot[0])
        widget.rotate_y(target_rot[1])
        widget.rotate_z(target_rot[2])
        widget.shift(*target_pos)

        self.plotter.at(1).add(widget)
        self.plotter.at(1).render()

        return widget, False

    def update_sensor(
        self, widget: None, msgs: list[TopicMessage]
    ) -> tuple[None, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}
        episode_number = msgs_dict["episode_number"]
        step_number = msgs_dict["step_number"]

        steps_mask = self.data_parser.extract(
            self._locators["steps_mask"], episode=str(episode_number)
        )
        mapping = np.flatnonzero(steps_mask)

        sensor_pos = self.data_parser.extract(
            self._locators["sensor_location"],
            episode=str(episode_number),
            sm_step=int(mapping[step_number]),
        )

        patch_pos = self.data_parser.extract(
            self._locators["patch_location"],
            episode=str(episode_number),
            step=step_number,
        )

        if self.sensor_circle is None:
            self.sensor_circle = Sphere(pos=sensor_pos, r=0.01)
            self.plotter.at(1).add(self.sensor_circle)
        self.sensor_circle.pos(sensor_pos)

        if self.gaze_line is None:
            self.gaze_line = Line(sensor_pos, patch_pos, c="black", lw=2)
            self.plotter.at(1).add(self.gaze_line)
        self.gaze_line.points = [sensor_pos, patch_pos]

        self.plotter.at(1).render()

        return widget, False


class MlhMeshWidgetOps:
    """WidgetOps implementation for rendering the MLH mesh.

    This widget is display-only. It listens for `"episode_number"` updates,
    loads the target object's YCB mesh, applies the episode-specific rotations,
    scales and positions it, and adds it to the plotter. It does not publish
    any messages.

    Attributes:
        plotter: A `vedo.Plotter` used to add and remove actors.
        data_parser: A parser that extracts entries from the JSON log.
        ycb_loader: Loader that returns a textured `vedo.Mesh` for a YCB object.
        updaters: A single `WidgetUpdater` that reacts to `"episode_number"`.
        _locators: Data accessors keyed by name for the parser.
    """

    def __init__(
        self, plotter: Plotter, data_parser: DataParser, ycb_loader: YCBMeshLoader
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.ycb_loader = ycb_loader
        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
                ],
                callback=self.update_mesh,
            ),
        ]
        self._locators = self.create_locators()

        self.default_object_position = (0, 1.5, 0)
        self.sensor_circle: Circle | None = None

        self.plotter.at(2).add(Text2D(txt="MLH", pos="top-center"))

    def create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this widget.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}

        locators["mlh"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="current_mlh"),
                DataLocatorStep.index(name="step"),
            ]
        )
        return locators

    def update_mesh(self, widget: Mesh, msgs: list[TopicMessage]) -> tuple[Mesh, bool]:
        """Update the target mesh when the episode changes.

        Removes any existing mesh, loads the episode's primary target object,
        applies its Euler rotations, scales and positions it, then adds it to
        the plotter.

        Args:
            widget: The currently displayed mesh, if any.
            msgs: Messages received from the `WidgetUpdater`.

        Returns:
            A tuple `(mesh, False)`. The second value is `False` to indicate
            that no publish should occur.
        """
        if widget is not None:
            self.plotter.at(2).remove(widget)

        if self.sensor_circle is not None:
            self.plotter.at(2).remove(self.sensor_circle)
            self.sensor_circle = None

        msgs_dict = {msg.name: msg.value for msg in msgs}
        mlh = self.data_parser.extract(
            self._locators["mlh"],
            episode=str(msgs_dict["episode_number"]),
            step=msgs_dict["step_number"],
        )
        mlh_id = mlh["graph_id"]
        mlh_rot = mlh["rotation"]
        mlh_pos = mlh["location"]

        # Add object_mesh
        widget = self.ycb_loader.create_mesh(mlh_id).clone(deep=True)
        widget.rotate_x(mlh_rot[0])
        widget.rotate_y(mlh_rot[1])
        widget.rotate_z(mlh_rot[2])
        widget.shift(*self.default_object_position)

        self.plotter.at(2).add(widget)

        # Add sensor circle
        self.sensor_circle = Sphere(pos=mlh_pos, r=0.01).c("green")
        self.plotter.at(2).add(self.sensor_circle)

        self.plotter.at(2).render()

        return widget, False


class InteractivePlot:
    """An interactive plot for a simple tutorial."""

    def __init__(
        self,
        exp_path: str,
        data_path: str,
    ):
        renderer_areas = [
            {"bottomleft": (0.0, 0.0), "topright": (1.0, 1.0)},
            {"bottomleft": (0.1, 0.3), "topright": (0.45, 0.8)},
            {"bottomleft": (0.55, 0.3), "topright": (0.9, 0.8)},
        ]

        self.axes_dict = {
            "xrange": (-0.05, 0.05),
            "yrange": (1.45, 1.55),
            "zrange": (-0.05, 0.05),
        }
        self.cam_dict = {"pos": (0, 0, 1), "focal_point": (0, 0, 0)}

        self.data_parser = DataParser(exp_path)
        self.ycb_loader = YCBMeshLoader(data_path)
        self.event_bus = Publisher()
        self.plotter = Plotter(shape=renderer_areas, sharecam=False).render()
        self.scheduler = VtkDebounceScheduler(self.plotter.interactor, period_ms=33)

        # Create and add the widgets to the plotter
        self._widgets = self.create_widgets()
        for w in self._widgets.values():
            w.add()
        self._widgets["episode_slider"].set_state(0)

        # Show the plot renderers
        self.plotter.at(0).show(
            camera=deepcopy(self.cam_dict),
            interactive=False,
            resetcam=False,
        )
        self.plotter.at(1).show(
            axes=deepcopy(self.axes_dict),
            interactive=False,
            resetcam=True,
        )
        self.plotter.at(2).show(
            axes=deepcopy(self.axes_dict),
            interactive=True,
            resetcam=True,
        )

    def create_widgets(self):
        widgets = {}

        widgets["episode_slider"] = Widget[Slider2D, int](
            widget_ops=EpisodeSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.3,
            dedupe=True,
        )

        widgets["step_slider"] = Widget[Slider2D, int](
            widget_ops=StepSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.3,
            dedupe=True,
        )

        widgets["primary_mesh"] = Widget[Mesh, None](
            widget_ops=GtMeshWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                ycb_loader=self.ycb_loader,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["Mlh_mesh"] = Widget[Mesh, None](
            widget_ops=MlhMeshWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                ycb_loader=self.ycb_loader,
            ),
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        return widgets


@register(
    "interactive_tutorial",
    description="Simple tutorial with four widgets",
)
def main(experiment_log_dir: str, objects_mesh_dir: str) -> int:
    """Interactive tutorial visualization.

    Args:
        experiment_log_dir: Path to the experiment directory containing the detailed
            stats file.
        objects_mesh_dir: Path to the root directory of YCB object meshes.

    Returns:
        Exit code.
    """
    if not Path(experiment_log_dir).exists():
        logger.error(f"Experiment path not found: {experiment_log_dir}")
        return 1

    data_path = str(Path(objects_mesh_dir).expanduser())

    InteractivePlot(experiment_log_dir, data_path)

    return 0


@attach_args("interactive_tutorial")
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
    p.add_argument(
        "--objects_mesh_dir",
        default="~/tbp/data/habitat/objects/ycb/meshes",
        help=("The directory containing the mesh objects."),
    )
