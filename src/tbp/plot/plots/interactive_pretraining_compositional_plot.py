# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import bisect
import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import vedo
from pubsub.core import Publisher
from scipy.spatial.transform import Rotation
from vedo import (
    Image,
    Line,
    Mesh,
    Plotter,
    Points,
    Slider2D,
    Sphere,
    Text2D,
    settings,
)

from tbp.interactive.animator import (
    WidgetAnimator,
    make_slider_step_actions_for_widget,
)
from tbp.interactive.colors import Palette
from tbp.interactive.data import (
    DataLocator,
    DataLocatorStep,
    DataParser,
    PretrainedModelsLoader,
    YCBMeshLoader,
)
from tbp.interactive.events import EventSpec
from tbp.interactive.scopes import ScopeViewer
from tbp.interactive.topics import TopicMessage, TopicSpec
from tbp.interactive.widget_updaters import WidgetUpdater
from tbp.interactive.widgets import (
    VtkDebounceScheduler,
    Widget,
    extract_slider_state,
    set_slider_state,
)
from tbp.plot.registry import attach_args, register

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

# Configure vedo settings
settings.immediate_rendering = False
settings.default_font = "Arial"

FONT = "Arial"
FONT_SIZE = 25

COLOR_PALETTE = {
    "Blue": Palette.as_hex("numenta_blue"),
    "Pink": Palette.as_hex("pink"),
    "Purple": Palette.as_hex("purple"),
    "Gold": Palette.as_hex("gold"),
    "Green": Palette.as_hex("green"),
    "Primary": Palette.as_hex("numenta_blue"),
    "Secondary": Palette.as_hex("purple"),
    "Accent": Palette.as_hex("charcoal"),
    "Accent2": Palette.as_hex("link_water"),
    "Accent3": Palette.as_hex("rich_black"),
}


MAIN_RENDERER_IX = 0
SIMULATOR_IX = 1
MODEL_IX = 2


class StepMapper:
    """Bidirectional mapping between global step indices and (episode, local_step).

    Global steps are defined as the concatenation of all local episode steps:

        episode 0: steps [0, ..., n0 - 1]
        episode 1: steps [0, ..., n1 - 1]
        ...

    Global index is:
        [0, ..., n0 - 1, n0, ..., n0 + n1 - 1, ...]
    """

    def __init__(self, data_parser: DataParser) -> None:
        self.data_parser = data_parser
        self._locators = self._create_locators()

        # number of steps in each episode
        self._episode_lengths: list[int] = self._compute_episode_lengths()

        # global offset of each episode
        self._prefix_sums: list[int] = self._compute_prefix_sums()

    def _create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used to access episode steps.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}
        locators["step"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="time"),
            ]
        )
        return locators

    def _compute_episode_lengths(self) -> list[int]:
        locator = self._locators["step"]

        episode_lengths: list[int] = []

        for episode in self.data_parser.query(locator):
            episode_lengths.append(
                len(self.data_parser.extract(locator, episode=episode))
            )

        if not episode_lengths:
            raise RuntimeError("No episodes found while computing episode lengths.")

        return episode_lengths

    def _compute_prefix_sums(self) -> list[int]:
        prefix_sums = [0]
        for length in self._episode_lengths:
            prefix_sums.append(prefix_sums[-1] + length)
        return prefix_sums

    @property
    def num_episodes(self) -> int:
        return len(self._episode_lengths)

    @property
    def total_num_steps(self) -> int:
        """Total number of steps across all episodes."""
        return self._prefix_sums[-1]

    def global_to_local(self, global_step: int) -> tuple[int, int]:
        """Convert a global step index into (episode, local_step).

        Args:
            global_step: Global step index in the range
                `[0, total_num_steps)`.

        Returns:
            A pair `(episode, local_step)` such that:
              * `episode` is the zero based episode index.
              * `local_step` is the zero based step index within that episode.

        Raises:
            IndexError: If `global_step` is negative or not less than
                `total_num_steps`.
        """
        if global_step < 0 or global_step >= self.total_num_steps:
            raise IndexError(
                f"global_step {global_step} is out of range [0, {self.total_num_steps})"
            )

        episode = bisect.bisect_right(self._prefix_sums, global_step) - 1
        local_step = global_step - self._prefix_sums[episode]
        return episode, local_step

    def local_to_global(self, episode: int, step: int) -> int:
        """Convert an (episode, local_step) pair into a global step index.

        Args:
            episode: Zero based episode index in the range
                `[0, num_episodes)`.
            step: Zero based step index within the given episode. Must be in
                `[0, number_of_steps_in_episode)`.

        Returns:
            The corresponding global step index, in the range
            `[0, total_num_steps)`.

        Raises:
            IndexError: If `episode` is out of range, or if `step` is out of
                range for the given episode.
        """
        if episode < 0 or episode >= self.num_episodes:
            raise IndexError(
                f"episode {episode} is out of range [0, {self.num_episodes})"
            )

        num_steps_in_episode = self._episode_lengths[episode]
        if step < 0 or step >= num_steps_in_episode:
            raise IndexError(
                f"step {step} is out of range [0, {num_steps_in_episode}) "
                f"for episode {episode}"
            )

        return self._prefix_sums[episode] + step


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
        self.subrenderers = [
            SIMULATOR_IX,
            MODEL_IX,
        ]

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
        self.plotter.at(0).add_callback("RightButtonPress", self.on_right_click)

    def align_camera(self, cams: list[Any], cam_clicked: Any) -> None:
        """Align the camera objects."""
        for cam in cams:
            cam.SetPosition(cam_clicked.GetPosition())
            cam.SetFocalPoint(cam_clicked.GetFocalPoint())
            cam.SetViewUp(cam_clicked.GetViewUp())
            cam.SetClippingRange(cam_clicked.GetClippingRange())
            cam.SetParallelScale(cam_clicked.GetParallelScale())

    def get_cam(self, i):
        return self.plotter.renderers[i].GetActiveCamera()

    def on_right_click(self, event):
        """Handle right mouse press (reset camera pose and render).

        Notes:
            Bound to the "RightButtonPress" event in `self.add()`.
        """
        if event.at == MAIN_RENDERER_IX:
            renderer = self.plotter.at(0).renderer
            if renderer is not None:
                cam = renderer.GetActiveCamera()
                cam.SetPosition(self.cam_dict["pos"])
                cam.SetFocalPoint(self.cam_dict["focal_point"])
                cam.SetViewUp((0, 1, 0))
                cam.SetClippingRange((0.01, 1000.01))
                self.plotter.at(0).render()
        elif event.at in self.subrenderers:
            cam_clicked = self.get_cam(event.at)
            cam_copy = [self.get_cam(i) for i in self.subrenderers if i != event.at]
            self.align_camera(cam_copy, cam_clicked)


class StepSliderWidgetOps:
    """WidgetOps implementation for a Step slider.

    This class adds a slider widget for the global step. It uses the step mapper
    to retrieve information about the total number of steps. The published state
    is the global step index; subscribers use their own step_mapper to decode
    episode and local step as needed.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        data_parser: A parser that extracts or queries information from the
            JSON log file.
        step_mapper: A mapper between local and global step indices.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
    """

    def __init__(
        self,
        plotter: Plotter,
        data_parser: DataParser,
        step_mapper: StepMapper,
    ) -> None:
        self.plotter = plotter
        self.data_parser = data_parser

        self._add_kwargs = {
            "xmin": 0,
            "xmax": 10,
            "value": 0,
            "pos": [(0.11, 0.06), (0.89, 0.06)],
            "title": "Step",
            "font": FONT,
            "show_value": False,
        }

        self.step_mapper = step_mapper

    def add(self, callback: Callable) -> Slider2D:
        kwargs = deepcopy(self._add_kwargs)
        kwargs.update({"xmax": self.step_mapper.total_num_steps - 1})
        widget = self.plotter.at(0).add_slider(callback, **kwargs)
        self.plotter.at(0).render()
        return widget

    def remove(self, widget: Slider2D) -> None:
        self.plotter.at(0).remove(widget)
        self.plotter.at(0).render()

    def extract_state(self, widget: Slider2D) -> int:
        return extract_slider_state(widget)

    def set_state(self, widget: Slider2D, value: int) -> None:
        set_slider_state(widget, value)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        return [TopicMessage(name="global_step", value=state)]


class GtMeshWidgetOps:
    """WidgetOps implementation for rendering the ground-truth target mesh.

    This widget is display-only. It listens for `"global_step"` updates,
    loads the target object's YCB mesh, applies the episode-specific rotations,
    scales and positions it, and adds it to the plotter. It does not publish
    any messages.

    Attributes:
        plotter: A `vedo.Plotter` used to add and remove actors.
        data_parser: A parser that extracts entries from the JSON log.
        ycb_loader: Loader that returns a textured `vedo.Mesh` for a YCB object.
        step_mapper: Mapper to convert global step indices to (episode, local_step).
        updaters: A list of `WidgetUpdater`s that react to `"global_step"`.
        _locators: Data accessors keyed by name for the parser.
    """

    def __init__(
        self,
        plotter: Plotter,
        data_parser: DataParser,
        ycb_loader: YCBMeshLoader,
        step_mapper: StepMapper,
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.ycb_loader = ycb_loader
        self.step_mapper = step_mapper

        # Track current episode to avoid reloading mesh unnecessarily
        self.current_episode: int | None = None

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("global_step", required=True),
                ],
                callback=self.update_mesh,
            ),
            WidgetUpdater(
                topics=[
                    TopicSpec("global_step", required=True),
                    EventSpec("KeyPressed", "KeyPressEvent", required=False),
                ],
                callback=self.update_agent,
            ),
            WidgetUpdater(
                topics=[
                    EventSpec("KeyPressed", "KeyPressEvent", required=True),
                ],
                callback=self.update_transparency,
            ),
        ]
        self._locators = self.create_locators()

        self.gaze_line: Line | None = None
        self.agent_sphere: Sphere | None = None
        self.text_label: Text2D = Text2D(
            txt="Ground Truth", pos="top-center", font=FONT
        )

        # Path visibility flags
        self.mesh_transparency: float = 0.0
        self.show_patch_past: bool = False

        # Path geometry
        self.patch_past_spheres: list[Sphere] = []
        self.patch_past_line: Line | None = None

        self.plotter.at(SIMULATOR_IX).add(self.text_label)

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

        locators["agent_location"] = DataLocator(
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
                DataLocatorStep.key(name="sm", value="patch_0"),
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
            self.plotter.at(SIMULATOR_IX).remove(widget)
            self.plotter.at(SIMULATOR_IX).render()

    def update_mesh(self, widget: Mesh, msgs: list[TopicMessage]) -> tuple[Mesh, bool]:
        """Update the target mesh when the episode changes.

        Removes any existing mesh, loads the episode's primary target object,
        applies its Euler rotations, scales and positions it, then adds it to
        the plotter. The mesh is only reloaded if the episode has changed.

        Args:
            widget: The currently displayed mesh, if any.
            msgs: Messages received from the `WidgetUpdater`.

        Returns:
            A tuple `(mesh, False)`. The second value is `False` to indicate
            that no publish should occur.
        """
        msgs_dict = {msg.name: msg.value for msg in msgs}
        global_step = msgs_dict["global_step"]
        episode_number, _ = self.step_mapper.global_to_local(global_step)

        # Only reload the mesh if the episode has changed
        if self.current_episode == episode_number:
            return widget, False

        self.current_episode = episode_number
        self.remove(widget)

        locator = self._locators["target"]
        target = self.data_parser.extract(locator, episode=str(episode_number))
        target_id = target["primary_target_object"]
        target_rot = target["primary_target_rotation_quat"]
        target_pos = target["primary_target_position"]

        try:
            widget = self.ycb_loader.create_mesh(target_id).clone(deep=True)
        except FileNotFoundError:
            return widget, False

        rot = Rotation.from_quat(np.array(target_rot), scalar_first=True)
        rot_euler = rot.as_euler("xyz", degrees=True)
        widget.rotate_x(rot_euler[0])
        widget.rotate_y(rot_euler[1])
        widget.rotate_z(rot_euler[2])
        widget.shift(*target_pos)
        widget.alpha(1.0 - self.mesh_transparency)

        self.plotter.at(SIMULATOR_IX).add(widget)

        return widget, False

    def update_agent(self, widget: None, msgs: list[TopicMessage]) -> tuple[None, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}
        global_step = msgs_dict["global_step"]
        episode_number, step_number = self.step_mapper.global_to_local(global_step)

        steps_mask = self.data_parser.extract(
            self._locators["steps_mask"], episode=str(episode_number)
        )
        mapping = np.flatnonzero(steps_mask)

        agent_pos = self.data_parser.extract(
            self._locators["agent_location"],
            episode=str(episode_number),
            sm_step=max(0, int(mapping[step_number]) - 1),
        )

        patch_pos = self.data_parser.extract(
            self._locators["patch_location"],
            episode=str(episode_number),
            step=step_number,
        )

        if self.agent_sphere is None:
            self.agent_sphere = Sphere(
                pos=agent_pos,
                r=0.004,
                c=COLOR_PALETTE["Secondary"],
            )

            self.plotter.at(SIMULATOR_IX).add(self.agent_sphere)
        self.agent_sphere.pos(agent_pos)

        if self.gaze_line is None:
            self.gaze_line = Line(
                agent_pos, patch_pos, c=COLOR_PALETTE["Accent3"], lw=4
            )
            self.plotter.at(SIMULATOR_IX).add(self.gaze_line)
        self.gaze_line.points = [agent_pos, patch_pos]

        # Handle keypress to toggle patch path visibility
        self._clear_patch_paths()
        key_event = msgs_dict.get("KeyPressEvent", None)
        if key_event is not None and getattr(key_event, "at", None) == SIMULATOR_IX:
            key = getattr(key_event, "keypress", None)
            if key == "s":
                self.show_patch_past = not self.show_patch_past

        # Expire the event so it only affects this call
        self.updaters[1].expire_topic("KeyPressEvent")

        # Get total steps for this episode
        num_steps = len(
            self.data_parser.query(
                self._locators["patch_location"], episode=str(episode_number)
            )
        )

        if self.show_patch_past:
            self._rebuild_patch_paths(episode_number, num_steps, step_number)

        return widget, False

    def _clear_patch_paths(self) -> None:
        """Clear all patch path geometry from the plotter."""
        for s in self.patch_past_spheres:
            self.plotter.at(SIMULATOR_IX).remove(s)
        self.patch_past_spheres.clear()

        if self.patch_past_line is not None:
            self.plotter.at(SIMULATOR_IX).remove(self.patch_past_line)
            self.patch_past_line = None

    def _rebuild_patch_paths(
        self,
        episode_number: int,
        num_steps: int,
        curr_idx: int,
    ) -> None:
        """Rebuild past patch (sensor) path with black spheres and lines.

        Args:
            episode_number: Current episode number.
            num_steps: Total number of steps in the episode.
            curr_idx: Current step index.
        """
        patch_positions: list[np.ndarray] = []
        for k in range(min(curr_idx + 1, num_steps)):
            pos = self.data_parser.extract(
                self._locators["patch_location"],
                episode=str(episode_number),
                step=k,
            )
            patch_positions.append(np.array(pos))

        if patch_positions:
            for p in patch_positions:
                s = Sphere(pos=p, r=0.002, c="black")
                self.plotter.at(SIMULATOR_IX).add(s)
                self.patch_past_spheres.append(s)
            if len(patch_positions) >= 2:
                self.patch_past_line = Line(patch_positions, c="black", lw=1)
                self.plotter.at(SIMULATOR_IX).add(self.patch_past_line)

    def update_transparency(
        self, widget: None, msgs: list[TopicMessage]
    ) -> tuple[None, bool]:
        msgs_dict = {msg.name: msg.value for msg in msgs}

        key_event = msgs_dict.get("KeyPressEvent", None)
        if key_event is not None and getattr(key_event, "at", None) == SIMULATOR_IX:
            key = getattr(key_event, "keypress", None)

            if key == "Left":
                self.mesh_transparency -= 0.5
            elif key == "Right":
                self.mesh_transparency += 0.5

        self.mesh_transparency = float(np.clip(self.mesh_transparency, 0.0, 1.0))
        if widget is not None:
            widget.alpha(1.0 - self.mesh_transparency)

        self.updaters[2].expire_topic("KeyPressEvent")

        return widget, False


class ModelViewerWidgetOps:
    """WidgetOps implementation for displaying a colored pretrained model pointcloud.

    This widget shows a colored pointcloud of a pretrained model. By default, it
    displays the model at `current_mlh` for the given `lm_id` at the current step.
    Users can cycle through available objects using "m" and "M" keys, reset
    back to showing `current_mlh` by pressing ".", and cycle through available
    input channels using "c" and "C" keys.

    Attributes:
        plotter: A `vedo.Plotter` used to add and remove actors.
        data_parser: A parser that extracts entries from the JSON log.
        models_loader: Loader for pretrained model pointclouds.
        renderer_id: The renderer index where this widget displays.
        lm_id: The learning module ID (0 or 1).
        steps_mapper: Mapper class to convert between steps in different timescales.
        mode: Current display mode, either "mlh" or "manual".
        manual_object_index: Current index in the list of available objects when
            in manual mode.
        channel_index: Current index in the list of available channels.
        available_objects: List of graph_ids available for the given lm_id.
        available_channels: List of input channels available for the given lm_id.
        current_pointcloud: The currently displayed pointcloud, if any.
    """

    def __init__(
        self,
        renderer_id: int,
        lm_id: int,
        plotter: Plotter,
        data_parser: DataParser,
        models_loader: PretrainedModelsLoader,
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.models_loader = models_loader
        self.renderer_id = renderer_id
        self.lm_id = lm_id

        # Mode tracking: "mlh" shows current_mlh, "manual" shows manually selected
        self.mode: str = "mlh"
        self.manual_object_index: int = 0

        # Get all available objects for this lm_id
        self.available_objects: list[str] = self.models_loader.query(lm_id=lm_id)
        self.available_channels: list[str] = self.models_loader.query(
            lm_id=lm_id, graph_id=self.available_objects[0]
        )

        # Channel tracking
        self.channel_index: int = 0
        self._channel = (
            self.available_channels[0] if self.available_channels else "patch"
        )

        # Current pointcloud actor
        self.current_pointcloud: Points | None = None

        # Current legend image (for H-LLM object_id coloring)
        self.current_legend: vedo.visual.Actor2D | None = None

        # LM key for data locators
        self._lm_key = f"LM_{lm_id}"

        self._locators = self._create_locators()

        self.updaters = [
            WidgetUpdater(
                topics=[
                    EventSpec("KeyPressed", "KeyPressEvent", required=True),
                ],
                callback=self.handle_keypress,
            ),
        ]

        # Add label for this viewer
        label_text = "H-LLM Model" if lm_id == 1 else "L-LLM Model"
        self.text_label = Text2D(txt=label_text, pos="top-center", font=FONT)
        self.plotter.at(self.renderer_id).add(self.text_label)

        # Add graph_id label in bottom-right
        self.graph_id_label = Text2D(txt="", pos="bottom-left", font=FONT, s=0.5)
        self.plotter.at(self.renderer_id).add(self.graph_id_label)

        # Add input channel label in bottom-right
        self.channel_label = Text2D(
            txt=self._channel, pos="bottom-right", font=FONT, s=0.5
        )
        self.plotter.at(self.renderer_id).add(self.channel_label)

    def add(self, callback: Callable) -> None:
        widget = self._create_pointcloud()
        self.plotter.at(self.renderer_id).add(widget)
        return widget

    def _create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this widget.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}

        # Locator for current_mlh
        locators["mlh"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value=self._lm_key),
                DataLocatorStep.key(name="telemetry", value="current_mlh"),
                DataLocatorStep.index(name="step"),
            ]
        )

        # Locator for target rotation/position
        locators["target"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="target"),
            ]
        )

        return locators

    def _clear_pointcloud(self) -> None:
        """Remove the current pointcloud and legend from the plotter."""
        if self.current_pointcloud is not None:
            self.plotter.at(self.renderer_id).remove(self.current_pointcloud)
            self.current_pointcloud = None
        if self.current_legend is not None:
            self.plotter.at(self.renderer_id).remove(self.current_legend)
            self.current_legend = None

    def _clear_labels(self) -> None:
        """Remove the current labels from the plotter."""
        self.graph_id_label.text("")
        self.channel_label.text("")

    def _create_legend_image(
        self, legend_data: dict[str, tuple[str, str]]
    ) -> vedo.visual.Actor2D | None:
        """Create a matplotlib legend as a fixed 2D overlay.

        Args:
            legend_data: Dictionary mapping graph_id -> (hash_str, color_hex).

        Returns:
            A vedo Actor2D containing the rendered legend, or None if no data.
        """
        if not legend_data:
            return None

        # Create figure with matplotlib
        fig, ax = plt.subplots(figsize=(1.5, 0.3 * len(legend_data)), dpi=100)
        ax.axis("off")

        # Create legend handles
        handles = []
        labels = []
        for gid, (_, color_hex) in legend_data.items():
            handle = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_hex,
                markersize=8,
                linestyle="None",
            )
            handles.append(handle)
            labels.append(gid)

        ax.legend(handles, labels, loc="center", frameon=False, fontsize=8, ncol=1)

        plt.tight_layout(pad=0.1)

        # Create vedo Image and convert to 2D overlay (fixed screen position)
        image = Image(fig)
        plt.close(fig)

        # clone2d creates a screen-space Actor2D with pos in [0,1] range
        return image.clone2d(pos=(0.0, 0.0), size=0.7)

    def _create_pointcloud(self) -> Points | None:
        """Display a pointcloud models.

        Returns:
            The created Points widget.
        """
        # Determine which object to show
        graph_id = self.available_objects[self.manual_object_index]
        channel = self.available_channels[self.channel_index]

        # Create and add the pointcloud
        points = self.models_loader.create_model(
            graph_id=graph_id,
            lm_id=self.lm_id,
            input_channel=channel,
            color=True,
        )
        # Extract legend_data before cloning (clone doesn't copy custom attributes)
        legend_data = getattr(points, "legend_data", None)
        widget = points.clone(deep=True)

        self.current_pointcloud = widget
        self.graph_id_label.text(graph_id)
        self.channel_label.text(self._channel)

        # Create legend if legend_data is available
        if legend_data:
            self.current_legend = self._create_legend_image(legend_data)
            if self.current_legend:
                self.plotter.at(self.renderer_id).add(self.current_legend)

        return widget

    def update_model(
        self, widget: Points | None, msgs: list[TopicMessage]
    ) -> tuple[Points | None, bool]:
        """Update the model display based on current episode and step.

        When in "mlh" mode, shows the model at current_mlh.
        When in "manual" mode, shows the manually selected object.

        Args:
            widget: The currently displayed pointcloud, if any.
            msgs: Messages received from the WidgetUpdater.

        Returns:
            A tuple (pointcloud, False). False indicates no publish.
        """
        self._clear_pointcloud()
        self._clear_labels()
        msgs_dict = {msg.name: msg.value for msg in msgs}

        episode_number = msgs_dict["episode_number"]
        step_number = msgs_dict["step_number"]

        widget = self._create_pointcloud(episode_number, step_number)
        return widget, False

    def handle_keypress(
        self, widget: Points | None, msgs: list[TopicMessage]
    ) -> tuple[Points | None, bool]:
        """Handle keypress events for cycling through objects and channels.

        "m" and "M" cycle through available objects (switches to manual mode).
        "." resets back to showing current_mlh.
        "c" and "C" cycle through available input channels.

        Args:
            widget: The currently displayed pointcloud, if any.
            msgs: Messages received from the WidgetUpdater.

        Returns:
            A tuple (pointcloud, False). False indicates no publish.
        """
        # Expire the event from the WidgetUpdater inbox
        self.updaters[0].expire_topic("KeyPressEvent")

        msgs_dict = {msg.name: msg.value for msg in msgs}

        key_event = msgs_dict.get("KeyPressEvent", None)
        if key_event is None:
            return widget, False

        # Only respond to events on our renderer
        if getattr(key_event, "at", None) != self.renderer_id:
            return widget, False

        key = key_event["keypress"]
        if key == "m":
            self.manual_object_index = (self.manual_object_index - 1) % len(
                self.available_objects
            )
        elif key == "M":
            self.manual_object_index = (self.manual_object_index + 1) % len(
                self.available_objects
            )
        elif key == "c":
            self.channel_index = (self.channel_index - 1) % len(self.available_channels)
        elif key == "C":
            self.channel_index = (self.channel_index + 1) % len(self.available_channels)

        self._clear_pointcloud()
        self._clear_labels()

        widget = self._create_pointcloud()
        self.plotter.at(self.renderer_id).add(widget)
        return widget, False

    def remove(self, widget: Points | None) -> None:
        """Remove the pointcloud widget.

        Args:
            widget: The pointcloud widget to remove.
        """
        self._clear_pointcloud()
        self._clear_labels()
        self.plotter.at(self.renderer_id).render()


class EvidenceViewerWidgetOps:
    """WidgetOps implementation for displaying evidence scores over time as a lineplot.

    This widget shows a seaborn lineplot of maximum evidence scores for each object
    across all episodes (global timeline) for a specific Learning Module (LM). Episodes
    are concatenated back-to-back to form a single continuous plot. A vertical black
    line marks the current global step position.

    Attributes:
        plotter: A `vedo.Plotter` used to add and remove actors.
        data_parser: A parser that extracts entries from the JSON log.
        renderer_id: The renderer index where this widget displays.
        lm_id: The learning module ID (0 for L-LLM, 1 for H-LLM).
        step_mapper: Mapper class to convert between local and global step indices.
        current_image: The currently displayed vedo Image, if any.
    """

    # TBP Palette colors for objects
    OBJECT_COLORS = [
        Palette.as_hex("numenta_blue"),
        Palette.as_hex("pink"),
        Palette.as_hex("purple"),
        Palette.as_hex("green"),
        Palette.as_hex("gold"),
        Palette.as_hex("indigo"),
        Palette.as_hex("bossanova"),
        Palette.as_hex("vivid_violet"),
        Palette.as_hex("blue_violet"),
        Palette.as_hex("amethyst"),
    ]

    def __init__(
        self,
        renderer_id: int,
        lm_id: int,
        plotter: Plotter,
        data_parser: DataParser,
        step_mapper: StepMapper,
        scale: float = 0.003,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.plotter = plotter
        self.data_parser = data_parser
        self.renderer_id = renderer_id
        self.lm_id = lm_id
        self.step_mapper = step_mapper
        self.scale = scale
        self.pos = pos

        # LM key for data locators
        self._lm_key = f"LM_{lm_id}"

        # Current image actor
        self.current_image: Image | None = None

        # Cache for evidence data across all episodes (global timeline)
        self._data_loaded: bool = False
        self._cached_max_evidence: dict[str, list[float]] = {}
        self._cached_objects: list[str] = []
        self._cached_bursts: list[bool] = []

        self._locators = self._create_locators()

        self.updaters = [
            WidgetUpdater(
                topics=[
                    TopicSpec("global_step", required=True),
                ],
                callback=self.update_plot,
            ),
        ]

    def _create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used by this widget.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}

        locators["evidences"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value=self._lm_key),
                DataLocatorStep.key(name="telemetry", value="evidences"),
                DataLocatorStep.index(name="step"),
            ]
        )

        # Locator for querying evidence via hypotheses_updater_telemetry
        locators["hyp_space"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value=self._lm_key),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(name="obj"),
                DataLocatorStep.key(name="channel", value="patch_0"),
                DataLocatorStep.key(name="telemetry", value="evidence"),
            ]
        )

        # Locator for querying added_ids (burst detection)
        locators["added_ids"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value=self._lm_key),
                DataLocatorStep.key(
                    name="telemetry", value="hypotheses_updater_telemetry"
                ),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(name="obj"),
                DataLocatorStep.key(name="channel", value="patch_0"),
                DataLocatorStep.key(name="telemetry2", value="hypotheses_updater"),
                DataLocatorStep.key(name="telemetry3", value="added_ids"),
            ]
        )

        return locators

    def _load_all_evidence_data(self) -> None:
        """Load and cache evidence data for all episodes.

        Extracts evidence from hypotheses_updater_telemetry for all episodes and
        computes the maximum evidence for each object at each global step. Episodes
        are concatenated back-to-back to form a global timeline. Also loads burst
        data (steps where new hypotheses were added).
        """
        if self._data_loaded:
            return

        # Initialize storage
        self._cached_max_evidence = {}
        self._cached_objects = []
        self._cached_bursts = []

        # Iterate over all episodes
        for episode in range(self.step_mapper.num_episodes):
            # Query available steps for this episode
            steps = self.data_parser.query(
                self._locators["hyp_space"], episode=str(episode)
            )

            if not steps:
                continue

            # Get object names from first step of first episode
            if not self._cached_objects:
                objects_list = self.data_parser.query(
                    self._locators["hyp_space"], episode=str(episode), step=0
                )
                # Sort objects to ensure consistent ordering with ModelViewer legend
                self._cached_objects = sorted(objects_list)
                self._cached_max_evidence = {obj: [] for obj in self._cached_objects}

            # Append evidence data for this episode to the global timeline
            for step in steps:
                step_had_burst = False
                for obj in self._cached_objects:
                    evidence_list = self.data_parser.extract(
                        self._locators["hyp_space"],
                        episode=str(episode),
                        step=step,
                        obj=obj,
                    )
                    # Get maximum evidence for this object at this step
                    max_evidence = max(evidence_list) if evidence_list else 0.0
                    self._cached_max_evidence[obj].append(max_evidence)

                    # Check if this object had a burst at this step
                    added_ids = self.data_parser.extract(
                        self._locators["added_ids"],
                        episode=str(episode),
                        step=step,
                        obj=obj,
                    )
                    if added_ids and len(added_ids) > 0:
                        step_had_burst = True

                self._cached_bursts.append(step_had_burst)

        self._data_loaded = True

    def _create_lineplot_image(self, global_step: int) -> Image | None:
        """Create a seaborn lineplot as a vedo Image.

        Args:
            global_step: The current global step index (0-indexed, across all
                episodes).

        Returns:
            A vedo Image containing the rendered lineplot, or None if no data.
        """
        if not self._cached_max_evidence:
            return None

        # Create figure with matplotlib/seaborn
        fig, ax = plt.subplots(1, 1, figsize=(14, 3), dpi=200)

        # Plot each object's evidence as a line
        for i, obj in enumerate(self._cached_objects):
            color = self.OBJECT_COLORS[i % len(self.OBJECT_COLORS)]
            steps = list(range(len(self._cached_max_evidence[obj])))
            ax.plot(
                steps,
                self._cached_max_evidence[obj],
                color=color,
                linewidth=1.5,
                label=obj,
            )

        # Burst locations (faint red lines for steps with added hypotheses)
        bursts = np.array(self._cached_bursts)
        add_idx = np.flatnonzero(bursts)
        if add_idx.size > 0:
            ymin, ymax = ax.get_ylim()
            ax.vlines(
                add_idx,
                ymin,
                ymax,
                colors="#FF0000",
                linestyles="-",
                alpha=0.3,
                linewidth=1.0,
                zorder=1,
                label="Burst",
            )

        # Draw vertical line at current global step
        ax.axvline(x=global_step, color="black", linewidth=2, linestyle="-")

        # Set title based on LM
        title_text = "H-LLM Evidence" if self.lm_id == 1 else "L-LLM Evidence"
        ax.set_title(title_text, fontsize=12)

        # Style the plot
        ax.set_xlabel("Global Step", fontsize=10)
        ax.set_ylabel("Evidence", fontsize=10)
        ax.set_xlim(
            0, max(1, len(self._cached_max_evidence[self._cached_objects[0]]) - 1)
        )

        # Set y-axis limits based on data
        all_values = [
            v for obj_vals in self._cached_max_evidence.values() for v in obj_vals
        ]
        if all_values:
            y_min = min(0, *all_values)
            y_max = max(all_values) * 1.1 if max(all_values) > 0 else 1
            ax.set_ylim(y_min, y_max)

        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, alpha=0.3)

        # Add legend with smaller font if not too many objects
        if len(self._cached_objects) <= 6:
            ax.legend(fontsize=6, loc="best")

        plt.tight_layout()

        # Create vedo Image directly from matplotlib figure
        image = Image(fig)
        plt.close(fig)

        return image.clone2d(pos=(0.08, 0.1), size=0.5)

    def _clear_image(self) -> None:
        """Remove the current image from the plotter."""
        if self.current_image is not None:
            self.plotter.at(self.renderer_id).remove(self.current_image)
            self.current_image = None

    def update_plot(
        self, widget: Image | None, msgs: list[TopicMessage]
    ) -> tuple[Image | None, bool]:
        """Update the evidence plot based on current global step.

        Args:
            widget: The currently displayed image, if any.
            msgs: Messages received from the WidgetUpdater.

        Returns:
            A tuple (image, False). False indicates no publish.
        """
        self._clear_image()
        msgs_dict = {msg.name: msg.value for msg in msgs}

        global_step = msgs_dict["global_step"]

        # Load evidence data for all episodes (cached)
        self._load_all_evidence_data()

        # Create the lineplot image with global step
        image = self._create_lineplot_image(global_step)

        if image is not None:
            self.plotter.at(self.renderer_id).add(image)
            self.current_image = image

        return image, False

    def remove(self, widget: Image | None) -> None:
        """Remove the image widget.

        Args:
            widget: The image widget to remove.
        """
        self._clear_image()
        self.plotter.at(self.renderer_id).render()


class InteractiveCompositionalPlot:
    """An interactive plot with 5 sub-renderers for compositional visualization.

    This visualization features:
    - Main renderer (full window background) for primary content
    - Five corner overlay renderers for auxiliary views:
        - Main-left: Overlay view 1
        - Top-left: Overlay view 2
        - Top-right: Overlay view 3
        - Bottom-left: Overlay view 4
        - Bottom-right: Overlay view 5

    Args:
        exp_path: Path to the experiment log directory.
        data_path: Root directory containing object meshes.

    Attributes:
        data_parser: Parser that reads the JSON log file.
        ycb_loader: Loader for YCB meshes.
        event_bus: Publisher for routing TopicMessage events.
        plotter: Vedo Plotter hosting all widgets.
        scheduler: Debounce scheduler bound to the plotter interactor.
        scope_viewer: Manages widget visibility by scope.
        _widgets: Mapping of widget names to Widget instances.
    """

    def __init__(
        self,
        exp_path: str,
        data_path: str,
        models_path: str,
    ):
        renderer_areas = [
            {"bottomleft": (0.0, 0.0), "topright": (1.0, 1.0)},  # Main renderer
            {"bottomleft": (0.05, 0.5), "topright": (0.49, 0.9)},  # Simulator Overlay
            {"bottomleft": (0.51, 0.5), "topright": (0.95, 0.9)},  # Pretrained Model
        ]

        self.axes_dict = {
            "xrange": (-0.1, 0.1),
            "yrange": (1.4, 1.6),
            "zrange": (-0.1, 0.1),
        }

        self.cam_dict = {
            "pos": (0.0, 0.0, 0.5),
            "focal_point": (0.0, 0.0, 0.0),
        }

        # Initialize data sources
        self.data_parser = DataParser(exp_path)
        self.ycb_loader = YCBMeshLoader(
            data_path, mesh_subpath="textured.glb", rotate_obj=False
        )
        self.models_loader = PretrainedModelsLoader(models_path)
        self.step_mapper = StepMapper(self.data_parser)
        self.animator = None

        # Initialize event bus and plotter
        self.event_bus = Publisher()
        self.plotter = Plotter(
            shape=renderer_areas,
            size=(1400, 1000),
            sharecam=False,
        ).render()

        self.scheduler = VtkDebounceScheduler(self.plotter.interactor, period_ms=33)

        # Create and add widgets
        self._widgets = self._create_widgets()
        for w in self._widgets.values():
            w.add()

        # Initialize widget states
        self._widgets["step_slider"].set_state(0)

        # Setup scope viewer for visibility management
        self.scope_viewer = ScopeViewer(self.plotter, self._widgets)

        # Configure and show renderers
        self._setup_renderers()

    def _create_widgets(self) -> dict[str, Widget]:
        """Create all widgets for the plot.

        Returns:
            Dictionary mapping widget names to Widget instances.
        """
        widgets = {}

        widgets["click_widget"] = Widget[None, None](
            widget_ops=ClickWidgetOps(
                plotter=self.plotter, cam_dict=deepcopy(self.cam_dict)
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.1,
            dedupe=True,
        )

        widgets["step_slider"] = Widget[Slider2D, int](
            widget_ops=StepSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                step_mapper=self.step_mapper,
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.1,
            dedupe=True,
        )

        widgets["primary_mesh"] = Widget[Mesh, None](
            widget_ops=GtMeshWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
                ycb_loader=self.ycb_loader,
                step_mapper=self.step_mapper,
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["model_viewer"] = Widget[Points, None](
            widget_ops=ModelViewerWidgetOps(
                renderer_id=MODEL_IX,
                lm_id=1,
                plotter=self.plotter,
                data_parser=self.data_parser,
                models_loader=self.models_loader,
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        widgets["evidence_viewer"] = Widget[Image, None](
            widget_ops=EvidenceViewerWidgetOps(
                renderer_id=MAIN_RENDERER_IX,
                lm_id=0,
                plotter=self.plotter,
                data_parser=self.data_parser,
                step_mapper=self.step_mapper,
                scale=0.5,
                pos=(-400, -150, 0),
            ),
            scopes=[],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        return widgets

    def _setup_renderers(self) -> None:
        """Configure and show all renderers."""
        for renderer_ix in [
            SIMULATOR_IX,
            MODEL_IX,
        ]:
            self.plotter.at(renderer_ix).show(
                axes=deepcopy(self.axes_dict),
                interactive=False,
                resetcam=True,
            )

        # Add keypress callback
        self.plotter.add_callback("KeyPress", self._on_keypress)

        # Final show call with interactive=True
        self.plotter.at(MAIN_RENDERER_IX).show(
            axes=0,
            camera=deepcopy(self.cam_dict),
            interactive=True,
            resetcam=False,
        )

    def create_step_animator(self):
        step_slider = self._widgets["step_slider"]
        slider_current_value = step_slider.widget_ops.extract_state(
            widget=step_slider.widget
        )
        slider_max_value = int(step_slider.widget.range[1])

        step_actions = make_slider_step_actions_for_widget(
            widget=step_slider,
            start_value=slider_current_value,
            stop_value=slider_max_value,
            num_steps=slider_max_value - slider_current_value + 1,
            step_dt=0.5,
        )

        return WidgetAnimator(
            scheduler=self.scheduler,
            actions=step_actions,
            key_prefix="step_animator",
        )

    def _on_keypress(self, event):
        """Handle keypress events.

        Args:
            event: The keypress event from VTK.
        """
        key = getattr(event, "keypress", None)
        if key is None:
            return

        if key.lower() == "q":
            self.plotter.interactor.ExitCallback()

        if hasattr(self, "animator") and event.at == 0:
            if key == "a":
                if self.animator is not None:
                    self.animator.stop()
                self.animator = self.create_step_animator()
                self.animator.start()

            elif key == "s":
                if self.animator is not None:
                    self.animator.stop()


@register(
    "interactive_pretraining_compositional_plot",
    description="Interactive compositional visualization with 2 renderers",
)
def main(
    experiment_log_dir: str, objects_mesh_dir: str, pretrained_models_file: str
) -> int:
    """Interactive compositional plot with 5 renderers.

    This visualization provides a multi-pane view for compositional analysis
    with a main renderer and four corner overlay renderers.

    Args:
        experiment_log_dir: Path to the experiment directory containing the
            detailed stats file.
        objects_mesh_dir: Path to the root directory of object meshes.
        pretrained_models_file: Path to the pretrained models pt file.

    Returns:
        Exit code.
    """
    vedo.settings.enable_default_keyboard_callbacks = False

    if not Path(experiment_log_dir).exists():
        logger.error(f"Experiment path not found: {experiment_log_dir}")
        return 1

    data_path = str(Path(objects_mesh_dir).expanduser())
    models_path = str(Path(pretrained_models_file).expanduser())

    InteractiveCompositionalPlot(experiment_log_dir, data_path, models_path)

    return 0


@attach_args("interactive_pretraining_compositional_plot")
def add_arguments(p: argparse.ArgumentParser) -> None:
    """Add command-line arguments for the interactive compositional plot.

    Args:
        p: The argument parser to add arguments to.
    """
    p.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
    p.add_argument(
        "--objects_mesh_dir",
        default="~/tbp/data/compositional_objects/meshes",
        help="The directory containing the mesh objects.",
    )

    p.add_argument(
        "--pretrained_models_file",
        # default="~/tbp/results/monty/pretrained_models/my_trained_models/supervised_pre_training_objects_with_logos_lvl1_comp_models_resampling/pretrained/model.pt",
        default="~/tbp/results/monty/projects/evidence_eval_runs/logs/supervised_pre_training_objects_with_logos_lvl1_comp_models_burst_sampling/pretrained/model.pt",
        help=("The file containing the pretrained models."),
    )
