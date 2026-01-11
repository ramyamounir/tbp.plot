# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import vedo
from pubsub.core import Publisher
from scipy.spatial.transform import Rotation
from vedo import (
    Image,
    Line,
    Mesh,
    Plotter,
    Slider2D,
    Sphere,
    Text2D,
    settings,
)

from tbp.interactive.colors import Palette
from tbp.interactive.data import (
    DataLocator,
    DataLocatorStep,
    DataParser,
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
HLLM_MODEL_IX = 2
HLLM_PREDICTION_IX = 3
LLLM_MODEL_IX = 4
LLLM_PREDICTION_IX = 5


class StepSliderWidgetOps:
    """WidgetOps implementation for a Step slider.

    This class adds a slider widget for navigating through timesteps.
    It publishes changes as `TopicMessage` items under the "step_number" topic.

    Attributes:
        plotter: A `vedo.Plotter` object to add or remove the slider and render.
        data_parser: A parser that extracts or queries information from the
            JSON log file.
        _add_kwargs: Default keyword arguments passed to `plotter.add_slider`.
        _locators: Data accessors keyed by name that instruct the `DataParser`
            how to retrieve the required information.
    """

    def __init__(
        self,
        plotter: Plotter,
        data_parser: DataParser,
    ) -> None:
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
            "pos": [(0.15, 0.06), (0.85, 0.06)],
            "title": "Step",
            "font": FONT,
            "show_value": True,
        }

        self._locators = self._create_locators()

    def _create_locators(self) -> dict[str, DataLocator]:
        """Create and return data locators used to access step information.

        Returns:
            A dictionary containing the created locators.
        """
        locators = {}
        locators["step"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="lm", value="LM_0"),
                DataLocatorStep.key(name="telemetry", value="time"),
                DataLocatorStep.index(name="step"),
            ]
        )
        return locators

    def add(self, callback: Callable) -> Slider2D:
        """Create the slider widget and set its range from the data.

        Args:
            callback: Function called with `(widget, event)` when the slider
                changes in the UI.

        Returns:
            The created widget as returned by the plotter.
        """
        widget = self.plotter.at(MAIN_RENDERER_IX).add_slider(
            callback, **self._add_kwargs
        )
        self.plotter.at(MAIN_RENDERER_IX).render()
        return widget

    def remove(self, widget: Slider2D) -> None:
        """Remove the slider widget and re-render.

        Args:
            widget: The widget object.
        """
        self.plotter.at(MAIN_RENDERER_IX).remove(widget)
        self.plotter.at(MAIN_RENDERER_IX).render()

    def extract_state(self, widget: Slider2D) -> int:
        """Read the current slider value.

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
            value: Desired step index.
        """
        set_slider_state(widget, value)

    def state_to_messages(self, state: int) -> Iterable[TopicMessage]:
        """Convert the slider state to pubsub messages.

        Args:
            state: Selected step index.

        Returns:
            A list with a single `TopicMessage` named `"step_number"`.
        """
        return [TopicMessage(name="step_number", value=state)]

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
                topics=[
                    TopicSpec("episode_number", required=True),
                ],
                callback=self.update_mesh,
            ),
            WidgetUpdater(
                topics=[
                    TopicSpec("episode_number", required=True),
                    TopicSpec("step_number", required=True),
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

        # SM rgb images
        self.sm0_image = Image | None
        self.sm1_image = Image | None
        self.sm0_label: Text2D = Text2D(txt="Sensor 0", pos=(0.085, 0.92), font=FONT)
        self.sm1_label: Text2D = Text2D(txt="Sensor 1", pos=(0.185, 0.92), font=FONT)

        # Path visibility flags
        self.mesh_transparency: float = 0.0
        self.show_agent_past: bool = False
        self.show_agent_future: bool = False
        self.show_patch_past: bool = False
        self.show_patch_future: bool = False

        # Path geometry
        self.agent_past_spheres: list[Sphere] = []
        self.agent_past_line: Line | None = None
        self.agent_future_spheres: list[Sphere] = []
        self.agent_future_line: Line | None = None

        self.patch_past_spheres: list[Sphere] = []
        self.patch_past_line: Line | None = None
        self.patch_future_spheres: list[Sphere] = []
        self.patch_future_line: Line | None = None

        self.plotter.at(SIMULATOR_IX).add(self.text_label)
        self.plotter.at(MAIN_RENDERER_IX).add(self.sm0_label)
        self.plotter.at(MAIN_RENDERER_IX).add(self.sm1_label)

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
                DataLocatorStep.key(name="system"),
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
                DataLocatorStep.key(name="system"),
                DataLocatorStep.key(name="telemetry", value="processed_observations"),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(name="telemetry_type", value="location"),
            ]
        )

        locators["patch_rgb"] = DataLocator(
            path=[
                DataLocatorStep.key(name="episode"),
                DataLocatorStep.key(name="system"),
                DataLocatorStep.key(name="telemetry", value="raw_observations"),
                DataLocatorStep.index(name="step"),
                DataLocatorStep.key(name="telemetry_type", value="rgba"),
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

    def _clear_agent_paths(self) -> None:
        for s in self.agent_past_spheres:
            self.plotter.at(SIMULATOR_IX).remove(s)
        for s in self.agent_future_spheres:
            self.plotter.at(SIMULATOR_IX).remove(s)

        self.agent_past_spheres.clear()
        self.agent_future_spheres.clear()

        if self.agent_past_line is not None:
            self.plotter.at(SIMULATOR_IX).remove(self.agent_past_line)
            self.agent_past_line = None
        if self.agent_future_line is not None:
            self.plotter.at(SIMULATOR_IX).remove(self.agent_future_line)
            self.agent_future_line = None

    def _clear_patch_paths(self) -> None:
        for s in self.patch_past_spheres:
            self.plotter.at(SIMULATOR_IX).remove(s)
        for s in self.patch_future_spheres:
            self.plotter.at(SIMULATOR_IX).remove(s)

        self.patch_past_spheres.clear()
        self.patch_future_spheres.clear()

        if self.patch_past_line is not None:
            self.plotter.at(SIMULATOR_IX).remove(self.patch_past_line)
            self.patch_past_line = None
        if self.patch_future_line is not None:
            self.plotter.at(SIMULATOR_IX).remove(self.patch_future_line)
            self.patch_future_line = None

    def _clear_all_paths(self) -> None:
        self._clear_agent_paths()
        self._clear_patch_paths()

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
        episode_number = msgs_dict["episode_number"]
        step_number = msgs_dict["step_number"]

        steps_mask_lm0 = self.data_parser.extract(
            self._locators["steps_mask"], system="LM_0", episode=str(episode_number)
        )
        mapping_lm0 = np.flatnonzero(steps_mask_lm0)

        steps_mask_lm1 = self.data_parser.extract(
            self._locators["steps_mask"], system="LM_1", episode=str(episode_number)
        )
        mapping_lm1 = np.flatnonzero(steps_mask_lm1)

        agent_pos = self.data_parser.extract(
            self._locators["agent_location"],
            episode=str(episode_number),
            sm_step=max(0, int(mapping_lm0[step_number]) - 1),
        )

        patch_pos = self.data_parser.extract(
            self._locators["patch_location"],
            episode=str(episode_number),
            system="SM_0",
            step=int(mapping_lm0[step_number]),
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

        # Clear Images
        if self.sm0_image is not None:
            self.plotter.at(MAIN_RENDERER_IX).remove(self.sm0_image)
            self.sm0_image = None
        if self.sm1_image is not None:
            self.plotter.at(MAIN_RENDERER_IX).remove(self.sm1_image)
            self.sm1_image = None

        # Plot Images
        patch_rgb_lm0 = self.data_parser.extract(
            self._locators["patch_rgb"],
            episode=str(episode_number),
            system="SM_0",
            step=int(mapping_lm0[step_number]),
        )
        patch_rgb_lm0 = np.array(patch_rgb_lm0)[:, :, :3]
        sm0_image = Image(patch_rgb_lm0).scale(0.0005).shift(-0.21, 0.07, 0.0)
        self.plotter.at(MAIN_RENDERER_IX).add(sm0_image)
        self.sm0_image = sm0_image

        if steps_mask_lm1[step_number]:
            patch_rgb_lm1 = self.data_parser.extract(
                self._locators["patch_rgb"],
                episode=str(episode_number),
                system="SM_1",
                step=int(mapping_lm0[step_number]),
            )
            patch_rgb_lm1 = np.array(patch_rgb_lm1)[:, :, :3]

            sm1_image = Image(patch_rgb_lm1).scale(0.0005).shift(-0.16, 0.07, 0.0)
            self.plotter.at(MAIN_RENDERER_IX).add(sm1_image)
            self.sm1_image = sm1_image

        self._clear_all_paths()
        key_event = msgs_dict.get("KeyPressEvent", None)
        if key_event is not None and getattr(key_event, "at", None) == 1:
            key = getattr(key_event, "keypress", None)

            if key == "a":
                self.show_agent_past = not self.show_agent_past
            elif key == "A":
                self.show_agent_future = not self.show_agent_future
            elif key == "s":
                self.show_patch_past = not self.show_patch_past
            elif key == "S":
                self.show_patch_future = not self.show_patch_future
            elif key == "d":
                self.show_agent_past = False
                self.show_agent_future = False
                self.show_patch_past = False
                self.show_patch_future = False

        # expire the event so it only affects this call
        self.updaters[1].expire_topic("KeyPressEvent")

        max_idx = len(mapping_lm0) - 1
        curr_idx = int(np.clip(step_number, 0, max_idx))

        if self.show_agent_past or self.show_agent_future:
            self._rebuild_agent_paths(episode_number, mapping_lm0, curr_idx)

        if self.show_patch_past or self.show_patch_future:
            self._rebuild_patch_paths(episode_number, mapping_lm0, curr_idx)

        return widget, False

    def _rebuild_agent_paths(
        self,
        episode_number: int,
        mapping: np.ndarray,
        curr_idx: int,
    ) -> None:
        """Rebuild past/future agent paths."""
        # Collect all agent positions
        agent_positions: list[np.ndarray] = []
        for k in range(len(mapping)):
            pos = self.data_parser.extract(
                self._locators["agent_location"],
                episode=str(episode_number),
                sm_step=max(0, int(mapping[k]) - 1),
            )
            agent_positions.append(pos)

        if self.show_agent_past and agent_positions:
            past_pts = agent_positions[: curr_idx + 1]
            for p in past_pts:
                s = Sphere(pos=p, r=0.002, c=COLOR_PALETTE["Secondary"])
                self.plotter.at(SIMULATOR_IX).add(s)
                self.agent_past_spheres.append(s)
            if len(past_pts) >= 2:
                self.agent_past_line = Line(
                    past_pts, c=COLOR_PALETTE["Secondary"], lw=1
                )
                self.plotter.at(SIMULATOR_IX).add(self.agent_past_line)

        if (
            self.show_agent_future
            and agent_positions
            and curr_idx < len(agent_positions) - 1
        ):
            future_pts = agent_positions[curr_idx + 1 :]
            for p in future_pts:
                s = Sphere(pos=p, r=0.002, c=COLOR_PALETTE["Secondary"])
                self.plotter.at(SIMULATOR_IX).add(s)
                self.agent_future_spheres.append(s)
            if len(future_pts) >= 2:
                self.agent_future_line = Line(
                    future_pts, c=COLOR_PALETTE["Secondary"], lw=1
                )
                self.plotter.at(SIMULATOR_IX).add(self.agent_future_line)

    def _rebuild_patch_paths(
        self,
        episode_number: int,
        mapping: list,
        curr_idx: int,
    ) -> None:
        """Rebuild past/future patch (sensor) paths."""
        patch_positions: list[np.ndarray] = []
        for k in range(len(mapping)):
            pos = self.data_parser.extract(
                self._locators["patch_location"],
                episode=str(episode_number),
                system="SM_0",
                step=int(mapping[k]),
            )
            patch_positions.append(pos)

        if self.show_patch_past and patch_positions:
            past_pts = patch_positions[: curr_idx + 1]
            for p in past_pts:
                s = Sphere(pos=p, r=0.002, c=COLOR_PALETTE["Accent3"])
                self.plotter.at(SIMULATOR_IX).add(s)
                self.patch_past_spheres.append(s)
            if len(past_pts) >= 2:
                self.patch_past_line = Line(past_pts, c=COLOR_PALETTE["Accent3"], lw=1)
                self.plotter.at(SIMULATOR_IX).add(self.patch_past_line)

        if (
            self.show_patch_future
            and patch_positions
            and curr_idx < len(patch_positions) - 1
        ):
            future_pts = patch_positions[curr_idx + 1 :]
            for p in future_pts:
                s = Sphere(pos=p, r=0.002, c=COLOR_PALETTE["Accent2"])
                self.plotter.at(SIMULATOR_IX).add(s)
                self.patch_future_spheres.append(s)
            if len(future_pts) >= 2:
                self.patch_future_line = Line(
                    future_pts, c=COLOR_PALETTE["Accent2"], lw=1
                )
                self.plotter.at(SIMULATOR_IX).add(self.patch_future_line)

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


class InteractiveCompositionalPlot:
    """An interactive plot with 5 renderers for compositional visualization.

    This visualization features:
    - Main renderer (full window background) for primary content
    - Four corner overlay renderers for auxiliary views:
        - Top-left: Overlay view 1
        - Top-right: Overlay view 2
        - Bottom-left: Overlay view 3
        - Bottom-right: Overlay view 4

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
    ):
        renderer_areas = [
            {"bottomleft": (0.0, 0.0), "topright": (1.0, 1.0)},  # Main renderer
            {"bottomleft": (0.02, 0.3), "topright": (0.28, 0.7)},  # Simulator Overlay
            {"bottomleft": (0.3, 0.52), "topright": (0.48, 0.8)},  # H-LLM Model
            {"bottomleft": (0.52, 0.52), "topright": (0.7, 0.8)},  # H-LLM Prediction
            {"bottomleft": (0.3, 0.2), "topright": (0.48, 0.48)},  # L-LLM Model
            {"bottomleft": (0.52, 0.2), "topright": (0.7, 0.48)},  # L-LLM Prediction
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
        self.event_bus.sendMessage(
            "episode_number", msg=TopicMessage("episode_number", 0)
        )

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

        # Step slider widget
        widgets["step_slider"] = Widget[Slider2D, int](
            widget_ops=StepSliderWidgetOps(
                plotter=self.plotter,
                data_parser=self.data_parser,
            ),
            scopes=[1],
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
            ),
            scopes=[1],
            bus=self.event_bus,
            scheduler=self.scheduler,
            debounce_sec=0.5,
            dedupe=True,
        )

        return widgets

    def _setup_renderers(self) -> None:
        """Configure and show all renderers."""
        # Configure overlay renderers (no axes for cleaner look)
        for renderer_ix in [
            SIMULATOR_IX,
            HLLM_MODEL_IX,
            HLLM_PREDICTION_IX,
            LLLM_MODEL_IX,
            LLLM_PREDICTION_IX,
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


@register(
    "interactive_compositional_plot",
    description="Interactive compositional visualization with 5 renderers",
)
def main(experiment_log_dir: str, objects_mesh_dir: str) -> int:
    """Interactive compositional plot with 5 renderers.

    This visualization provides a multi-pane view for compositional analysis
    with a main renderer and four corner overlay renderers.

    Args:
        experiment_log_dir: Path to the experiment directory containing the
            detailed stats file.
        objects_mesh_dir: Path to the root directory of object meshes.

    Returns:
        Exit code.
    """
    vedo.settings.enable_default_keyboard_callbacks = False

    if not Path(experiment_log_dir).exists():
        logger.error(f"Experiment path not found: {experiment_log_dir}")
        return 1

    data_path = str(Path(objects_mesh_dir).expanduser())

    InteractiveCompositionalPlot(experiment_log_dir, data_path)

    return 0


@attach_args("interactive_compositional_plot")
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
