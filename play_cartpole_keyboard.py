# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL Cartpole policy with additive A/D keyboard pushes."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--keyboard_livestream",
    action="store_true",
    help="Enable omni.kit.livestream.app for keyboard input using the existing streaming client.",
)
parser.add_argument(
    "--policy_delay", type=float, default=0.0,
    help="Delay policy actions sent to the cart, in simulation seconds (default: no delay).",
)
parser.add_argument("--push_force", type=float, default=10.0, help="A/D force at the pole tip in newtons.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Use the same streaming extension as the working standalone example.
if args_cli.keyboard_livestream:
    args_cli.headless = True
    # Avoid starting a second, different streaming backend through AppLauncher.
    args_cli.livestream = 0
    if not args_cli.experience:
        args_cli.experience = "isaaclab.python.kit"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if args_cli.keyboard_livestream:
    from isaacsim.core.experimental.utils.app import enable_extension

    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/window/hideUi", False)
    # SimulationContext reads these before creating the environment. Keep rendering
    # and event processing enabled even though there is no local desktop window.
    simulation_app.set_setting("/isaaclab/render/active_viewport", True)
    simulation_app.set_setting("/app/livestream/enabled", True)
    enable_extension("omni.kit.livestream.app")
    simulation_app.update()
    print("[INFO] Keyboard livestream enabled via omni.kit.livestream.app.", flush=True)

"""Rest everything follows."""

import math
import os
import time
from collections import deque

import carb
import omni.appwindow

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


class CartpoleKeyboardPush:
    """Apply a held-key world-Y force at the pole tip, leaving policy efforts intact."""

    def __init__(self, env, force: float = 10.0):
        if force < 0:
            raise ValueError("--push_force must be non-negative.")
        self.force = force
        self.pressed_keys = set()
        self._last_applied_push = 0.0
        # Manager-based Cartpole uses "robot"; direct Cartpole uses "cartpole".
        self.cartpole = next(
            (asset for asset in env.scene.articulations.values() if "slider_to_cart" in asset.joint_names),
            None,
        )
        if self.cartpole is None:
            raise ValueError("Keyboard pushes require a Cartpole articulation with joint 'slider_to_cart'.")
        # Resolve the pole body through its joint instead of assuming a body index.
        stage = env.scene.stage
        env_prim = stage.GetPrimAtPath(env.scene.env_prim_paths[0])
        # A pushes along world -Y; D pushes along world +Y.
        self.push_direction = torch.tensor(
            [0.0, 1.0, 0.0], dtype=torch.float32, device=env.device
        ).reshape(1, 1, 3)
        pole_joint = next(
            (prim for prim in Usd.PrimRange(env_prim, Usd.TraverseInstanceProxies())
             if prim.GetName() == "cart_to_pole" and prim.IsA(UsdPhysics.RevoluteJoint)),
            None,
        )
        if pole_joint is None:
            raise ValueError("Could not find the Cartpole joint 'cart_to_pole'.")
        joint = UsdPhysics.RevoluteJoint(pole_joint)
        slider_prim = next(
            (prim for prim in Usd.PrimRange(env_prim, Usd.TraverseInstanceProxies())
             if prim.GetName() == "slider_to_cart" and prim.IsA(UsdPhysics.PrismaticJoint)),
            None,
        )
        if slider_prim is None:
            raise ValueError("Could not find the cart slider joint.")
        slider = UsdPhysics.PrismaticJoint(slider_prim)
        slider_bodies = set(slider.GetBody0Rel().GetTargets() + slider.GetBody1Rel().GetTargets())
        pole_candidates = [
            (side, path)
            for side, relation in enumerate((joint.GetBody0Rel(), joint.GetBody1Rel()))
            for path in relation.GetTargets()
            if path not in slider_bodies and path.name in self.cartpole.body_names
        ]
        if len(pole_candidates) != 1:
            raise ValueError(f"Could not identify the pole body independently of the cart: {pole_candidates}")
        pole_side, pole_path = pole_candidates[0]
        self.pole_body_ids, _ = self.cartpole.find_bodies(pole_path.name)
        if len(self.pole_body_ids) != 1:
            raise ValueError(f"Expected one pole body, found {self.pole_body_ids}.")
        # Find the end furthest from the hinge using the pole's local geometry bounds.
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
        ).ComputeUntransformedBound(stage.GetPrimAtPath(pole_path)).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise ValueError("Could not determine the pole tip from its geometry.")
        lower, upper = bounds.GetMin(), bounds.GetMax()
        axis = max(range(3), key=lambda i: upper[i] - lower[i])
        hinge = (joint.GetLocalPos0Attr() if pole_side == 0 else joint.GetLocalPos1Attr()).Get()
        tip = [(lower[i] + upper[i]) * 0.5 for i in range(3)]
        tip[axis] = max((lower[axis], upper[axis]), key=lambda value: abs(value - hinge[axis]))
        self.tip_local = torch.tensor(tip, dtype=torch.float32, device=env.device).reshape(1, 1, 3)
        self.tip_local = self.tip_local.repeat(env.num_envs, 1, 1)
        self.forces = torch.zeros_like(self.tip_local)
        self.tip_marker = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/KeyboardPoleTip",
            markers={"tip": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.0)),
            )},
        ))
        print(f"[INFO] Pole push body: {pole_path}, body IDs: {self.pole_body_ids}, local tip: {tip}", flush=True)
        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self._on_keyboard_event)
        self.original_write_data_to_sim = self.cartpole.write_data_to_sim
        # Inject an external wrench on every physics substep, after policy actions.
        self.cartpole.write_data_to_sim = self._write_data_to_sim
        print(f"[INFO] Hold A/D to push the pole tip along world -Y/+Y ({force:g} N). Release to stop pushing.")

    def _on_keyboard_event(self, event, *args):
        # Streaming can provide a string; local Carb events usually provide an enum.
        key = event.input if isinstance(event.input, str) else getattr(event.input, "name", None)
        if key in {"A", "D"}:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if key not in self.pressed_keys:
                    print(f"[KEYBOARD] {key} pressed", flush=True)
                self.pressed_keys.add(key)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self.pressed_keys.discard(key)
                print(f"[KEYBOARD] {key} released", flush=True)
        return True

    def _write_data_to_sim(self):
        push = self.force * (int("D" in self.pressed_keys) - int("A" in self.pressed_keys))
        data = self.cartpole.data
        tip_world = data.body_link_pos_w[:, self.pole_body_ids] + quat_apply(
            data.body_link_quat_w[:, self.pole_body_ids], self.tip_local
        )
        self.tip_marker.visualize(translations=tip_world[:, 0])
        if push != 0.0:
            self.forces.zero_()
            self.forces.copy_(self.push_direction * push)
            # A force at the tip also creates the appropriate lever-arm torque.
            # Instantaneous wrenches are cleared by write_data_to_sim each substep.
            self.cartpole.instantaneous_wrench_composer.add_forces_and_torques(
                forces=self.forces,
                positions=tip_world,
                body_ids=self.pole_body_ids,
                is_global=True,
            )
        self.original_write_data_to_sim()
        if push != self._last_applied_push:
            if push != 0.0:
                print(
                    f"[PUSH] Applied {push:+g} N along world Y at pole tip "
                    f"{tip_world[0, 0].tolist()} (environment 0)",
                    flush=True,
                )
            else:
                print("[PUSH] Stopped", flush=True)
            self._last_applied_push = push

    def close(self):
        self.cartpole.write_data_to_sim = self.original_write_data_to_sim
        self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)
        self.pressed_keys.clear()
        self.tip_marker.set_visibility(False)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Disable all automatic episode endings for interactive evaluation.
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.terminations = {}

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if isinstance(env_cfg, DirectRLEnvCfg):
        # Keep task-specific state updates in _get_dones, but suppress both reset signals.
        original_get_dones = env.unwrapped._get_dones

        def get_dones_without_reset():
            terminated, time_outs = original_get_dones()
            return torch.zeros_like(terminated), torch.zeros_like(time_outs)

        env.unwrapped._get_dones = get_dones_without_reset
    print("[INFO] Automatic resets disabled (time limits and failure conditions).")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    if args_cli.policy_delay < 0:
        raise ValueError("--policy_delay must be non-negative.")
    delay_steps = math.ceil(args_cli.policy_delay / dt)
    delayed_actions = deque()
    if delay_steps:
        print(f"[INFO] Policy action delay: {delay_steps * dt:g} simulation seconds ({delay_steps} steps).")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    keyboard_push = CartpoleKeyboardPush(env.unwrapped, force=args_cli.push_force)
    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                if delay_steps:
                    delayed_actions.append(actions.clone())
                    # During startup, apply zero effort until the first action has aged enough.
                    actions = delayed_actions.popleft() if len(delayed_actions) > delay_steps else torch.zeros_like(actions)
                # env stepping
                obs, _, dones, _ = env.step(actions)
                # reset recurrent states for episodes that have terminated
                policy_nn.reset(dones)
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        keyboard_push.close()
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
