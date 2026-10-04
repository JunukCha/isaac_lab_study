# Cartpole Keyboard Push Demo

[한국어 문서](README.ko.md)

This Isaac Lab demo applies an external force to the free end of the pole while a trained RSL-RL policy controls the cart. Use `A` and `D` to observe the policy's balance response.

## Features

- Hold `A` for world `-Y` force and `D` for world `+Y` force.
- Adjust force magnitude with `--push_force`.
- Disable automatic episode resets during interactive evaluation.
- Visualize the force application point with an orange marker.
- Optionally record video or run in real-time mode.

Releasing the key immediately removes the force. The default force is `10 N`.

## Environment

- Isaac Lab: record the exact version used, for example `v2.3.x`.
- Isaac Sim: record the exact version used, for example `4.5.x`.
- Python environment provided by Isaac Lab.
- A trained Cartpole RSL-RL checkpoint.

Replace the example versions with the exact versions used by your project before publishing.

## Installation

Copy `play_cartpole_keyboard.py` into Isaac Lab's RSL-RL script directory:

```bash
cp play_cartpole_keyboard.py /path/to/IsaacLab/scripts/reinforcement_learning/rsl_rl/
```

## Remote streaming (optional)

If Isaac Sim is running on a remote GPU server and you want to view the evaluation through livestreaming, set the following environment variables before launching evaluation:

```bash
export LIVESTREAM=1
export PUBLIC_IP="${PUBLIC_IPADDR:-$(curl -fsS https://api.ipify.org)}"
echo "PUBLIC_IP=${PUBLIC_IP}"
```

This avoids hardcoding a server-specific IP address. On Vast.ai, `PUBLIC_IPADDR` is used automatically when available. On other remote Linux servers, the command falls back to querying the machine's public IP with `curl`.

`LIVESTREAM=1` enables livestreaming. These variables are only needed for remote streaming and are not required for headless training or local GUI execution.

## Training and usage

First train the Cartpole policy with Isaac Lab's original RSL-RL training script:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-v0 \
  --headless
```

After training, find the checkpoint under `logs/rsl_rl/cartpole/<run>/`. For example:

```text
logs/rsl_rl/cartpole/2026-10-04_03-45-58/model_149.pt
```

Use that checkpoint path with the keyboard demo:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_cartpole_keyboard.py \
  --task Isaac-Cartpole-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/cartpole/2026-10-04_03-45-58/model_149.pt
```

| Option | Default | Description |
|---|---:|---|
| `--task NAME` | — | Isaac Lab task name, here `Isaac-Cartpole-v0` |
| `--num_envs N` | Task default | Number of environments, here `1` |
| `--checkpoint PATH` | — | Path to the trained RSL-RL checkpoint |

Focus the Isaac Sim window and hold `A` or `D` during execution. This demo evaluates an existing checkpoint; it does not train a new policy.
