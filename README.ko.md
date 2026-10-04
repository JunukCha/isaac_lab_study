# Cartpole Keyboard Push Demo

[English](README.md)

학습된 RSL-RL 정책이 카트를 제어하는 동안 A/D 키로 막대 끝에 외력을 가하는 Isaac Lab 데모입니다. 정책의 균형 유지 반응을 관찰할 수 있습니다.

## 기능

- `A`: 월드 `-Y` 방향 외력
- `D`: 월드 `+Y` 방향 외력
- `--push_force`: 외력 크기 조절
- 상호작용 평가 중 자동 리셋 비활성화
- 외력 작용점을 주황색 마커로 표시
- 영상 녹화 및 실시간 실행 지원

키를 놓으면 외력은 즉시 0이 되며 기본값은 `10 N`입니다.

## 실행 환경

- Isaac Lab: 실제 사용 버전을 기록하세요.
- Isaac Sim: 실제 사용 버전을 기록하세요.
- Isaac Lab이 제공하는 Python 환경
- 학습된 Cartpole RSL-RL 체크포인트

## 설치 방법

스크립트를 Isaac Lab의 `scripts/reinforcement_learning/rsl_rl/` 디렉터리에 복사합니다.

```bash
cp play_cartpole_keyboard.py /path/to/IsaacLab/scripts/reinforcement_learning/rsl_rl/
```

## 학습 및 실행

먼저 Isaac Lab의 기본 RSL-RL 학습 스크립트로 Cartpole 정책을 학습합니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-v0 \
  --headless
```

학습이 끝나면 `logs/rsl_rl/cartpole/<run>/` 아래에서 체크포인트를 찾습니다. 예시는 다음과 같습니다.

```text
logs/rsl_rl/cartpole/2026-10-04_03-45-58/model_149.pt
```

찾은 체크포인트 경로를 키보드 데모 실행 명령에 지정합니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_cartpole_keyboard.py \
  --task Isaac-Cartpole-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/cartpole/2026-10-04_03-45-58/model_149.pt
```

| 옵션 | 설명 |
|---|---|
| `--task NAME` | Isaac Lab 태스크 이름. 여기서는 `Isaac-Cartpole-v0` |
| `--num_envs N` | 실행할 환경 수. 여기서는 `1` |
| `--checkpoint PATH` | 학습된 RSL-RL 체크포인트 경로 |

실행 중 Isaac Sim 창에 포커스를 두고 `A` 또는 `D`를 누르세요. 이 데모는 기존 체크포인트를 평가하며 새 정책을 학습하지 않습니다.

