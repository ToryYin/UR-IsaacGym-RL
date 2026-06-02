# UR Grasping Reinforcement Learning

[English](README.md) | [简体中文](README_CN.md)

## Overview

This project is a reinforcement learning framework for Universal Robots (UR) grasping. It integrates:

* UR RTDE communication interface
* Isaac Gym simulation environment
* Reinforcement learning training
* Real robot deployment utilities

The framework is designed for policy training, evaluation, and sim-to-real transfer.

---

## Features

* Real-time communication with UR robots via RTDE
* Isaac Gym-based simulation and training
* Reinforcement learning policy training
* Support for real robot deployment

---

## Repository Structure and Code Description

```text
.
├── rtde_python/
│   ├── control_loop_configuration.xml
│   └── rl_grasp.py
│
├── isaacgymenvs/
│   ├── cfg/
│   ├── runs/
│   ├── tasks/
│   ├── train.py
│   └── assets/
│
├── assets/
│   └── urdf/
│
├── README.md
└── README_CN.md
```

### rtde_python/

This directory mainly handles real UR robot communication and control.

| File/Folder                    | Description                                             |
| ------------------------------ | ------------------------------------------------------- |
| RL-checkpoint/                 | Stores trained model checkpoints and experiment weights |
| control_loop_configuration.xml | RTDE communication configuration                        |
| rl_grasp.py                    | Grasping control script for real robot deployment       |

### isaacgymenvs/

This directory contains simulation environments, task definitions, and training scripts.

| File/Folder | Description                                 |
| ----------- | ------------------------------------------- |
| cfg/        | Environment and task configuration files    |
| runs/       | Training logs and checkpoints               |
| tasks/      | Reinforcement learning task implementations |
| train.py    | Main training entry point                   |
| assets/     | Robot models, meshes, and simulation assets |

### assets/

This directory contains URDF files.

| File/Folder | Description         |
| ----------- | ----------- |
| urdf/       | URDF files  |

---

## Requirements

* Ubuntu 20.04
* Python 3.8
* Isaac Gym

---

## Installation

Install Isaac Gym according to the official instructions.

Clone the repository:

```bash
git clone https://github.com/ToryYin/UR-IsaacGym-RL.git
cd UR-IsaacGym-RL
```

---

## Training

Launch reinforcement learning training:

```bash
python train.py task=UR
```

Modify the task configuration files under `cfg/` according to your experiment setup.

---

## Real Robot Deployment

Before deploying on a real UR robot, configure:

* Robot IP address
* Deploy visual detectors Deploy visual detectors (eg. GraspNet)

It is recommended to validate policies in simulation before real-world deployment.

---

## Demo

<p align="center">
  <img src="demo/Isaac_Gym_Grasp.gif" width="800">
</p>

---

## License

MIT License

