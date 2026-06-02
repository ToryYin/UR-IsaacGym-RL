# UR机械臂强化学习抓取

[English](README.md) | [简体中文](README_CN.md)

## 项目简介

本项目面向 Universal Robots（UR）机械臂的强化学习抓取控制，集成了：

* UR RTDE 通信接口
* Isaac Gym 仿真环境
* 强化学习仿真训练框架
* 真实机械臂部署

该框架可用于机械臂控制策略的训练、评估，以及仿真到真实系统（Sim-to-Real）的迁移。

---

## 功能特点

* 基于 RTDE 协议与 UR 机械臂实时通信
* 基于 Isaac Gym 的仿真训练
* 强化学习策略训练
* 支持真实机器人部署

---

## 项目结构与代码说明

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

该目录主要负责真实 UR 机械臂通信与控制。

| 文件/目录                          | 作用           |
| ------------------------------ | ------------ |
| RL-checkpoint/                 | 保存模型权重 |
| control_loop_configuration.xml | RTDE通信配置文件   |
| rl_grasp.py                    | 机械臂真机抓取控制脚本  |

### isaacgymenvs/

该目录主要负责仿真环境、任务定义和训练流程。

| 文件/目录    | 作用         |
| -------- | ---------- |
| cfg/     | 环境与任务配置文件  |
| runs/    | 训练日志和模型输出  |
| tasks/   | 强化学习任务实现   |
| train.py | 训练脚本     |

### assets/

该目录主要保存URDF模型

| 文件/目录    | 作用         |
| -------- | ---------- |
| urdf/    | URDF文件  |

---

## 环境要求

* Ubuntu 20.04
* Python 3.8
* Isaac Gym

---

## 安装方法

按照官方文档完成 Isaac Gym 安装

克隆仓库：

```bash
git clone https://github.com/ToryYin/UR-IsaacGym-RL.git
cd UR-IsaacGym-RL
```

---

## 模型训练

启动强化学习训练：

```bash
python train.py task=UR
```

根据实验需求修改 `cfg/` 中的任务配置文件。

---

## 机械臂真机部署

在真实机械臂上运行前，需要配置：

* 机械臂 IP 地址
* 部署视觉检测器（例如 GraspNet）

建议先在仿真环境验证策略，再部署到真实机械臂。

---

## 结果

<p align="center">
  <img src="demo/Isaac_Gym_Grasp.gif" width="800">
</p>

---

## 开源协议

MIT License

