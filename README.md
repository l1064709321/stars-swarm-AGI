# STARS-Gateway v0.1.0: 概念验证
[English](README-en.md) | 中文.

[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue)](LICENSE)
[![Stage](https://img.shields.io/badge/Stage-Proof%20of%20Concept-yellow)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-black?style=flat-square)](LICENSE_MIT)
##  许可证与原创声明
许可证与知识产权保护
本项目采用双许可策略，以保护星主的知识产权。
1. 核心引擎（STARS-Sink）：采用 GPL v3 许可。
   * 若您将本引擎集成至您的产品中，您必须遵守 GPL v3 条款。这意味着您必须开源您的衍生作品。
2. 架构逻辑与白皮书：采用 MIT 许可。
   * 您可自由参考与学习，但需保留原始版权声明。
版权所有 2026 星主。保留所有权利。
当前状态：概念验证
本仓库实现了首次实验性验证（**STARS-Sink**）。
完整的架构愿景详见随附白皮书。
技术路线图请参阅 [RFC.md](docs/RFC.md)。
关于**内生自主性**的理论框架，请参阅**[架构白皮书（词源 v0.1.0）](群星架构白皮书.md)**。
---
## STARS-Gateway 是什么？
STARS-Gateway 是一个源自**群星 A.I. OS** 研究的实验性边缘流量治理框架。
它不采用传统“一概拦截”的 WAF 行为，而是部署**非对称资源消耗陷阱**。它接受可疑流量，迫使客户端浪费资源等待一个永远不会完成的响应。
### 核心模块（当前与规划中）
| 模块 | 状态 | 描述 |
| :--- | :--- | :--- |
| **STARS-Sink** | **已实现** | 异步陷阱引擎。以指数延迟的1字节载荷进行响应。 |
| **STARS-Filter** | 设计中 | eBPF/XDP 内核分类器。 |
| **STARS-Pulse** | 设计中 | 基于 SWIM 的八卦协议。 |
---
## 环境要求与安装
### 前置条件
- Linux 内核 5.4 或更高版本（Android AidLux / Termux 均可）
- Rust 1.75 或更高版本（通过 rustup 安装）
### 安装 Rust 环境
若尚未安装 Rust，请在终端中运行以下命令：
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
### 安装过程中，按 Enter 选择默认选项。完成后立即运行以下命令使环境变量生效：
```
source "$HOME/.cargo/env"
```
### 验证安装：
```
rustc --version
cargo --version
```
### 快速开始
克隆仓库并进入目录
```
git clone https://github.com/l1064709321/stars-swarm-AGI.git
git checkout v0.1.0 # 确保使用 v0.1.0 标签
```
### 构建
```
cargo build --release
```
###  运行
```
sudo ./target/release/stars-gateway
```
###  成功启动后，将显示：
```
[星主] 陷阱引擎已启动，监听端口 9999
```
###  测试（使用 slowhttptest 模拟攻击）
在另一个终端中安装 slowhttptest：
```
sudo apt-get update
sudo apt-get install -y slowhttptest
```
发起攻击：
```
slowhttptest -c 500 -H -g -o report -i 10 -r 200 -t GET -u http://127.0.0.1:9999/
```
###  观察要点：
攻击者客户端将挂起并最终超时。
防御方 stars-gateway 进程的 CPU 使用率持续保持在 5% 以下。
陷阱端口上的连接将保持 ESTABLISHED 状态（ss -ntp | grep 9999）。
项目结构
```
stars-gateway/
├── LICENSE_MIT          # 核心算法思想 (MIT)
├── LICENSE_GPLv3        # 核心引擎代码 (GPL v3)
├── Cargo.toml           # Rust 项目配置
├── README.md            # 项目门面 (放置徽章和白皮书链接)
├── src/
│   └── main.rs         # 核心代码
└── docs/
    ├── RFC.md           # 技术规范
    └── Starswarm-AIOS-WhitePaper-v0.1.0-Lexicon.md  # 群星AI.OS
```
# 命名与哲学
STARS 代表 无状态流量吸收与资源沉没。
它体现了群星 A.I. OS 的概念，其中每个节点都扮演着消耗熵的独立“星体”。
路线图
v0.1.x：稳定 STARS-Sink（当前）。
v0.2.0：引入 STARS-Filter（eBPF/XDP）。
v0.3.0：集成 STARS-Pulse（去中心化网格）。
#  许可证与原创声明
本项目采用 GNU 通用公共许可证 v3.0 进行授权。
架构白皮书（词源 v0.1.0） 中定义的所有概念与术语——包括但不限于“内生自主性”、“分形认知”和“PPU（脉冲势单位）”——均为作者的原创知识产权。
#  版权所有 2026 星主
