# STARS-Gateway v0.1.0: Proof of Concept 
[Chinese](README.md) | English

[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue)](LICENSE)
[![Stage](https://img.shields.io/badge/Stage-Proof%20of%20Concept-yellow)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-black?style=flat-square)](LICENSE_MIT)
## License & Original Declaration

License & IP Protection

This project adopts a Dual Licensing Strategy to protect the intellectual property of lord of the star.

1. Core Engine (STARS-Sink): Licensed under GPL v3.
   * If you integrate this engine into your products, you must comply with the GPL v3 terms. This means you must open-source your derivative works.
2. Architectural Logic & White Paper: Licensed under MIT.
   * You are free to reference and learn, provided you retain the original copyright notice.

Copyright 2026 lord of the star. All Rights Reserved.

Current Status: Proof-of-Concept

This repository implements the first experimental validation (**STARS-Sink**). 
The full architectural vision is detailed in the accompanying white paper.

See  RFC.md (docs/RFC.md) for the technical roadmap.  
See also the ** Architectural White Paper (Lexicon v0.1.0) (群星架构白皮书.md)** for the theoretical framework of **Endogenous Autonomy**.

---

## What is STARS-Gateway?

STARS-Gateway is an experimental edge traffic governance framework born from the **Starswarm A.I. OS** research.

Instead of traditional "block-on-sight" WAF behavior, it employs **asymmetric resource consumption traps**. It accepts suspicious traffic, forcing the client to waste resources waiting for a response that never completes.

### Core Modules (Current & Planned)

| Module | Status | Description |
| :--- | :--- | :--- |
| **STARS-Sink** | **Implemented** | Async tarpit engine. Responds with exponentially delayed 1-byte payloads. |
| **STARS-Filter** | Design | eBPF/XDP in-kernel classifier. |
| **STARS-Pulse** | Design | SWIM-based gossip protocol. |

---
## Environment Requirements and Installation

### Prerequisites
- Linux kernel 5.4 or higher (Android AidLux / Termux both OK)
- Rust 1.75 or higher (install via rustup)

### Install Rust Environment

If Rust is not yet installed, run the following command in the terminal:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

During installation, press Enter to select the default options. Immediately after completion, run the following command to make the environment variables take effect:

```bash
source "$HOME/.cargo/env"
```

## Verify installation:

```bash
rustc --version
cargo --version
```

## Quick Start

1. Clone the repository and enter the directory

```bash
git clone https://github.com/l1064709321/stars-swarm-AGI.git
git checkout v0.1.0  # ensure using the v0.1.0 tag
```

2. Build

```
cargo build --release
```

3. Run

```
sudo ./target/release/stars-gateway
```

## After successful startup, it will show:

```
 星主  陷阱引擎已启动，监听端口 9999
```
4. Test (simulate an attack using slowhttptest)

Install slowhttptest in another terminal:

```
sudo apt-get update
sudo apt-get install -y slowhttptest
```

## Launch the attack:

```
slowhttptest -c 500 -H -g -o report -i 10 -r 200 -t GET -u http://127.0.0.1:9999/
```

## Things to observe:

*   The attacker client will hang and eventually time out.
*   The defender `stars-gateway` process CPU usage remains consistently below 5%.
*   Connections on the trap port stay in ESTABLISHED state (`ss -ntp | grep 9999`).---

## Project Structure
```
stars-gateway/
├── LICENSE_MIT          # 核心算法思想 (MIT)
├── LICENSE_GPLv3        # 核心引擎代码 (GPL v3)
├── Cargo.toml           # Rust 项目配置
├── README.md            # 项目门面 (放徽章和白皮书链接)
├── src/
│   └── main.rs         # 核心代码
└── docs/
    ├── RFC.md           # 技术规范
    └── Starswarm-AIOS-WhitePaper-v0.1.0-Lexicon.md  # 群星AI.OS
```

## Naming & Philosophy
**STARS** stands for *Stateless Traffic Absorption & Resource-Sink*. 
It reflects the **Starswarm A.I. OS** concept where each node acts as an independent "star" consuming entropy.

## Roadmap
*   **v0.1.x**: Stabilize STARS-Sink (Current).
*   **v0.2.0**: Introduce STARS-Filter (eBPF/XDP).
*   **v0.3.0**: Integrate STARS-Pulse (Decentralized Mesh).

## License & Original Declaration
This project is licensed under the **GNU General Public License v3.0**.

All concepts and terminology defined in the **Architectural White Paper (Lexicon v0.1.0)** — including but not limited to **"Endogenous Autonomy"**, **"Fractal Cognition"**, and **"PPU (Pulse-Potential Units)"** — are original intellectual property of the author.

# **Copyright 2026 lord of the star**
