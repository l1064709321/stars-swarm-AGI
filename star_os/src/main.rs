//! 星OS v0.0.0.1 - 十层神经架构AI系统
//!
//! 面向 AidLux 端侧部署的轻量级 AI 推理框架
//! 核心架构：MoE Router + Mamba SSM + 多专家模型
//!
//! # 系统层次
//! - L1  感知/路由/融合：Mamba + MoE Router + SSM
//! - L1  记忆：VAE
//! - L1.5 语言：Transformer
//! - L1.5 语义编码：Encoder
//! - L3.5 因果发现/推理：PC Algorithm + GNN
//! - L4.5 决策搜索：MCTS
//! - L5  记忆巩固 + 语义网络：VAE + GNN
//! - L6  伦理判断 + 防御进化：Symbolic + GA
//!
//! # 技术栈
//! - 语言: Rust
//! - 框架: burn (ndarray backend, 支持 ARM CPU)
//! - 目标: AidLux 端侧推理部署

mod models;
mod bus;
mod utils;

use bus::message_bus::MessageBus;
use utils::scheduler::ModelScheduler;

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════╗");
    println!("║  星OS v0.0.0.1 - 十层神经架构AI系统  ║");
    println!("║  目标平台: AidLux (ARM端侧部署)      ║");
    println!("║  核心框架: Rust + burn (ndarray)     ║");
    println!("╚══════════════════════════════════════╝");

    // 初始化消息总线
    let bus = MessageBus::new();
    println!("\n[MessageBus] 初始化完成");

    // 初始化模型调度器
    let scheduler = ModelScheduler::new();
    println!("[Scheduler] 初始化完成");
    println!("[Scheduler] {}", scheduler.memory_report());

    // 运行 demo 测试
    demo_run(&bus);
}

/// Demo 测试：模拟消息路由流程
fn demo_run(bus: &MessageBus) {
    println!("\n--- Demo 测试开始 ---");

    // 模拟输入
    let test_inputs = vec![
        ("安全查询请求", "security"),
        ("语义理解请求", "semantic"),
        ("语言生成请求", "language"),
        ("因果推理请求", "causal"),
    ];

    for (text, category) in &test_inputs {
        println!("\n[输入] {} (类别: {})", text, category);
        let result = bus.process(text, category);
        println!("[输出] 路由→{} | 置信度→{:.2} | 伦理→{}", 
            result.route_target, result.confidence, result.ethics_status);
    }

    println!("\n--- Demo 测试完成 ---");
}
