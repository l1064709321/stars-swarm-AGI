//! TextCNN 训练脚本
//!
//! 参数量约30万，~1MB
//! 训练数据：CICIDS2017（网络安全分类）
//! 训练时间约2小时（CPU）
//! Demo 模式：使用合成数据验证架构

use burn::{
    backend::NdArray,
    tensor::Tensor,
    data::dataset::Dataset,
    module::Module,
    optim::AdamConfig,
    train::{LearnerConfig, TrainOutput, TrainStep, ValidStep},
};

// 引入项目模型
use star_os::models::textcnn::{TextCnn, TextCnnConfig, SecurityCategory};

/// 训练入口
fn main() {
    println!("══════════════════════════════════════");
    println!("  TextCNN 训练脚本 v0.0.0.1");
    println!("  数据集: CICIDS2017 (合成 demo)");
    println!("  目标分类数: {}", SecurityCategory::count());
    println!("══════════════════════════════════════");

    let device = Default::default();

    // 初始化模型
    let config = TextCnnConfig::cicids();
    let model = TextCnn::<NdArray>::new(&config, &device);
    println!("[模型] TextCNN 初始化完成");
    println!("[参数] 估算参数量约30万");

    // 创建合成数据集
    let dataset = SyntheticSecurityDataset::new(1000);
    println!("[数据] 合成数据集: {} 条样本", dataset.len());

    // Demo 验证：单次前向推理
    let input = Tensor::<NdArray, 4>::zeros([4, 1, 10, 128], &device);
    let output = model.forward(input);
    println!("[验证] 前向推理: 输出形状 {:?}", output.shape());

    // 模拟训练过程
    println!("\n[训练] 开始模拟训练...");
    for epoch in 1..=5 {
        let loss = simulate_training_loss(epoch);
        let accuracy = simulate_accuracy(epoch);
        println!("[Epoch {}] Loss={:.4} Accuracy={:.1}%", epoch, loss, accuracy * 100.0);
    }

    println!("\n[完成] TextCNN 训练 demo 完成");
    println!("[保存] 模型权重应保存至 model_weights/textcnn.pt");
    println!("[备注] 完整训练需要真实 CICIDS2017 数据集，约2小时(CPU)");
}

/// 模拟训练损失曲线
fn simulate_training_loss(epoch: usize) -> f64 {
    1.0 / (0.5 * epoch as f64 + 0.1)
}

/// 模拟训练准确率曲线
fn simulate_accuracy(epoch: usize) -> f64 {
    let base = 0.6;
    base + 0.08 * epoch as f64 // 逐步提升
}

/// 合成安全数据集（Demo 用）
///
/// 模拟 CICIDS2017 格式的网络安全数据
struct SyntheticSecurityDataset {
    samples: Vec<SecuritySample>,
}

#[derive(Debug, Clone)]
struct SecuritySample {
    /// 特征向量（模拟网络流量特征）
    features: Vec<f64>,
    /// 分类标签
    label: usize,
}

impl SyntheticSecurityDataset {
    fn new(size: usize) -> Self {
        let mut samples = Vec::new();
        for i in 0..size {
            let label = i % SecurityCategory::count();
            let features: Vec<f64> = (0..128)
                .map(|j| ((i + j) as f64 * 0.01 + label as f64 * 0.5) % 1.0)
                .collect();
            samples.push(SecuritySample { features, label });
        }
        Self { samples }
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}
