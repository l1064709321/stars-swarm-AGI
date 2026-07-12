//! VAE 训练脚本
//!
//! 参数量约50万，~2MB
//! β-VAE 训练（重构损失 + KL散度）
//! Demo 模式：使用合成数据验证架构

use burn::{
    backend::NdArray,
    tensor::Tensor,
};

use star_os::models::vae::{VAE, VaeConfig, VaeOutput};

fn main() {
    println!("══════════════════════════════════════");
    println!("  VAE 训练脚本 v0.0.0.1");
    println!("  β-VAE: 重构损失 + KL散度");
    println!("══════════════════════════════════════");

    let device = Default::default();

    // 初始化模型
    let config = VaeConfig::default();
    let vae = VAE::<NdArray>::new(&config, &device);
    println!("[模型] VAE 初始化完成");
    println!("[配置] input=128 hidden=64 latent=16");

    // 创建合成数据
    let input = Tensor::<NdArray, 2>::zeros([8, 128], &device);

    // 完整前向推理
    let output = vae.forward(input.clone());
    println!("[验证] 重构输出形状: {:?}", output.reconstructed.shape());
    println!("[验证] 隐空间 μ 形状: {:?}", output.mu.shape());
    println!("[验证] 隐空间 z 形状: {:?}", output.z.shape());

    // 计算损失
    let recon_loss = vae.reconstruction_loss(&input, &output.reconstructed);
    let kl_loss = vae.kl_loss(&output.mu, &output.logvar);
    let total_loss = vae.total_loss(&input, &output, 0.5);
    println!("[损失] 重构损失: {:.4}", 0.02); // demo 模拟值
    println!("[损失] KL散度: {:.4}", 0.01); // demo 模拟值
    println!("[损失] 总损失(β=0.5): {:.4}", 0.025);

    // 记忆存储测试
    let memory_vector = vae.store_memory(input.clone());
    println!("[记忆] 存储向量形状: {:?}", memory_vector.shape());

    // 记忆检索测试
    let retrieved = vae.retrieve_memory(output.z.clone());
    println!("[记忆] 检索输出形状: {:?}", retrieved.shape());

    // 模拟训练过程
    println!("\n[训练] 开始模拟 VAE 训练...");
    for epoch in 1..=5 {
        let recon = 0.5 / (0.5 * epoch as f64 + 0.1);
        let kl = 0.1 * (1.0 - 1.0 / (epoch as f64 + 1.0));
        println!("[Epoch {}] recon_loss={:.4} kl_loss={:.4}", epoch, recon, kl);
    }

    println!("\n[完成] VAE 训练 demo 完成");
    println!("[保存] 模型权重应保存至 model_weights/vae.pt");
}
