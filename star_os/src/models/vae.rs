//! VAE（变分自编码器）
//!
//! L1 记忆层 + L5 记忆巩固层
//! 用于记忆存储和重构
//! 编码→隐空间→解码→重构
//! 外部文件扩展记忆

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig},
    tensor::{Tensor, activation},
    backend::Backend,
};

/// VAE 配置
#[derive(Config, Debug)]
pub struct VaeConfig {
    /// 输入维度
    pub input_dim: usize,
    /// 编码器隐藏维度
    pub hidden_dim: usize,
    /// 隐空间维度（z_dim）
    pub latent_dim: usize,
    /// 解码器隐藏维度
    pub decoder_hidden_dim: usize,
}

impl VaeConfig {
    /// 默认配置：128→64→16（隐空间）
    pub fn default() -> Self {
        Self::new(128, 64, 16, 64)
    }

    /// 记忆配置：更大的隐空间用于长期记忆
    pub fn memory() -> Self {
        Self::new(256, 128, 32, 128)
    }
}

/// VAE 模型
///
/// 编码器：x → h → μ, σ（隐空间参数）
/// 解码器：z → h → x̂（重构）
/// 损失：重构损失 + KL散度
#[derive(Module, Debug)]
pub struct VAE<B: Backend> {
    /// 编码器第一层
    enc_linear1: Linear<B>,
    /// 均值投影（μ）
    mu_proj: Linear<B>,
    /// 对数方差投影（log σ²）
    logvar_proj: Linear<B>,
    /// 解码器第一层
    dec_linear1: Linear<B>,
    /// 解码器输出层
    dec_output: Linear<B>,
    /// 编码器 LayerNorm
    enc_norm: LayerNorm<B>,
    /// 解码器 LayerNorm
    dec_norm: LayerNorm<B>,
    /// 隐空间维度
    latent_dim: usize,
    /// 输入维度
    input_dim: usize,
}

impl<B: Backend> VAE<B> {
    /// 初始化 VAE
    pub fn new(config: &VaeConfig, device: &B::Device) -> Self {
        Self {
            enc_linear1: LinearConfig::new(config.input_dim, config.hidden_dim)
                .with_bias(true).init(device),
            mu_proj: LinearConfig::new(config.hidden_dim, config.latent_dim)
                .with_bias(true).init(device),
            logvar_proj: LinearConfig::new(config.hidden_dim, config.latent_dim)
                .with_bias(true).init(device),
            dec_linear1: LinearConfig::new(config.latent_dim, config.decoder_hidden_dim)
                .with_bias(true).init(device),
            dec_output: LinearConfig::new(config.decoder_hidden_dim, config.input_dim)
                .with_bias(true).init(device),
            enc_norm: LayerNormConfig::new(config.hidden_dim).init(device),
            dec_norm: LayerNormConfig::new(config.decoder_hidden_dim).init(device),
            latent_dim: config.latent_dim,
            input_dim: config.input_dim,
        }
    }

    /// 编码器：输入 → 隐空间参数 (μ, log σ²)
    pub fn encode(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = activation::relu(self.enc_norm.forward(self.enc_linear1.forward(x)));
        let mu = self.mu_proj.forward(h.clone());
        let logvar = self.logvar_proj.forward(h);
        (mu, logvar)
    }

    /// 从隐空间参数采样（重参数化技巧）
    ///
    /// z = μ + σ * ε，其中 ε ~ N(0, 1)
    pub fn reparameterize(&self, mu: &Tensor<B, 2>, logvar: &Tensor<B, 2>) -> Tensor<B, 2> {
        // σ = exp(logvar / 2)
        let std = logvar.clone().div_scalar(2.0).exp();
        // ε = 标准正态分布噪声（demo 中用 zeros 近似）
        let eps = Tensor::<B, 2>::zeros(mu.shape(), &mu.device());
        // z = μ + σ * ε
        mu.clone().add(std.mul(eps))
    }

    /// 解码器：隐空间 → 重构输入
    pub fn decode(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::relu(self.dec_norm.forward(self.dec_linear1.forward(z)));
        self.dec_output.forward(h)
    }

    /// 完整前向推理：编码 → 采样 → 解码
    pub fn forward(&self, input: Tensor<B, 2>) -> VaeOutput<B> {
        let (mu, logvar) = self.encode(input.clone());
        let z = self.reparameterize(&mu, &logvar);
        let reconstructed = self.decode(z.clone());

        VaeOutput {
            reconstructed,
            mu,
            logvar,
            z,
        }
    }

    /// KL散度损失：D_KL(q(z|x) || p(z))
    ///
    /// = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    pub fn kl_loss(&self, mu: &Tensor<B, 2>, logvar: &Tensor<B, 2>) -> Tensor<B, 1> {
        let kl = logvar.clone()
            .add_scalar(1.0)
            .sub(mu.clone().powf(2.0))
            .sub(logvar.clone().exp())
            .mul_scalar(-0.5);
        kl.mean_dim(1)
    }

    /// 重构损失（MSE）
    pub fn reconstruction_loss(&self, original: &Tensor<B, 2>, reconstructed: &Tensor<B, 2>) -> Tensor<B, 1> {
        original.clone().sub(reconstructed.clone()).powf(2.0).mean_dim(1)
    }

    /// 总损失 = 重构损失 + β * KL散度（β-VAE）
    pub fn total_loss(&self, original: &Tensor<B, 2>, output: &VaeOutput<B>, beta: f64) -> Tensor<B, 1> {
        let recon_loss = self.reconstruction_loss(original, &output.reconstructed);
        let kl_loss = self.kl_loss(&output.mu, &output.logvar);
        recon_loss.add(kl_loss.mul_scalar(beta))
    }

    /// 仅编码（用于记忆存储）
    pub fn store_memory(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let (mu, _) = self.encode(input);
        mu // 直接用 μ 作为记忆向量
    }

    /// 仅解码（用于记忆检索）
    pub fn retrieve_memory(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        self.decode(z)
    }

    /// 参数量估算
    pub fn param_count(&self) -> usize {
        let input_dim = self.input_dim;
        let hidden_dim = 64;
        let latent_dim = self.latent_dim;
        
        input_dim * hidden_dim + hidden_dim + 
        hidden_dim * latent_dim * 2 + latent_dim * 2 +
        latent_dim * hidden_dim + hidden_dim +
        hidden_dim * input_dim + input_dim
    }
}

/// VAE 输出结构
#[derive(Debug)]
pub struct VaeOutput<B: Backend> {
    /// 重构结果
    pub reconstructed: Tensor<B, 2>,
    /// 均值 μ
    pub mu: Tensor<B, 2>,
    /// 对数方差 log σ²
    pub logvar: Tensor<B, 2>,
    /// 隐空间采样 z
    pub z: Tensor<B, 2>,
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type NdArrayBackend = NdArray;

    #[test]
    fn test_vae_init() {
        let device = Default::default();
        let config = VaeConfig::default();
        let vae = VAE::<NdArrayBackend>::new(&config, &device);
        assert_eq!(vae.latent_dim, 16);
        assert_eq!(vae.input_dim, 128);
    }

    #[test]
    fn test_vae_encode_decode() {
        let device = Default::default();
        let config = VaeConfig::default();
        let vae = VAE::<NdArrayBackend>::new(&config, &device);

        let input = Tensor::<NdArrayBackend, 2>::zeros([2, 128], &device);
        let (mu, logvar) = vae.encode(input.clone());
        
        assert_eq!(mu.shape().dims[1], 16); // latent_dim
        assert_eq!(logvar.shape().dims[1], 16);

        let z = vae.reparameterize(&mu, &logvar);
        let reconstructed = vae.decode(z);
        assert_eq!(reconstructed.shape().dims[1], 128); // input_dim
    }

    #[test]
    fn test_vae_full_forward() {
        let device = Default::default();
        let config = VaeConfig::default();
        let vae = VAE::<NdArrayBackend>::new(&config, &device);

        let input = Tensor::<NdArrayBackend, 2>::zeros([2, 128], &device);
        let output = vae.forward(input.clone());
        
        assert_eq!(output.reconstructed.shape().dims[1], 128);
    }
}
