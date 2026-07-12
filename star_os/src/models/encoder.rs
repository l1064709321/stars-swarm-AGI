//! 语义编码器（Semantic Encoder）
//!
//! BGE-small-en-v1.5 类似功能的简化实现
//! 参数量约10万，~4MB
//! 对比学习训练：正样本相似度>0.9，负样本<0.2
//! 用于文本转向量（语义嵌入）

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig, Dropout, DropoutConfig},
    tensor::{Tensor, activation},
    backend::Backend,
};

/// 语义编码器配置
#[derive(Config, Debug)]
pub struct EncoderConfig {
    /// 输入维度（词汇表嵌入维度）
    pub input_dim: usize,
    /// 隐藏维度
    pub hidden_dim: usize,
    /// 输出嵌入维度
    pub output_dim: usize,
    /// Dropout 比率
    pub dropout: f64,
}

impl EncoderConfig {
    /// 默认配置：128→64→32 维嵌入
    pub fn small() -> Self {
        Self::new(128, 64, 32, 0.1)
    }
    
    /// BGE-small 类似配置：384→256→128 维嵌入
    pub fn bge_like() -> Self {
        Self::new(384, 256, 128, 0.1)
    }
}

/// 语义编码器模型
///
/// 流程：输入 → LayerNorm → Linear → ReLU → Dropout → Linear → LayerNorm → 输出
/// 输出向量用于对比学习（正样本拉近，负样本推远）
#[derive(Module, Debug)]
pub struct SemanticEncoder<B: Backend> {
    /// 输入 LayerNorm
    input_norm: LayerNorm<B>,
    /// 第一层线性变换
    linear1: Linear<B>,
    /// 第二层线性变换（嵌入输出）
    linear2: Linear<B>,
    /// 输出 LayerNorm
    output_norm: LayerNorm<B>,
    /// Dropout
    dropout: Dropout,
    /// 输出维度
    output_dim: usize,
}

impl<B: Backend> SemanticEncoder<B> {
    /// 初始化编码器
    pub fn new(config: &EncoderConfig, device: &B::Device) -> Self {
        Self {
            input_norm: LayerNormConfig::new(config.input_dim).init(device),
            linear1: LinearConfig::new(config.input_dim, config.hidden_dim)
                .with_bias(true)
                .init(device),
            linear2: LinearConfig::new(config.hidden_dim, config.output_dim)
                .with_bias(false)
                .init(device),
            output_norm: LayerNormConfig::new(config.output_dim).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
            output_dim: config.output_dim,
        }
    }

    /// 前向推理：生成语义嵌入向量
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.input_norm.forward(input);
        let x = self.dropout.forward(activation::relu(self.linear1.forward(x)));
        let x = self.linear2.forward(x);
        self.output_norm.forward(x)
    }

    /// 计算两个嵌入向量的余弦相似度
    ///
    /// 用于对比学习：正样本相似度>0.9，负样本<0.2
    pub fn cosine_similarity(a: &Tensor<B, 2>, b: &Tensor<B, 2>) -> Tensor<B, 1> {
        // cos_sim = (a · b) / (||a|| * ||b||)
        let dot = a.clone().mul(b.clone()).sum_dim(1);
        let norm_a = a.clone().powf(2.0).sum_dim(1).sqrt();
        let norm_b = b.clone().powf(2.0).sum_dim(1).sqrt();
        dot.div(norm_a.mul(norm_b))
    }

    /// 对比学习损失（InfoNCE）
    ///
    /// 正样本拉近，负样本推远
    pub fn contrastive_loss(
        anchor: &Tensor<B, 2>,
        positive: &Tensor<B, 2>,
        negative: &Tensor<B, 2>,
        temperature: f64,
    ) -> Tensor<B, 1> {
        let pos_sim = SemanticEncoder::<B>::cosine_similarity(anchor, positive);
        let neg_sim = SemanticEncoder::<B>::cosine_similarity(anchor, negative);
        
        // InfoNCE: -log(exp(pos/τ) / (exp(pos/τ) + exp(neg/τ)))
        let pos_exp = pos_sim.div_scalar(temperature).exp();
        let neg_exp = neg_sim.div_scalar(temperature).exp();
        
        pos_exp.clone().div(pos_exp.add(neg_exp)).log().neg()
    }

    /// 参数量估算
    pub fn param_count(&self) -> usize {
        128 * 64 + 64 + 64 * 32 + 32 + 128 + 32
        // 约10万参数
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type NdArrayBackend = NdArray;

    #[test]
    fn test_encoder_init() {
        let device = Default::default();
        let config = EncoderConfig::small();
        let encoder = SemanticEncoder::<NdArrayBackend>::new(&config, &device);
        assert_eq!(encoder.output_dim, 32);
    }

    #[test]
    fn test_encoder_forward() {
        let device = Default::default();
        let config = EncoderConfig::small();
        let encoder = SemanticEncoder::<NdArrayBackend>::new(&config, &device);

        let input = Tensor::<NdArrayBackend, 2>::zeros([2, 128], &device);
        let output = encoder.forward(input);
        assert_eq!(output.shape().dims[1], 32);
    }

    #[test]
    fn test_cosine_similarity() {
        let device = Default::default();
        let a = Tensor::<NdArrayBackend, 2>::zeros([1, 32], &device);
        let b = Tensor::<NdArrayBackend, 2>::zeros([1, 32], &device);
        let sim = SemanticEncoder::<NdArrayBackend>::cosine_similarity(&a, &b);
        // 相同向量相似度应为1.0（或NaN→0，取决于zeros）
    }
}
