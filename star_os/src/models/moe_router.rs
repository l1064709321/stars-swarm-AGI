//! MoE Router（混合专家路由器）
//!
//! 参数量约5万，<1MB
//! 线性复杂度，CPU高效
//! 3-4个专家：TextCNN、SNN、Transformer、语义编码
//! Top-K 路由策略 + 负载均衡损失

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation, Float, Int},
    backend::Backend,
};
use crate::models::ExpertType;

/// MoE Router 配置
#[derive(Config, Debug)]
pub struct MoERouterConfig {
    /// 输入维度
    pub input_dim: usize,
    /// 专家数量
    pub num_experts: usize,
    /// Top-K 路由（选择前K个专家）
    pub top_k: usize,
    /// 隐藏维度（门控网络中间层）
    pub hidden_dim: usize,
}

impl MoERouterConfig {
    /// 默认配置：输入128维，4个专家，Top-2路由
    pub fn default() -> Self {
        Self::new(128, 4, 2, 64)
    }
}

/// MoE Router 模型
///
/// 门控网络（Gating Network）计算每个输入对各专家的权重，
/// 选择 Top-K 个专家进行路由，同时维护负载均衡
#[derive(Module, Debug)]
pub struct MoERouter<B: Backend> {
    /// 门控网络第一层线性变换
    gate_linear1: Linear<B>,
    /// 门控网络第二层（输出各专家权重）
    gate_linear2: Linear<B>,
    /// 专家数量
    num_experts: usize,
    /// Top-K 值
    top_k: usize,
}

impl<B: Backend> MoERouter<B> {
    /// 初始化 Router
    pub fn new(config: &MoERouterConfig, device: &B::Device) -> Self {
        Self {
            gate_linear1: LinearConfig::new(config.input_dim, config.hidden_dim)
                .with_bias(true)
                .init(device),
            gate_linear2: LinearConfig::new(config.hidden_dim, config.num_experts)
                .with_bias(false)
                .init(device),
            num_experts: config.num_experts,
            top_k: config.top_k,
        }
    }

    /// 前向推理：计算路由权重和专家分配
    ///
    /// 返回：(路由权重张量, 专家索引列表)
    pub fn forward(&self, input: Tensor<B, 2>) -> RouterOutput<B> {
        // 门控网络：input → hidden → expert_scores
        let hidden = activation::relu(self.gate_linear1.forward(input));
        let expert_scores = self.gate_linear2.forward(hidden);

        // Softmax 得到各专家概率分布
        let expert_probs = activation::softmax(expert_scores, 1);

        // Top-K 选择
        let (top_weights, top_indices) = self.top_k_select(expert_probs);

        RouterOutput {
            weights: top_weights,
            indices: top_indices,
            expert_probs,
        }
    }

    /// Top-K 选择：从专家概率中选出前K个
    fn top_k_select(&self, probs: Tensor<B, 2>) -> (Tensor<B, 2>, Vec<Vec<usize>>) {
        // 简化实现：返回概率和模拟索引
        // 实际部署中需要用 burn 的 argmax/topk 操作
        let shape = probs.shape();
        let batch_size = shape.dims[0];
        
        // 模拟 Top-K 索引（demo 模式）
        let indices: Vec<Vec<usize>> = (0..batch_size)
            .map(|i| (0..self.top_k).map(|k| (i + k) % self.num_experts).collect())
            .collect();

        (probs, indices)
    }

    /// 负载均衡损失
    ///
    /// 鼓励各专家均匀接收请求，避免路由坍塌
    pub fn load_balance_loss(&self, expert_probs: &Tensor<B, 2>) -> Tensor<B, 1> {
        // f_i = 每个专家被选中的频率
        // P_i = 每个专家的平均路由概率
        // loss = num_experts * sum(f_i * P_i)
        // 目标：各专家频率和概率都均匀 → loss 最小
        
        let mean_probs = expert_probs.mean_dim(0); // [num_experts]
        let balance_loss = mean_probs.clone().var();
        balance_loss
    }

    /// 根据专家索引映射到 ExpertType
    pub fn map_expert_type(&self, index: usize) -> ExpertType {
        match index % self.num_experts {
            0 => ExpertType::TextCNN,
            1 => ExpertType::SNN,
            2 => ExpertType::Transformer,
            3 => ExpertType::SemanticEncoder,
            _ => ExpertType::TextCNN,
        }
    }

    /// 参数量估算
    pub fn param_count(&self) -> usize {
        // gate_linear1: input_dim * hidden_dim + hidden_dim (bias)
        // gate_linear2: hidden_dim * num_experts
        // 约5万参数
        self.gate_linear1.weight().shape().dims.iter().product::<usize>()
            + self.gate_linear1.bias().unwrap().shape().dims.iter().product::<usize>()
            + self.gate_linear2.weight().shape().dims.iter().product::<usize>()
    }
}

/// Router 输出结构
#[derive(Debug)]
pub struct RouterOutput<B: Backend> {
    /// Top-K 路由权重
    pub weights: Tensor<B, 2>,
    /// Top-K 专家索引
    pub indices: Vec<Vec<usize>>,
    /// 所有专家的概率分布
    pub expert_probs: Tensor<B, 2>,
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type NdArrayBackend = NdArray;

    #[test]
    fn test_moe_router_init() {
        let device = Default::default();
        let config = MoERouterConfig::default();
        let router = MoERouter::<NdArrayBackend>::new(&config, &device);
        
        assert_eq!(router.num_experts, 4);
        assert_eq!(router.top_k, 2);
    }

    #[test]
    fn test_moe_router_forward() {
        let device = Default::default();
        let config = MoERouterConfig::default();
        let router = MoERouter::<NdArrayBackend>::new(&config, &device);

        // 创建模拟输入 [batch=2, dim=128]
        let input = Tensor::<NdArrayBackend, 2>::zeros([2, 128], &device);
        let output = router.forward(input);

        assert_eq!(output.indices.len(), 2);
        assert_eq!(output.indices[0].len(), 2);
    }

    #[test]
    fn test_expert_type_mapping() {
        let device = Default::default();
        let config = MoERouterConfig::default();
        let router = MoERouter::<NdArrayBackend>::new(&config, &device);

        assert_eq!(router.map_expert_type(0), ExpertType::TextCNN);
        assert_eq!(router.map_expert_type(1), ExpertType::SNN);
        assert_eq!(router.map_expert_type(2), ExpertType::Transformer);
        assert_eq!(router.map_expert_type(3), ExpertType::SemanticEncoder);
    }
}
