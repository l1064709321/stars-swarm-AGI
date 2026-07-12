//! 模型层模块：定义所有神经架构组件

pub mod moe_router;
pub mod mamba;
pub mod textcnn;
pub mod encoder;
pub mod transformer;
pub mod vae;
pub mod snn;
pub mod gnn;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::backend::Backend;

/// 统一模型接口：所有神经网络模型都实现此 trait
pub trait StarModel<B: Backend>: Module<B> {
    /// 模型名称
    fn name(&self) -> &str;
    
    /// 模型参数量估算
    fn param_count(&self) -> usize;
    
    /// 前向推理（接收输入张量，返回输出张量）
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;
}

/// 模型层级枚举，对应十层架构
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// L1 感知/路由/融合
    L1,
    /// L1 记忆
    L1Memory,
    /// L1.5 语言/语义编码
    L1_5,
    /// L3.5 因果发现/推理
    L3_5,
    /// L4.5 决策搜索
    L4_5,
    /// L5 记忆巩固/语义网络
    L5,
    /// L6 伦理判断/防御进化
    L6,
}

/// 专家类型枚举（MoE Router 路由目标）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertType {
    TextCNN,
    SNN,
    Transformer,
    SemanticEncoder,
}
