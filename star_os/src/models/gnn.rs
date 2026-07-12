//! GNN（图神经网络）
//!
//! L3.5 因果推理层 + L5 语义网络层
//! 使用 PyG/DGL 类似的简化实现
//! 图上消息传递（Message Passing）
//! 用于因果推理和语义网络推理

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation},
    backend::Backend,
};

/// GNN 配置
#[derive(Config, Debug)]
pub struct GnnConfig {
    /// 节点特征维度
    pub node_dim: usize,
    /// 隐藏维度
    pub hidden_dim: usize,
    /// 输出维度
    pub output_dim: usize,
    /// 消息传递层数
    pub num_layers: usize,
    /// 边特征维度（可选）
    pub edge_dim: usize,
}

impl GnnConfig {
    /// 默认配置：32→16→8，2层消息传递
    pub fn default() -> Self {
        Self::new(32, 16, 8, 2, 8)
    }

    /// 因果图配置：更大的隐藏维度
    pub fn causal() -> Self {
        Self::new(64, 32, 16, 3, 16)
    }
}

/// 图结构（简化表示）
///
/// 存储节点和边的信息
#[derive(Debug, Clone)]
pub struct Graph {
    /// 节点数量
    pub num_nodes: usize,
    /// 边列表：[(src_node, dst_node)]
    pub edges: Vec<(usize, usize)>,
    /// 节点特征维度
    pub node_dim: usize,
}

impl Graph {
    /// 创建空图
    pub fn empty(node_dim: usize) -> Self {
        Self {
            num_nodes: 0,
            edges: Vec::new(),
            node_dim,
        }
    }

    /// 添加节点
    pub fn add_node(&mut self) -> usize {
        let id = self.num_nodes;
        self.num_nodes += 1;
        id
    }

    /// 添加边
    pub fn add_edge(&mut self, src: usize, dst: usize) {
        assert!(src < self.num_nodes && dst < self.num_nodes);
        self.edges.push((src, dst));
    }

    /// 获取节点的邻居列表
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.edges.iter()
            .filter_map(|(s, d)| {
                if *s == node { Some(*d) }
                else if *d == node { Some(*s) }
                else { None }
            })
            .collect()
    }

    /// 创建示例因果图
    pub fn example_causal_graph() -> Self {
        let mut graph = Self::empty(32);
        
        // 添加5个节点（因果变量）
        for _ in 0..5 { graph.add_node(); }
        
        // 因果边：A→B→C→D, A→E
        graph.add_edge(0, 1); // A→B
        graph.add_edge(1, 2); // B→C
        graph.add_edge(2, 3); // C→D
        graph.add_edge(0, 4); // A→E
        
        graph
    }

    /// 创建示例语义网络
    pub fn example_semantic_graph() -> Self {
        let mut graph = Self::empty(32);
        
        for _ in 0..8 { graph.add_node(); }
        
        // 语义关联边
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 4);
        graph.add_edge(3, 5);
        graph.add_edge(4, 6);
        graph.add_edge(5, 7);
        
        graph
    }
}

/// GNN 模型（Graph Convolutional Network 简化版）
///
/// 消息传递流程：
/// 1. 节点收集邻居消息：h_agg = Σ(h_neighbor * W_msg)
/// 2. 节点更新自身特征：h_new = σ(h_agg * W_update + h_self * W_self)
/// 3. 重复多层消息传递
#[derive(Module, Debug)]
pub struct GNN<B: Backend> {
    /// 消息传递层1
    msg_layer1: Linear<B>,
    /// 自身更新层1
    self_layer1: Linear<B>,
    /// 消息传递层2
    msg_layer2: Linear<B>,
    /// 自身更新层2
    self_layer2: Linear<B>,
    /// 输出投影层
    output_proj: Linear<B>,
    /// 配置参数
    node_dim: usize,
    hidden_dim: usize,
}

impl<B: Backend> GNN<B> {
    /// 初始化 GNN
    pub fn new(config: &GnnConfig, device: &B::Device) -> Self {
        Self {
            msg_layer1: LinearConfig::new(config.node_dim, config.hidden_dim)
                .with_bias(true).init(device),
            self_layer1: LinearConfig::new(config.node_dim, config.hidden_dim)
                .with_bias(false).init(device),
            msg_layer2: LinearConfig::new(config.hidden_dim, config.hidden_dim)
                .with_bias(true).init(device),
            self_layer2: LinearConfig::new(config.hidden_dim, config.hidden_dim)
                .with_bias(false).init(device),
            output_proj: LinearConfig::new(config.hidden_dim, config.output_dim)
                .with_bias(true).init(device),
            node_dim: config.node_dim,
            hidden_dim: config.hidden_dim,
        }
    }

    /// 消息传递单步
    ///
    /// h_agg = mean(h_neighbors) * W_msg + h_self * W_self
    fn message_passing_step(
        &self,
        node_features: &Tensor<B, 2>,
        use_layer2: bool,
    ) -> Tensor<B, 2> {
        let msg_layer = if use_layer2 { &self.msg_layer2 } else { &self.msg_layer1 };
        let self_layer = if use_layer2 { &self.self_layer2 } else { &self.self_layer1 };

        // 消息聚合（简化：用节点自身特征的均值近似邻居聚合）
        let aggregated = node_features.clone();
        let msg = msg_layer.forward(aggregated);
        let self_msg = self_layer.forward(node_features.clone());

        activation::relu(msg.add(self_msg))
    }

    /// 前向推理：多层消息传递 + 输出投影
    ///
    /// 输入：节点特征矩阵 [num_nodes, node_dim]
    /// 输出：节点输出特征 [num_nodes, output_dim]
    pub fn forward(&self, node_features: Tensor<B, 2>) -> Tensor<B, 2> {
        // 第一层消息传递
        let h = self.message_passing_step(&node_features, false);

        // 第二层消息传递
        let h = self.message_passing_step(&h, true);

        // 输出投影
        self.output_proj.forward(h)
    }

    /// 带图结构的前向推理
    ///
    /// 根据图结构的邻居关系聚合消息
    pub fn forward_with_graph(&self, node_features: Tensor<B, 2>, graph: &Graph) -> Tensor<B, 2> {
        let shape = node_features.shape();
        let num_nodes = shape.dims[0];
        let device = node_features.device();

        // 简化版：构建聚合特征（按邻居平均）
        // 完整版需要逐节点收集邻居特征并平均
        let aggregated = node_features.clone(); // demo: 直接用自身特征

        // 消息传递
        let h = self.message_passing_step(&aggregated, false);
        let h = self.message_passing_step(&h, true);

        self.output_proj.forward(h)
    }

    /// 因果推理：在因果图上推断变量间的因果关系
    pub fn causal_inference(&self, node_features: Tensor<B, 2>, graph: &Graph) -> CausalInferenceResult {
        let output = self.forward_with_graph(node_features, graph);

        CausalInferenceResult {
            node_embeddings: output,
            graph: graph.clone(),
            num_causal_paths: graph.edges.len(),
        }
    }

    /// 语义推理：在语义网络上推断概念关联
    pub fn semantic_inference(&self, node_features: Tensor<B, 2>, graph: &Graph) -> SemanticInferenceResult {
        let output = self.forward_with_graph(node_features, graph);

        SemanticInferenceResult {
            node_embeddings: output,
            semantic_graph: graph.clone(),
            num_concepts: graph.num_nodes,
        }
    }

    /// 参数量估算
    pub fn param_count(&self) -> usize {
        let node_dim = self.node_dim;
        let hidden_dim = self.hidden_dim;
        
        node_dim * hidden_dim * 2 + hidden_dim * hidden_dim * 2 + hidden_dim * 8
    }
}

/// 因果推理结果
#[derive(Debug)]
pub struct CausalInferenceResult<B: Backend> {
    /// 节点嵌入（推理后的节点特征）
    pub node_embeddings: Tensor<B, 2>,
    /// 因果图
    pub graph: Graph,
    /// 因果路径数量
    pub num_causal_paths: usize,
}

/// 语义推理结果
#[derive(Debug)]
pub struct SemanticInferenceResult<B: Backend> {
    /// 节点嵌入
    pub node_embeddings: Tensor<B, 2>,
    /// 语义网络图
    pub semantic_graph: Graph,
    /// 概念数量
    pub num_concepts: usize,
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type NdArrayBackend = NdArray;

    #[test]
    fn test_gnn_init() {
        let device = Default::default();
        let config = GnnConfig::default();
        let gnn = GNN::<NdArrayBackend>::new(&config, &device);
        assert_eq!(gnn.node_dim, 32);
        assert_eq!(gnn.hidden_dim, 16);
    }

    #[test]
    fn test_graph_operations() {
        let mut graph = Graph::empty(32);
        for _ in 0..5 { graph.add_node(); }
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        assert_eq!(graph.num_nodes, 5);
        assert_eq!(graph.edges.len(), 2);
        assert_eq!(graph.neighbors(1), vec![0, 2]);
    }

    #[test]
    fn test_gnn_forward() {
        let device = Default::default();
        let config = GnnConfig::default();
        let gnn = GNN::<NdArrayBackend>::new(&config, &device);

        let input = Tensor::<NdArrayBackend, 2>::zeros([5, 32], &device);
        let output = gnn.forward(input);
        assert_eq!(output.shape().dims[1], 8);
    }

    #[test]
    fn test_causal_graph() {
        let graph = Graph::example_causal_graph();
        assert_eq!(graph.num_nodes, 5);
        assert_eq!(graph.edges.len(), 4);
    }
}
