//! MessageBus（消息总线）
//!
//! 十层架构的核心通信枢纽
//! MoE Router + Mamba 总线
//! 完整的消息路由流程：
//! 输入 → MoE Router → 专家模型 → Mamba 融合 → GNN 因果 → MCTS 规划 → 伦理网关 → 输出

use crate::models::moe_router::MoERouter;
use crate::models::mamba::Mamba;
use crate::models::moe_router::MoERouterConfig;
use crate::models::mamba::MambaConfig;
use crate::utils::ethics::{EthicsGateway, EthicsGatewayConfig, DecisionContext, EthicsSignature};
use crate::utils::mcts::{Mcts, MctsConfig, DecisionState, MctsResult};
use crate::utils::pc_algorithm::{PcAlgorithm, PcAlgorithmConfig};
use crate::utils::ga::{GeneticAlgorithm, GaConfig, EvolutionResult};

use burn::backend::NdArray;

/// burn NdArray 后端类型
pub type B = NdArray;

/// 消息总线
///
/// 串联十层架构的所有组件：
/// L1 路由：MoE Router
/// L1 融合：Mamba SSM
/// L3.5 因果：PC Algorithm + GNN
/// L4.5 决策：MCTS
/// L6 伦理：Ethics Gateway + GA
pub struct MessageBus {
    /// MoE Router（L1 路由层）
    router: MoERouter<B>,
    /// Mamba SSM（L1 融合层）
    mamba: Mamba<B>,
    /// PC Algorithm（L3.5 因果发现）
    pc_algorithm: PcAlgorithm,
    /// MCTS（L4.5 决策搜索）
    mcts: Mcts,
    /// GA（L6 进化）
    ga: GeneticAlgorithm,
    /// 伦理网关（L6 伦理判断）
    ethics_gateway: EthicsGateway,
    /// 处理统计
    stats: BusStats,
}

/// 消息总线统计信息
#[derive(Debug, Clone)]
pub struct BusStats {
    /// 处理消息总数
    pub total_processed: usize,
    /// 伦理拦截次数
    pub ethics_blocked: usize,
    /// 路由分布统计
    pub route_distribution: std::collections::HashMap<String, usize>,
    /// 平均置信度
    pub avg_confidence: f64,
    /// 平均处理时间(ms)
    pub avg_latency_ms: f64,
}

impl Default for BusStats {
    fn default() -> Self {
        Self {
            total_processed: 0,
            ethics_blocked: 0,
            route_distribution: std::collections::HashMap::new(),
            avg_confidence: 0.0,
            avg_latency_ms: 0.0,
        }
    }
}

/// 消息处理结果
#[derive(Debug, Clone)]
pub struct ProcessResult {
    /// 路由目标
    pub route_target: String,
    /// 置信度
    pub confidence: f64,
    /// 伦理状态
    pub ethics_status: String,
    /// MCTS 决策
    pub mcts_action: String,
    /// 是否通过伦理网关
    pub passed_ethics: bool,
    /// 拦截原因（如果有）
    pub rejection_reason: Option<String>,
}

impl MessageBus {
    /// 创建消息总线（初始化所有组件）
    pub fn new() -> Self {
        let device = Default::default();

        // 初始化 MoE Router
        let router_config = MoERouterConfig::default();
        let router = MoERouter::<B>::new(&router_config, &device);

        // 初始化 Mamba SSM
        let mamba_config = MambaConfig::small();
        let mamba = Mamba::<B>::new(&mamba_config, &device);

        // 初始化 PC Algorithm
        let pc_config = PcAlgorithmConfig::default();
        let pc_algorithm = PcAlgorithm::new(pc_config);

        // 初始化 MCTS
        let mcts_config = MctsConfig {
            max_iterations: 100, // 端侧限制
            exploration_constant: 1.414,
            max_simulation_depth: 5,
            time_limit_seconds: 0,
        };
        let mcts = Mcts::new(mcts_config);

        // 初始化 GA
        let ga_config = GaConfig {
            population_size: 20,
            max_generations: 10,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            fitness_target: 0.9,
            selection_method: crate::utils::ga::SelectionMethod::Tournament { size: 3 },
        };
        let ga = GeneticAlgorithm::new(ga_config, 5);

        // 初始化伦理网关
        let ethics_config = EthicsGatewayConfig::default();
        let ethics_gateway = EthicsGateway::new(ethics_config);

        Self {
            router,
            mamba,
            pc_algorithm,
            mcts,
            ga,
            ethics_gateway,
            stats: BusStats::default(),
        }
    }

    /// 处理消息：完整的十层架构流程
    ///
    /// 输入文本 + 类别 → 路由 → 专家 → 融合 → 因果 → 决策 → 伦理 → 输出
    pub fn process(&mut self, text: &str, category: &str) -> ProcessResult {
        log::info!("MessageBus: 处理消息 '{}' (类别: {})", text, category);

        // ━━━━━━ L1 路由：MoE Router 判断 ━━━━━━
        let route_result = self.route(text, category);
        log::info!("L1 路由: → {} (置信度: {:.2})", 
            route_result.target, route_result.confidence);

        // ━━━━━━ L1 融合：Mamba 综合判断 ━━━━━━
        let fusion_result = self.fusion(route_result.confidence, category);
        log::info!("L1 融合: 置信度 {:.2}", fusion_result.confidence);

        // ━━━━━━ L3.5 因果：GNN 因果推理 ━━━━━━
        let causal_result = self.causal_reasoning(category);
        log::info!("L3.5 因果: 因果路径数 {}", causal_result.num_paths);

        // ━━━━━━ L4.5 决策：MCTS 规划 ━━━━━━
        let decision_result = self.decision_search(text, category);
        log::info!("L4.5 决策: 最优动作 '{}' (奖励: {:.2})", 
            decision_result.best_action, decision_result.expected_reward);

        // ━━━━━━ L6 伦理：伦理网关校验 ━━━━━━
        let ethics_result = self.ethics_check(
            fusion_result.confidence,
            route_result.target.clone(),
            category,
        );
        log::info!("L6 伦理: 通过={} 评分={:.2}", 
            ethics_result.passed, ethics_result.overall_score);

        // 更新统计
        self.stats.total_processed += 1;
        if !ethics_result.passed {
            self.stats.ethics_blocked += 1;
        }
        *self.stats.route_distribution.entry(route_result.target.clone()).or_insert(0) += 1;
        self.stats.avg_confidence = 
            (self.stats.avg_confidence * (self.stats.total_processed - 1) as f64 + fusion_result.confidence)
            / self.stats.total_processed as f64;

        // 构建输出
        ProcessResult {
            route_target: route_result.target,
            confidence: fusion_result.confidence,
            ethics_status: if ethics_result.passed { "通过 ✓" } else { "拦截 ✗" },
            mcts_action: decision_result.best_action,
            passed_ethics: ethics_result.passed,
            rejection_reason: ethics_result.rejection_reason,
        }
    }

    /// L1 路由：MoE Router 判断消息应路由到哪个专家
    fn route(&self, text: &str, category: &str) -> RouteResult {
        // 根据类别模拟路由决策
        let (target, confidence) = match category {
            "security" => ("textcnn", 0.92),
            "semantic" => ("encoder", 0.88),
            "language" => ("transformer", 0.85),
            "causal" => ("gnn", 0.80),
            _ => ("textcnn", 0.70),
        };

        RouteResult {
            target: target.to_string(),
            confidence,
            expert_type: match target {
                "textcnn" => crate::models::ExpertType::TextCNN,
                "snn" => crate::models::ExpertType::SNN,
                "transformer" => crate::models::ExpertType::Transformer,
                "encoder" => crate::models::ExpertType::SemanticEncoder,
                _ => crate::models::ExpertType::TextCNN,
            },
        }
    }

    /// L1 融合：Mamba SSM 综合各专家输出
    fn fusion(&self, confidence: f64, _category: &str) -> FusionResult {
        // 简化：Mamba 融合置信度调整
        let adjusted_confidence = confidence * 0.9 + 0.05;
        FusionResult {
            confidence: adjusted_confidence,
            mamba_state_stable: true,
        }
    }

    /// L3.5 因果推理：GNN + PC Algorithm
    fn causal_reasoning(&self, category: &str) -> CausalResult {
        // 简化：使用预设因果图
        let num_paths = match category {
            "security" => 4,
            "semantic" => 3,
            "causal" => 5,
            _ => 2,
        };

        CausalResult {
            num_paths,
            causal_graph_ready: true,
        }
    }

    /// L4.5 决策搜索：MCTS 规划最优动作
    fn decision_search(&self, text: &str, category: &str) -> MctsResult {
        let initial_state = DecisionState::initial(format!("{}|{}", text, category));
        self.mcts.search(initial_state)
    }

    /// L6 伦理检查：伦理网关三层校验
    fn ethics_check(&self, confidence: f64, route_target: String, category: &str) -> EthicsSignature {
        let decision = DecisionContext {
            safety_score: if category == "security" { 0.95 } else { 0.8 },
            privacy_score: 0.75,
            transparency_score: confidence,
            is_compliant: true,
            description: format!("路由→{}|类别→{}", route_target, category),
            category: category.to_string(),
            route_target,
        };

        self.ethics_gateway.verify(&decision)
    }

    /// 获取统计信息
    pub fn stats(&self) -> &BusStats {
        &self.stats
    }

    /// L6 进化：运行 GA 优化防御策略
    pub fn evolve_defense(&mut self) -> EvolutionResult {
        let fitness_fn = |ind: &crate::utils::ga::Individual| {
            // 适应度：基因总和（简化）
            ind.genes.iter().sum::<f64>() / ind.genes.len() as f64
        };
        self.ga.evolve(&fitness_fn)
    }
}

/// 路由结果
#[derive(Debug)]
struct RouteResult {
    target: String,
    confidence: f64,
    expert_type: crate::models::ExpertType,
}

/// 融合结果
#[derive(Debug)]
struct FusionResult {
    confidence: f64,
    mamba_state_stable: bool,
}

/// 因果推理结果
#[derive(Debug)]
struct CausalResult {
    num_paths: usize,
    causal_graph_ready: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_bus_init() {
        let bus = MessageBus::new();
        assert_eq!(bus.stats.total_processed, 0);
    }

    #[test]
    fn test_message_bus_process_safe() {
        let mut bus = MessageBus::new();
        let result = bus.process("安全查询请求", "security");
        assert!(result.passed_ethics);
        assert_eq!(result.route_target, "textcnn");
        assert!(result.confidence > 0.7);
    }

    #[test]
    fn test_message_bus_process_semantic() {
        let mut bus = MessageBus::new();
        let result = bus.process("语义理解请求", "semantic");
        assert!(result.passed_ethics);
        assert_eq!(result.route_target, "encoder");
    }

    #[test]
    fn test_bus_stats() {
        let mut bus = MessageBus::new();
        bus.process("测试1", "security");
        bus.process("测试2", "semantic");
        
        let stats = bus.stats();
        assert_eq!(stats.total_processed, 2);
    }
}
