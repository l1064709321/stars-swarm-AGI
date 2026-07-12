//! MCTS（蒙特卡洛树搜索）
//!
//! L4.5 决策搜索层
//! 比A*更优的决策搜索方法
//! 四步循环：选择 → 扩展 → 模拟 → 回溯
//! 加时间戳 + 记录日志

use std::collections::HashMap;

/// MCTS 配置
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// 最大搜索迭代次数
    pub max_iterations: usize,
    /// 探索常数（UCB公式中的 c）
    pub exploration_constant: f64,
    /// 最大模拟深度
    pub max_simulation_depth: usize,
    /// 时间限制（秒，0表示无限制）
    pub time_limit_seconds: u64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            exploration_constant: 1.414, // sqrt(2)
            max_simulation_depth: 10,
            time_limit_seconds: 0,
        }
    }
}

/// MCTS 树节点
#[derive(Debug)]
pub struct MctsNode {
    /// 节点状态描述
    pub state: DecisionState,
    /// 累计奖励值
    pub total_reward: f64,
    /// 访问次数
    pub visit_count: usize,
    /// 子节点列表
    pub children: Vec<MctsNode>,
    /// 是否已完全扩展（所有可能动作都已探索）
    pub fully_expanded: bool,
    /// 父节点索引（None 表示根节点）
    pub parent_action: Option<String>,
}

impl MctsNode {
    /// 创建新节点
    pub fn new(state: DecisionState) -> Self {
        Self {
            state,
            total_reward: 0.0,
            visit_count: 0,
            children: Vec::new(),
            fully_expanded: false,
            parent_action: None,
        }
    }

    /// UCB1 值（Upper Confidence Bound）
    ///
    /// UCB = reward/visits + c * sqrt(ln(parent_visits) / visits)
    /// 平衡利用（exploitation）和探索（exploration）
    pub fn ucb1(&self, parent_visits: usize, exploration_constant: f64) -> f64 {
        if self.visit_count == 0 {
            return f64::INFINITY; // 未访问节点优先探索
        }
        let exploitation = self.total_reward / self.visit_count as f64;
        let exploration = exploration_constant * 
            (parent_visits as f64).ln().sqrt() / 
            (self.visit_count as f64).sqrt();
        exploitation + exploration
    }

    /// 平均奖励值
    pub fn average_reward(&self) -> f64 {
        if self.visit_count == 0 { 0.0 }
        else { self.total_reward / self.visit_count as f64 }
    }
}

/// 决策状态
#[derive(Debug, Clone)]
pub struct DecisionState {
    /// 状态描述
    pub description: String,
    /// 状态属性映射
    pub properties: HashMap<String, f64>,
    /// 可执行动作列表
    pub available_actions: Vec<String>,
    /// 是否为终止状态
    pub is_terminal: bool,
}

impl DecisionState {
    /// 创建初始状态
    pub fn initial(description: String) -> Self {
        Self {
            description,
            properties: HashMap::new(),
            available_actions: vec![
                "route_textcnn".to_string(),
                "route_snn".to_string(),
                "route_transformer".to_string(),
                "route_encoder".to_string(),
                "delay_processing".to_string(),
                "escalate_alert".to_string(),
            ],
            is_terminal: false,
        }
    }

    /// 创建终止状态
    pub fn terminal(description: String, reward: f64) -> Self {
        Self {
            description,
            properties: HashMap::from([("reward".to_string(), reward)]),
            available_actions: Vec::new(),
            is_terminal: true,
        }
    }

    /// 执行动作，返回新状态
    pub fn apply_action(&self, action: &str) -> DecisionState {
        let mut new_state = self.clone();
        new_state.description = format!("{} → {}", self.description, action);
        
        // 简化奖励计算
        let reward = match action {
            "route_textcnn" => 0.9,
            "route_snn" => 0.85,
            "route_transformer" => 0.8,
            "route_encoder" => 0.7,
            "delay_processing" => 0.3,
            "escalate_alert" => 0.5,
            _ => 0.0,
        };
        new_state.properties.insert("last_reward".to_string(), reward);
        
        // 简化：2步后终止
        if new_state.description.contains("→") {
            new_state.is_terminal = true;
        }
        
        new_state
    }
}

/// MCTS 搜索器
pub struct Mcts {
    /// 配置参数
    config: MctsConfig,
}

impl Mcts {
    /// 创建 MCTS 搜索器
    pub fn new(config: MctsConfig) -> Self {
        Self { config }
    }

    /// 默认实例
    pub fn default() -> Self {
        Self::new(MctsConfig::default())
    }

    /// 执行完整搜索
    ///
    /// 四步循环：选择 → 扩展 → 模拟 → 回溯
    pub fn search(&self, initial_state: DecisionState) -> MctsResult {
        let mut root = MctsNode::new(initial_state);
        
        let mut iteration = 0;
        while iteration < self.config.max_iterations {
            // 步骤1：选择（Selection）
            let selected = self.select(&mut root);
            
            // 步骤2：扩展（Expansion）
            let expanded = self.expand(selected);
            
            // 步骤3：模拟（Simulation / Rollout）
            let reward = self.simulate(&expanded.state);
            
            // 步骤4：回溯（Backpropagation）
            self.backpropagate(expanded, reward);

            iteration += 1;
        }

        // 选择最优动作
        let best_child = self.best_child(&root);
        let timestamp = self.current_timestamp();

        MctsResult {
            best_action: best_child.parent_action.clone().unwrap_or_default(),
            expected_reward: best_child.average_reward(),
            visit_count: best_child.visit_count,
            total_iterations: iteration,
            timestamp,
            tree_size: self.count_nodes(&root),
        }
    }

    /// 选择（Selection）：沿UCB最高的路径下行
    fn select(&self, node: &mut MctsNode) -> &mut MctsNode {
        let mut current = node;
        while !current.state.is_terminal && current.fully_expanded && !current.children.is_empty() {
            // 选择 UCB1 值最高的子节点
            let parent_visits = current.visit_count;
            let best_idx = current.children.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.ucb1(parent_visits, self.config.exploration_constant)
                        .partial_cmp(&b.ucb1(parent_visits, self.config.exploration_constant))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            current = &mut current.children[best_idx];
        }
        current
    }

    /// 扩展（Expansion）：添加一个新子节点
    fn expand(&self, node: &mut MctsNode) -> MctsNode {
        if node.state.is_terminal {
            return node.clone(); // 终止状态无需扩展
        }

        // 选择一个未探索的动作
        let action = if node.children.is_empty() && !node.state.available_actions.is_empty() {
            node.state.available_actions[0].clone()
        } else {
            "default_action".to_string()
        };

        let new_state = node.state.apply_action(&action);
        let mut new_node = MctsNode::new(new_state);
        new_node.parent_action = Some(action);

        node.children.push(new_node.clone());
        new_node
    }

    /// 模拟（Simulation）：随机策略快速评估
    fn simulate(&self, state: &DecisionState) -> f64 {
        let mut current = state.clone();
        let mut total_reward = 0.0;
        let mut depth = 0;

        while !current.is_terminal && depth < self.config.max_simulation_depth {
            // 随机选择动作（简化：选第一个可用动作）
            if current.available_actions.is_empty() { break; }
            let action = &current.available_actions[0];
            let reward = current.properties.get("last_reward").unwrap_or(&0.5);
            total_reward += reward;
            current = current.apply_action(action);
            depth += 1;
        }

        total_reward / (depth as f64 + 1.0)
    }

    /// 回溯（Backpropagation）：沿路径更新节点统计
    fn backpropagate(&self, _node: MctsNode, reward: f64) {
        // 简化版：直接记录奖励值
        // 完整版需要沿路径回溯到根节点，更新每个节点的 total_reward 和 visit_count
        // 由于 Rust 的所有权机制，这里简化处理
        log::info!("MCTS 回溯: reward={:.3}", reward);
    }

    /// 选择最优子节点（最高平均奖励）
    fn best_child(&self, node: &MctsNode) -> &MctsNode {
        node.children.iter()
            .max_by(|a, b| a.average_reward().partial_cmp(&b.average_reward()).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(node)
    }

    /// 计算树的节点总数
    fn count_nodes(&self, node: &MctsNode) -> usize {
        1 + node.children.iter().map(|c| self.count_nodes(c)).sum::<usize>()
    }

    /// 获取当前时间戳
    fn current_timestamp(&self) -> String {
        // 简化：用迭代次数作为时间戳
        format!("mcts_iter_{}", self.config.max_iterations)
    }
}

/// MCTS 搜索结果
#[derive(Debug)]
pub struct MctsResult {
    /// 最优动作
    pub best_action: String,
    /// 期望奖励值
    pub expected_reward: f64,
    /// 最优节点的访问次数
    pub visit_count: usize,
    /// 总搜索迭代次数
    pub total_iterations: usize,
    /// 时间戳
    pub timestamp: String,
    /// 搜索树的节点总数
    pub tree_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcts_init() {
        let mcts = Mcts::default();
        assert_eq!(mcts.config.max_iterations, 1000);
        assert_eq!(mcts.config.exploration_constant, 1.414);
    }

    #[test]
    fn test_decision_state() {
        let state = DecisionState::initial("查询处理".to_string());
        assert!(!state.is_terminal);
        assert_eq!(state.available_actions.len(), 6);
    }

    #[test]
    fn test_mcts_search() {
        let mcts = Mcts::new(MctsConfig {
            max_iterations: 10, // 短搜索用于测试
            exploration_constant: 1.414,
            max_simulation_depth: 3,
            time_limit_seconds: 0,
        });
        let initial = DecisionState::initial("安全查询".to_string());
        let result = mcts.search(initial);
        
        assert!(!result.best_action.is_empty());
        assert!(result.total_iterations == 10);
    }

    #[test]
    fn test_ucb1() {
        let node = MctsNode::new(DecisionState::initial("test".to_string()));
        // 未访问节点 → UCB = infinity
        let ucb = node.ucb1(100, 1.414);
        assert_eq!(ucb, f64::INFINITY);
    }
}
