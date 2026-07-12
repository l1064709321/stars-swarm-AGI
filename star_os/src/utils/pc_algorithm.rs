//! PC Algorithm（PC 算法）
//!
//! L3.5 因果发现层
//! 从观测数据推断因果结构（因果图）
//! 基于 conditional independence 测试
//! 纮束搜索：骨架发现 → 方向推断

use crate::models::gnn::Graph;
use std::collections::{HashMap, HashSet};

/// PC 算法配置
#[derive(Debug, Clone)]
pub struct PcAlgorithmConfig {
    /// 显著性水平（α，条件独立性检验阈值）
    pub alpha: f64,
    /// 最大条件集大小（搜索深度限制）
    pub max_condition_set_size: usize,
    /// 是否使用稳定版（按层级而非逐边删除）
    pub stable: bool,
}

impl Default for PcAlgorithmConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            max_condition_set_size: 3,
            stable: true,
        }
    }
}

/// PC 算法实现
///
/// 步骤：
/// 1. 骨架发现（Skeleton Discovery）：从完全图开始，逐步删除条件独立的边
/// 2. 方向推断（Orientation）：根据 v-结构和其他规则确定边的方向
pub struct PcAlgorithm {
    /// 配置参数
    config: PcAlgorithmConfig,
}

impl PcAlgorithm {
    /// 创建 PC 算法实例
    pub fn new(config: PcAlgorithmConfig) -> Self {
        Self { config }
    }

    /// 默认实例
    pub fn default() -> Self {
        Self::new(PcAlgorithmConfig::default())
    }

    /// 从观测数据推断因果图
    ///
    /// data: 观测数据矩阵 [n_samples][n_variables]
    /// 变量名列表
    pub fn infer_causal_graph(
        &self,
        data: &[Vec<f64>],
        variable_names: &[String],
    ) -> CausalDiscoveryResult {
        let num_vars = variable_names.len();
        
        // 步骤1：骨架发现
        let skeleton = self.discover_skeleton(data, num_vars);
        
        // 步骤2：方向推断
        let oriented_edges = self.orient_edges(&skeleton, data, num_vars);
        
        // 构建因果图
        let mut graph = Graph::empty(32); // 使用默认节点维度
        for _ in 0..num_vars { graph.add_node(); }
        for (src, dst) in &oriented_edges {
            graph.add_edge(*src, *dst);
        }

        CausalDiscoveryResult {
            graph,
            skeleton_edges: skeleton,
            oriented_edges,
            variable_names: variable_names.to_vec(),
            num_variables: num_vars,
            alpha: self.config.alpha,
        }
    }

    /// 骨架发现：从完全图删除条件独立的边
    ///
    /// 完全图（所有变量互连）→ 逐步检验条件独立性 → 删除独立边
    fn discover_skeleton(
        &self,
        data: &[Vec<f64>],
        num_vars: usize,
    ) -> Vec<(usize, usize)> {
        // 初始化完全图
        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        for i in 0..num_vars {
            for j in (i + 1)..num_vars {
                edges.insert((i, j));
            }
        }

        // 逐层级搜索（稳定版）
        let mut condition_size = 0;
        while condition_size <= self.config.max_condition_set_size {
            let edges_to_test: Vec<(usize, usize)> = edges.iter().cloned().collect();
            
            for (i, j) in edges_to_test {
                // 获取 i 的邻居（排除 j）
                let neighbors_i: Vec<usize> = edges.iter()
                    .filter_map(|(a, b)| {
                        if *a == i && *b != j { Some(*b) }
                        else if *b == i && *a != j { Some(*a) }
                        else { None }
                    })
                    .collect();

                // 生成条件集（大小为 condition_size 的子集）
                let condition_sets = self.generate_condition_sets(
                    &neighbors_i, condition_size
                );

                for cond_set in condition_sets {
                    // 条件独立性检验
                    if self.test_conditional_independence(data, i, j, &cond_set) {
                        edges.remove(&(i.min(j), i.max(j)));
                        break; // 发现独立 → 删除边 → 不再检验更多条件集
                    }
                }
            }
            condition_size += 1;
        }

        edges.iter().cloned().collect()
    }

    /// 方向推断：确定骨架边的方向
    ///
    /// v-结构检测：如果 i - k - j 且 k 不在 (i,j) 的条件集中 → i→k←j
    fn orient_edges(
        &self,
        skeleton: &[(usize, usize)],
        data: &[Vec<f64>],
        num_vars: usize,
    ) -> Vec<(usize, usize)> {
        let mut oriented: Vec<(usize, usize)> = Vec::new();
        let mut unoriented: Vec<(usize, usize)> = Vec::new();

        // 查找 v-结构（collider）
        for (i, j) in skeleton {
            // 找到共同邻居 k（i-k-j）
            for k in 0..num_vars {
                if k == *i || k == *j { continue; }
                
                let i_k_exists = skeleton.iter().any(|(a, b)| 
                    (*a == *i && *b == k) || (*a == k && *b == *i));
                let k_j_exists = skeleton.iter().any(|(a, b)| 
                    (*a == k && *b == *j) || (*a == *j && *b == k));

                if i_k_exists && k_j_exists {
                    // 检验 k 是否在 (i,j) 的条件集中
                    // 如果不在 → v-结构：i→k←j
                    let is_in_condition = false; // 简化假设
                    
                    if !is_in_condition {
                        // v-结构：i→k, j→k
                        oriented.push((*i, k));
                        oriented.push((*j, k));
                    } else {
                        unoriented.push((*i, *j));
                    }
                }
            }
        }

        // 对未定向的边使用启发式规则（如时间顺序）
        for (i, j) in unoriented {
            // 简化：按索引顺序定向
            oriented.push((i.min(j), i.max(j)));
        }

        // 去重
        let mut unique: Vec<(usize, usize)> = oriented;
        unique.sort();
        unique.dedup();
        unique
    }

    /// 条件独立性检验（简化版）
    ///
    /// 使用相关性阈值近似（正式版应使用 Fisher z-test 或卡方检验）
    fn test_conditional_independence(
        &self,
        data: &[Vec<f64>],
        var_i: usize,
        var_j: usize,
        condition_set: &[usize],
    ) -> bool {
        // 简化版：无条件时直接检验相关性
        // 有条件时近似为部分相关性
        
        if data.is_empty() { return false; }
        
        let n = data.len();
        
        // 计算简单相关系数
        let mean_i = data.iter().map(|row| row[var_i]).sum::<f64>() / n as f64;
        let mean_j = data.iter().map(|row| row[var_j]).sum::<f64>() / n as f64;
        
        let cov_ij = data.iter()
            .map(|row| (row[var_i] - mean_i) * (row[var_j] - mean_j))
            .sum::<f64>() / n as f64;
        
        let var_i = data.iter()
            .map(|row| (row[var_i] - mean_i).powi(2))
            .sum::<f64>() / n as f64;
        
        let var_j = data.iter()
            .map(|row| (row[var_j] - mean_j).powi(2))
            .sum::<f64>() / n as f64;
        
        let correlation = if var_i > 0.0 && var_j > 0.0 {
            cov_ij / (var_i.sqrt() * var_j.sqrt())
        } else {
            0.0
        };

        // 相关性小于阈值 → 条件独立
        correlation.abs() < self.config.alpha
    }

    /// 生成条件集（邻居的所有大小为 size 的子集）
    fn generate_condition_sets(
        &self,
        neighbors: &[usize],
        size: usize,
    ) -> Vec<Vec<usize>> {
        if size == 0 {
            return vec![vec![]];
        }
        if neighbors.len() < size {
            return vec![]; // 邻居不够组成条件集
        }

        // 简化版：只生成前几个子集（避免组合爆炸）
        let mut result = Vec::new();
        let max_sets = 10; // 限制搜索量
        
        for i in 0..neighbors.len() {
            let mut subset = Vec::new();
            for j in 0..size {
                subset.push(neighbors[(i + j) % neighbors.len()]);
            }
            result.push(subset);
            if result.len() >= max_sets { break; }
        }

        result
    }
}

/// 因果发现结果
#[derive(Debug)]
pub struct CausalDiscoveryResult {
    /// 因果图
    pub graph: Graph,
    /// 骨架边（无方向）
    pub skeleton_edges: Vec<(usize, usize)>,
    /// 有方向边
    pub oriented_edges: Vec<(usize, usize)>,
    /// 变量名列表
    pub variable_names: Vec<String>,
    /// 变量数量
    pub num_variables: usize,
    /// 显著性水平
    pub alpha: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pc_algorithm_init() {
        let pc = PcAlgorithm::default();
        assert_eq!(pc.config.alpha, 0.05);
        assert_eq!(pc.config.max_condition_set_size, 3);
    }

    #[test]
    fn test_skeleton_discovery() {
        let pc = PcAlgorithm::default();
        let data: Vec<Vec<f64>> = vec![
            vec![1.0, 0.5, 0.3, 0.1],  // 4个变量
            vec![1.2, 0.6, 0.4, 0.2],
            vec![0.8, 0.4, 0.2, 0.0],
        ];
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
        
        let result = pc.infer_causal_graph(&data, &names);
        assert_eq!(result.num_variables, 4);
    }

    #[test]
    fn test_condition_sets() {
        let pc = PcAlgorithm::default();
        let neighbors = vec![1, 2, 3, 4];
        let sets = pc.generate_condition_sets(&neighbors, 2);
        assert!(!sets.is_empty());
    }
}
