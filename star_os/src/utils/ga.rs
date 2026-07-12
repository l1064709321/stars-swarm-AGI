//! GA（遗传算法 - Evolver）
//!
//! L6 防御+进化层
//! 遗传算法优化代码/策略
//! 选择 → 交叉 → 变异 → 评估
//! 用于防御策略进化

use rand::Rng;
use std::collections::HashMap;

/// GA 配置
#[derive(Debug, Clone)]
pub struct GaConfig {
    /// 种群大小
    pub population_size: usize,
    /// 最大进化代数
    pub max_generations: usize,
    /// 交叉率
    pub crossover_rate: f64,
    /// 变异率
    pub mutation_rate: f64,
    /// 选择方法
    pub selection_method: SelectionMethod,
    /// 适应度目标
    pub fitness_target: f64,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_generations: 100,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            selection_method: SelectionMethod::Tournament { size: 3 },
            fitness_target: 0.95,
        }
    }
}

/// 选择方法
#[derive(Debug, Clone)]
pub enum SelectionMethod {
    /// 轮盘赌选择
    RouletteWheel,
    /// 锦标赛选择
    Tournament { size: usize },
    /// 排名选择
    RankSelection,
}

/// 个体（染色体）
///
/// 表示一个策略/配置的编码
#[derive(Debug, Clone)]
pub struct Individual {
    /// 基因编码（策略参数）
    pub genes: Vec<f64>,
    /// 适应度值
    pub fitness: f64,
    /// 个体ID
    pub id: usize,
    /// 世代
    pub generation: usize,
    /// 策略描述
    pub description: String,
}

impl Individual {
    /// 创建随机个体
    pub fn random(gene_length: usize, rng: &mut impl Rng) -> Self {
        Self {
            genes: (0..gene_length).map(|_| rng.gen_range(0.0..1.0)).collect(),
            fitness: 0.0,
            id: 0,
            generation: 0,
            description: String::new(),
        }
    }

    /// 交叉（Crossover）：两个个体产生后代
    pub fn crossover(&self, other: &Individual, rng: &mut impl Rng) -> (Individual, Individual) {
        let crossover_point = rng.gen_range(1..self.genes.len());
        
        let child1_genes = self.genes[..crossover_point].iter()
            .chain(other.genes[crossover_point..].iter())
            .cloned().collect();
        let child2_genes = other.genes[..crossover_point].iter()
            .chain(self.genes[crossover_point..].iter())
            .cloned().collect();

        let child1 = Individual {
            genes: child1_genes,
            fitness: 0.0,
            id: 0,
            generation: self.generation + 1,
            description: format!("交叉后代({}×{})", self.id, other.id),
        };
        let child2 = Individual {
            genes: child2_genes,
            fitness: 0.0,
            id: 0,
            generation: self.generation + 1,
            description: format!("交叉后代({}×{})", other.id, self.id),
        };

        (child1, child2)
    }

    /// 变异（Mutation）：随机改变基因
    pub fn mutate(&mut self, mutation_rate: f64, rng: &mut impl Rng) {
        for gene in &mut self.genes {
            if rng.gen_bool(mutation_rate) {
                *gene = rng.gen_range(0.0..1.0); // 随机重置
            }
        }
    }

    /// 解码基因到防御策略
    pub fn decode_to_strategy(&self) -> DefenseStrategy {
        DefenseStrategy {
            routing_threshold: self.genes.get(0).unwrap_or(&0.5),
            ethics_weight: self.genes.get(1).unwrap_or(&0.3),
            timeout_ms: (self.genes.get(2).unwrap_or(&0.5) * 5000.0) as u64,
            retry_limit: (self.genes.get(3).unwrap_or(&0.2) * 5.0) as usize,
            alert_level: if self.genes.get(4).unwrap_or(&0.5) > 0.7 { AlertLevel::High } 
                         else if self.genes.get(4).unwrap_or(&0.5) > 0.3 { AlertLevel::Medium }
                         else { AlertLevel::Low },
            description: self.description.clone(),
        }
    }
}

/// 防御策略
#[derive(Debug, Clone)]
pub struct DefenseStrategy {
    /// 路由阈值
    pub routing_threshold: &f64,
    /// 伦理权重
    pub ethics_weight: &f64,
    /// 超时时间(ms)
    pub timeout_ms: u64,
    /// 重试限制
    pub retry_limit: usize,
    /// 警报级别
    pub alert_level: AlertLevel,
    /// 策略描述
    pub description: String,
}

/// 警报级别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertLevel {
    Low,
    Medium,
    High,
}

/// GA 遗传算法引擎
pub struct GeneticAlgorithm {
    /// 配置参数
    config: GaConfig,
    /// 当前种群
    population: Vec<Individual>,
    /// 基因长度
    gene_length: usize,
    /// 当前世代
    current_generation: usize,
    /// 进化历史
    history: Vec<GenerationRecord>,
}

/// 世代记录
#[derive(Debug)]
pub struct GenerationRecord {
    /// 世代编号
    pub generation: usize,
    /// 最佳适应度
    pub best_fitness: f64,
    /// 平均适应度
    pub avg_fitness: f64,
    /// 最佳个体描述
    pub best_individual_desc: String,
}

impl GeneticAlgorithm {
    /// 创建 GA 实例
    pub fn new(config: GaConfig, gene_length: usize) -> Self {
        let mut rng = rand::thread_rng();
        let population: Vec<Individual> = (0..config.population_size)
            .map(|i| {
                let mut ind = Individual::random(gene_length, &mut rng);
                ind.id = i;
                ind
            })
            .collect();

        Self {
            config,
            population,
            gene_length,
            current_generation: 0,
            history: Vec::new(),
        }
    }

    /// 运行进化
    ///
    /// 选择 → 交叉 → 变异 → 评估，循环 max_generations 次
    pub fn evolve(&mut self, fitness_fn: &dyn Fn(&Individual) -> f64) -> EvolutionResult {
        // 初始评估
        self.evaluate_population(fitness_fn);

        for gen in 0..self.config.max_generations {
            self.current_generation = gen;

            // 选择
            let parents = self.select_parents();

            // 交叉 + 变异 → 产生新种群
            let new_population = self.reproduce(&parents);

            // 替换种群
            self.population = new_population;

            // 评估新种群
            self.evaluate_population(fitness_fn);

            // 记录历史
            self.record_generation();

            // 检查是否达到目标
            let best_fitness = self.population.iter()
                .map(|i| i.fitness)
                .max()
                .unwrap_or(0.0);
            
            if best_fitness >= self.config.fitness_target {
                log::info!("GA: 在第{}代达到适应度目标 {:.3}", gen, best_fitness);
                break;
            }
        }

        // 返回最优个体
        let best = self.best_individual();
        EvolutionResult {
            best_individual: best.clone(),
            best_fitness: best.fitness,
            total_generations: self.current_generation + 1,
            history: self.history.clone(),
            population_size: self.config.population_size,
        }
    }

    /// 评估种群适应度
    fn evaluate_population(&mut self, fitness_fn: &dyn Fn(&Individual) -> f64) {
        for ind in &mut self.population {
            ind.fitness = fitness_fn(ind);
        }
    }

    /// 选择父代
    fn select_parents(&self) -> Vec<Individual> {
        let mut rng = rand::thread_rng();
        let num_parents = self.config.population_size / 2;
        
        match &self.config.selection_method {
            SelectionMethod::Tournament { size } => {
                (0..num_parents)
                    .map(|_| self.tournament_select(*size, &mut rng))
                    .collect()
            }
            SelectionMethod::RouletteWheel => {
                (0..num_parents)
                    .map(|_| self.roulette_select(&mut rng))
                    .collect()
            }
            SelectionMethod::RankSelection => {
                (0..num_parents)
                    .map(|_| self.rank_select(&mut rng))
                    .collect()
            }
        }
    }

    /// 锦标赛选择
    fn tournament_select(&self, size: usize, rng: &mut impl Rng) -> Individual {
        let candidates: Vec<&Individual> = (0..size)
            .map(|_| &self.population[rng.gen_range(0..self.population.len())])
            .collect();
        candidates.iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
            .clone()
    }

    /// 轮盘赌选择
    fn roulette_select(&self, rng: &mut impl Rng) -> Individual {
        let total_fitness = self.population.iter().map(|i| i.fitness).sum::<f64>();
        if total_fitness == 0.0 {
            return self.population[rng.gen_range(0..self.population.len())].clone();
        }
        let mut target = rng.gen_range(0.0..total_fitness);
        for ind in &self.population {
            target -= ind.fitness;
            if target <= 0.0 { return ind.clone(); }
        }
        self.population.last().unwrap().clone()
    }

    /// 排名选择
    fn rank_select(&self, rng: &mut impl Rng) -> Individual {
        let mut sorted = self.population.clone();
        sorted.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        let rank_weights: Vec<f64> = sorted.iter().enumerate()
            .map(|(i, _)| (sorted.len() - i) as f64)
            .collect();
        let total_weight = rank_weights.iter().sum::<f64>();
        let mut target = rng.gen_range(0.0..total_weight);
        for (i, ind) in sorted.iter().enumerate() {
            target -= rank_weights[i];
            if target <= 0.0 { return ind.clone(); }
        }
        sorted.last().unwrap().clone()
    }

    /// 繁殖：交叉 + 变异
    fn reproduce(&self, parents: &[Individual]) -> Vec<Individual> {
        let mut rng = rand::thread_rng();
        let mut new_population = Vec::new();

        // 保留精英（top 10%）
        let elite_count = self.config.population_size / 10;
        let mut sorted = self.population.clone();
        sorted.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        for i in 0..elite_count {
            let mut elite = sorted[i].clone();
            elite.id = i;
            new_population.push(elite);
        }

        // 交叉 + 变异产生其余个体
        while new_population.len() < self.config.population_size {
            let p1_idx = rng.gen_range(0..parents.len());
            let p2_idx = rng.gen_range(0..parents.len());
            
            if rng.gen_bool(self.config.crossover_rate) {
                let (child1, child2) = parents[p1_idx].crossover(&parents[p2_idx], &mut rng);
                let mut child1 = child1;
                let mut child2 = child2;
                child1.mutate(self.config.mutation_rate, &mut rng);
                child2.mutate(self.config.mutation_rate, &mut rng);
                child1.id = new_population.len();
                child2.id = new_population.len() + 1;
                new_population.push(child1);
                if new_population.len() < self.config.population_size {
                    new_population.push(child2);
                }
            } else {
                let mut child = parents[p1_idx].clone();
                child.mutate(self.config.mutation_rate, &mut rng);
                child.id = new_population.len();
                child.generation = self.current_generation + 1;
                new_population.push(child);
            }
        }

        new_population
    }

    /// 记录世代信息
    fn record_generation(&mut self) {
        let best_fitness = self.population.iter().map(|i| i.fitness).max().unwrap_or(0.0);
        let avg_fitness = self.population.iter().map(|i| i.fitness).sum::<f64>() / self.population.len() as f64;
        let best = self.best_individual();

        self.history.push(GenerationRecord {
            generation: self.current_generation,
            best_fitness,
            avg_fitness,
            best_individual_desc: best.description.clone(),
        });
    }

    /// 获取最优个体
    fn best_individual(&self) -> &Individual {
        self.population.iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
    }
}

/// 进化结果
#[derive(Debug)]
pub struct EvolutionResult {
    /// 最优个体
    pub best_individual: Individual,
    /// 最优适应度
    pub best_fitness: f64,
    /// 总进化代数
    pub total_generations: usize,
    /// 进化历史
    pub history: Vec<GenerationRecord>,
    /// 种群大小
    pub population_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_individual_random() {
        let mut rng = rand::thread_rng();
        let ind = Individual::random(5, &mut rng);
        assert_eq!(ind.genes.len(), 5);
        assert_eq!(ind.fitness, 0.0);
    }

    #[test]
    fn test_crossover() {
        let mut rng = rand::thread_rng();
        let p1 = Individual { genes: vec![1.0, 1.0, 1.0, 1.0, 1.0], fitness: 0.8, id: 1, generation: 0, description: "p1".to_string() };
        let p2 = Individual { genes: vec![0.0, 0.0, 0.0, 0.0, 0.0], fitness: 0.6, id: 2, generation: 0, description: "p2".to_string() };
        let (c1, c2) = p1.crossover(&p2, &mut rng);
        assert_eq!(c1.genes.len(), 5);
        assert_eq!(c2.genes.len(), 5);
    }

    #[test]
    fn test_mutation() {
        let mut rng = rand::thread_rng();
        let mut ind = Individual { genes: vec![0.5, 0.5, 0.5], fitness: 0.0, id: 0, generation: 0, description: "".to_string() };
        ind.mutate(1.0, &mut rng); // 100%变异率
        // 基因应该全部改变
    }

    #[test]
    fn test_ga_evolve() {
        let config = GaConfig {
            population_size: 20,
            max_generations: 10,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            selection_method: SelectionMethod::Tournament { size: 3 },
            fitness_target: 0.99,
        };
        let mut ga = GeneticAlgorithm::new(config, 5);
        
        // 简单适应度函数：基因之和越大越好
        let fitness_fn = |ind: &Individual| ind.genes.iter().sum::<f64>() / ind.genes.len() as f64;
        let result = ga.evolve(&fitness_fn);
        
        assert!(result.best_fitness > 0.0);
        assert!(result.total_generations > 0);
    }
}
