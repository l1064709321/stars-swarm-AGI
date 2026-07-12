//! ModelScheduler（模型调度器）
//!
//! AidLux 端侧模型管理
//! 高频模型常驻内存，低频模型按需加载
//! 内存水位监控，超过阈值自动卸载
//! 推理超时保护

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 模型加载策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStrategy {
    /// 常驻内存（高频模型）
    Resident,
    /// 按需加载（低频但关键）
    OnDemand,
}

/// 模型状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    /// 已加载就绪
    Ready,
    /// 正在加载
    Loading,
    /// 未加载（已卸载）
    Unloaded,
    /// 加载失败
    Failed,
}

/// 模型注册信息
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    /// 模型名称
    pub name: String,
    /// 加载策略
    pub strategy: LoadStrategy,
    /// 当前状态
    pub state: ModelState,
    /// 估算内存占用(bytes)
    pub estimated_memory: usize,
    /// 加载优先级（数值越小优先级越高）
    pub priority: usize,
    /// 推理超时(ms)
    pub timeout_ms: u64,
    /// 加载次数统计
    pub load_count: usize,
    /// 推理次数统计
    pub inference_count: usize,
}

impl ModelRegistry {
    /// 创建常驻模型注册
    pub fn resident(name: &str, memory: usize, priority: usize) -> Self {
        Self {
            name: name.to_string(),
            strategy: LoadStrategy::Resident,
            state: ModelState::Unloaded,
            estimated_memory: memory,
            priority,
            timeout_ms: 500,
            load_count: 0,
            inference_count: 0,
        }
    }

    /// 创建按需加载模型注册
    pub fn on_demand(name: &str, memory: usize, priority: usize) -> Self {
        Self {
            name: name.to_string(),
            strategy: LoadStrategy::OnDemand,
            state: ModelState::Unloaded,
            estimated_memory: memory,
            priority,
            timeout_ms: 3000, // 按需模型超时更长
            load_count: 0,
            inference_count: 0,
        }
    }
}

/// 模型调度器配置
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// 最大内存限制(bytes)
    pub max_memory: usize,
    /// 内存水位警告阈值（占总内存百分比）
    pub memory_warning_threshold: f64,
    /// 内存水位卸载阈值（占总内存百分比）
    pub memory_unload_threshold: f64,
    /// 预热模式：启动时是否自动加载常驻模型
    pub warmup_on_start: bool,
    /// 推理超时保护(ms)
    pub default_timeout_ms: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_memory: 6 * 1024 * 1024 * 1024, // 6GB（含临时缓存）
            memory_warning_threshold: 0.7,
            memory_unload_threshold: 0.85,
            warmup_on_start: true,
            default_timeout_ms: 1000,
        }
    }
}

/// 模型调度器
///
/// 负责管理模型加载/卸载、内存监控、推理调度
pub struct ModelScheduler {
    /// 配置参数
    config: SchedulerConfig,
    /// 已注册的模型
    models: HashMap<String, ModelRegistry>,
    /// 当前内存占用
    current_memory: usize,
    /// 是否已预热
    warmed_up: bool,
}

impl ModelScheduler {
    /// 创建调度器
    pub fn new() -> Self {
        let config = SchedulerConfig::default();
        let mut scheduler = Self {
            config,
            models: HashMap::new(),
            current_memory: 0,
            warmed_up: false,
        };

        // 注册所有模型
        scheduler.register_all_models();

        // 预热常驻模型
        if config.warmup_on_start {
            scheduler.warmup();
        }

        scheduler
    }

    /// 注册所有模型
    fn register_all_models(&mut self) {
        // 常驻模型（高频使用）
        self.register(ModelRegistry::resident("textcnn", 1_000_000, 1));
        self.register(ModelRegistry::resident("ga", 500_000, 2));

        // 按需加载模型（低频但关键）
        self.register(ModelRegistry::on_demand("moe_router", 1_000_000, 3));
        self.register(ModelRegistry::on_demand("mamba", 4_000_000, 4));
        self.register(ModelRegistry::on_demand("transformer", 12_000_000, 5));
        self.register(ModelRegistry::on_demand("encoder", 4_000_000, 6));
        self.register(ModelRegistry::on_demand("gnn", 1_000_000, 7));
        self.register(ModelRegistry::on_demand("vae", 2_000_000, 8));
        self.register(ModelRegistry::on_demand("snn", 1_000_000, 9));
    }

    /// 注册模型
    fn register(&mut self, model: ModelRegistry) {
        self.models.insert(model.name.clone(), model);
    }

    /// 预热：自动加载所有常驻模型
    fn warmup(&mut self) {
        for (_, model) in self.models.iter_mut() {
            if model.strategy == LoadStrategy::Resident && model.state == ModelState::Unloaded {
                model.state = ModelState::Ready;
                self.current_memory += model.estimated_memory;
                model.load_count += 1;
            }
        }
        self.warmed_up = true;
        log::info!("Scheduler: 预热完成，常驻模型已加载");
    }

    /// 加载模型
    pub fn load_model(&mut self, name: &str) -> Result<(), String> {
        let model = self.models.get_mut(name);
        if model.is_none() {
            return Err(format!("模型 '{}' 未注册", name));
        }

        let model = model.unwrap();

        // 已就绪，无需重复加载
        if model.state == ModelState::Ready {
            return Ok(());
        }

        // 检查内存水位
        if self.check_memory_overflow(model.estimated_memory) {
            // 自动卸载低频模型腾出空间
            self.auto_unload_for(model.estimated_memory);
        }

        // 加载模型
        model.state = ModelState::Loading;
        log::info!("Scheduler: 正在加载模型 '{}' ({}KB)", 
            name, model.estimated_memory / 1024);

        // 模拟加载过程
        model.state = ModelState::Ready;
        self.current_memory += model.estimated_memory;
        model.load_count += 1;

        log::info!("Scheduler: 模型 '{}' 加载完成", name);
        Ok(())
    }

    /// 卸载模型（仅按需加载模型）
    pub fn unload_model(&mut self, name: &str) -> Result<(), String> {
        let model = self.models.get_mut(name);
        if model.is_none() {
            return Err(format!("模型 '{}' 未注册", name));
        }

        let model = model.unwrap();

        // 常驻模型不能卸载
        if model.strategy == LoadStrategy::Resident {
            return Err(format!("模型 '{}' 是常驻模型，不可卸载", name));
        }

        if model.state != ModelState::Ready {
            return Err(format!("模型 '{}' 当前状态为 {:?}，无法卸载", name, model.state));
        }

        model.state = ModelState::Unloaded;
        self.current_memory -= model.estimated_memory;
        log::info!("Scheduler: 模型 '{}' 已卸载，释放 {}KB", 
            name, model.estimated_memory / 1024);
        Ok(())
    }

    /// 获取模型实例（确保已加载）
    pub fn get_model(&mut self, name: &str) -> Result<ModelState, String> {
        self.load_model(name)?;
        let model = self.models.get(name).unwrap();
        
        // 更新推理计数
        self.models.get_mut(name).unwrap().inference_count += 1;
        
        Ok(model.state)
    }

    /// 列出所有模型及状态
    pub fn list_models(&self) -> Vec<(String, LoadStrategy, ModelState, usize)> {
        self.models.iter()
            .map(|(name, reg)| (name.clone(), reg.strategy, reg.state, reg.inference_count))
            .collect()
    }

    /// 内存报告
    pub fn memory_report(&self) -> String {
        let total = self.current_memory;
        let max = self.config.max_memory;
        let usage_pct = total as f64 / max as f64 * 100.0;

        let resident_memory = self.models.iter()
            .filter(|(_, m)| m.strategy == LoadStrategy::Resident && m.state == ModelState::Ready)
            .map(|(_, m)| m.estimated_memory)
            .sum::<usize>();
        
        let on_demand_memory = self.models.iter()
            .filter(|(_, m)| m.strategy == LoadStrategy::OnDemand && m.state == ModelState::Ready)
            .map(|(_, m)| m.estimated_memory)
            .sum::<usize>();

        format!(
            "内存占用: {:.1}MB / {:.1}GB ({:.1}%) | 常驻: {:.1}MB | 按需: {:.1}MB | 预热: {}",
            total as f64 / 1024.0 / 1024.0,
            max as f64 / 1024.0 / 1024.0 / 1024.0,
            usage_pct,
            resident_memory as f64 / 1024.0 / 1024.0,
            on_demand_memory as f64 / 1024.0 / 1024.0,
            self.warmed_up,
        )
    }

    /// 根据任务类型调度合适的模型组合
    pub fn schedule_inference(&mut self, task_type: &str) -> InferenceSchedule {
        let required_models = match task_type {
            "security" => vec!["moe_router", "textcnn", "mamba"],
            "semantic" => vec!["moe_router", "encoder", "mamba"],
            "language" => vec!["moe_router", "transformer", "mamba"],
            "causal" => vec!["moe_router", "gnn", "mamba", "vae"],
            "decision" => vec!["moe_router", "mamba", "gnn", "snn"],
            _ => vec!["moe_router", "textcnn", "mamba"],
        };

        // 加载所需模型
        let mut loaded_models = Vec::new();
        let mut load_errors = Vec::new();
        for name in &required_models {
            match self.load_model(name) {
                Ok(_) => loaded_models.push(name.to_string()),
                Err(e) => load_errors.push(format!("{}: {}", name, e)),
            }
        }

        InferenceSchedule {
            task_type: task_type.to_string(),
            required_models: required_models.iter().map(|s| s.to_string()).collect(),
            loaded_models,
            load_errors,
            estimated_memory: self.calculate_schedule_memory(&required_models),
            timeout_ms: self.config.default_timeout_ms,
        }
    }

    /// 检查内存是否溢出
    fn check_memory_overflow(&self, additional: usize) -> bool {
        (self.current_memory + additional) as f64 / self.config.max_memory as f64 
            > self.config.memory_unload_threshold
    }

    /// 自动卸载低频模型以腾出空间
    fn auto_unload_for(&mut self, required_space: usize) {
        let mut on_demand_models: Vec<(String, usize, usize)> = self.models.iter()
            .filter(|(_, m)| m.strategy == LoadStrategy::OnDemand && m.state == ModelState::Ready)
            .map(|(name, m)| (name.clone(), m.priority, m.estimated_memory))
            .collect();

        // 按优先级排序（优先级低的先卸载）
        on_demand_models.sort_by(|a, b| b.1.cmp(&a.1)); // 降序：大的先卸载

        let mut freed = 0;
        for (name, _, memory) in on_demand_models {
            if freed >= required_space { break; }
            self.unload_model(&name).ok();
            freed += memory;
        }

        log::info!("Scheduler: 自动卸载释放 {:.1}KB", freed as f64 / 1024.0);
    }

    /// 计算调度所需内存
    fn calculate_schedule_memory(&self, model_names: &[&str]) -> usize {
        model_names.iter()
            .filter_map(|name| self.models.get(name))
            .map(|m| m.estimated_memory)
            .sum()
    }
}

/// 推理调度结果
#[derive(Debug)]
pub struct InferenceSchedule {
    /// 任务类型
    pub task_type: String,
    /// 所需模型列表
    pub required_models: Vec<String>,
    /// 已成功加载的模型
    pub loaded_models: Vec<String>,
    /// 加载失败的模型及错误
    pub load_errors: Vec<String>,
    /// 估算内存占用
    pub estimated_memory: usize,
    /// 推理超时(ms)
    pub timeout_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_init() {
        let scheduler = ModelScheduler::new();
        let models = scheduler.list_models();
        assert_eq!(models.len(), 9); // 9个模型
    }

    #[test]
    fn test_scheduler_warmup() {
        let scheduler = ModelScheduler::new();
        // 常驻模型应该已预热
        let resident_ready = scheduler.models.iter()
            .filter(|(_, m)| m.strategy == LoadStrategy::Resident && m.state == ModelState::Ready)
            .count();
        assert_eq!(resident_ready, 2); // textcnn + ga
    }

    #[test]
    fn test_load_model() {
        let mut scheduler = ModelScheduler::new();
        scheduler.load_model("moe_router").unwrap();
        let model = scheduler.models.get("moe_router").unwrap();
        assert_eq!(model.state, ModelState::Ready);
    }

    #[test]
    fn test_unload_model() {
        let mut scheduler = ModelScheduler::new();
        scheduler.load_model("mamba").unwrap();
        scheduler.unload_model("mamba").unwrap();
        let model = scheduler.models.get("mamba").unwrap();
        assert_eq!(model.state, ModelState::Unloaded);
    }

    #[test]
    fn test_cannot_unload_resident() {
        let mut scheduler = ModelScheduler::new();
        let result = scheduler.unload_model("textcnn");
        assert!(result.is_err());
    }

    #[test]
    fn test_schedule_inference() {
        let mut scheduler = ModelScheduler::new();
        let schedule = scheduler.schedule_inference("security");
        assert_eq!(schedule.required_models.len(), 3);
    }

    #[test]
    fn test_memory_report() {
        let scheduler = ModelScheduler::new();
        let report = scheduler.memory_report();
        assert!(report.contains("内存占用"));
    }
}
