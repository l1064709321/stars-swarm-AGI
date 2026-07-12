//! Ethics Symbolic（伦理符号系统）
//!
//! L6 伦理判断层
//! 伦理判断不同流派
//! 三层校验拦截低伦理签名消息
//! 伦理网关校验后才能执行

use std::collections::HashMap;

/// 伦理流派
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EthicsSchool {
    /// 功利主义（后果导向）
    Utilitarianism,
    /// 义务论（规则导向）
    Deontology,
    /// 德性伦理（品格导向）
    VirtueEthics,
    /// 儒家伦理（关系导向）
    Confucian,
    /// 关怀伦理（关怀导向）
    CareEthics,
}

/// 伦理评分维度
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EthicsDimension {
    /// 安全性（是否会造成伤害）
    Safety,
    /// 公平性（是否公平对待各方）
    Fairness,
    /// 透明度（是否可解释）
    Transparency,
    /// 自主权（是否尊重自主选择）
    Autonomy,
    /// 隐私（是否保护隐私）
    Privacy,
    /// 可信赖（是否可信）
    Trustworthiness,
}

impl EthicsDimension {
    /// 所有维度列表
    pub fn all() -> Vec<EthicsDimension> {
        vec![
            Self::Safety,
            Self::Fairness,
            Self::Transparency,
            Self::Autonomy,
            Self::Privacy,
            Self::Trustworthiness,
        ]
    }

    /// 维度名称
    pub fn name(&self) -> &str {
        match self {
            Self::Safety => "安全性",
            Self::Fairness => "公平性",
            Self::Transparency => "透明度",
            Self::Autonomy => "自主权",
            Self::Privacy => "隐私",
            Self::Trustworthiness => "可信赖",
        }
    }
}

/// 伦理签名（Ethics Signature）
///
/// 每个决策都有伦理签名，包含各维度的评分
#[derive(Debug, Clone)]
pub struct EthicsSignature {
    /// 各维度评分 [0.0, 1.0]
    pub scores: HashMap<EthicsDimension, f64>,
    /// 总体伦理评分
    pub overall_score: f64,
    /// 评估所用的伦理流派
    pub school: EthicsSchool,
    /// 是否通过伦理网关
    pub passed: bool,
    /// 拦截原因（如果未通过）
    pub rejection_reason: Option<String>,
}

impl EthicsSignature {
    /// 创建空签名
    pub fn empty() -> Self {
        Self {
            scores: HashMap::new(),
            overall_score: 0.0,
            school: EthicsSchool::Utilitarianism,
            passed: false,
            rejection_reason: None,
        }
    }

    /// 创建默认签名（所有维度0.5分）
    pub fn neutral() -> Self {
        let scores: HashMap<EthicsDimension, f64> = EthicsDimension::all()
            .iter().map(|d| (*d, 0.5)).collect();
        Self {
            scores,
            overall_score: 0.5,
            school: EthicsSchool::Utilitarianism,
            passed: true,
            rejection_reason: None,
        }
    }
}

/// 伦理网关配置
#[derive(Debug, Clone)]
pub struct EthicsGatewayConfig {
    /// 通过阈值（总体评分需要超过此值才能通过）
    pub pass_threshold: f64,
    /// 各维度的最低阈值
    pub dimension_thresholds: HashMap<EthicsDimension, f64>,
    /// 三层校验是否都启用
    pub triple_check_enabled: bool,
    /// 使用的伦理流派权重
    pub school_weights: HashMap<EthicsSchool, f64>,
}

impl Default for EthicsGatewayConfig {
    fn default() -> Self {
        Self {
            pass_threshold: 0.6,
            dimension_thresholds: EthicsDimension::all()
                .iter().map(|d| (*d, 0.3)).collect(),
            triple_check_enabled: true,
            school_weights: HashMap::from([
                (EthicsSchool::Utilitarianism, 0.3),
                (EthicsSchool::Deontology, 0.25),
                (EthicsSchool::VirtueEthics, 0.15),
                (EthicsSchool::Confucian, 0.2),
                (EthicsSchool::CareEthics, 0.1),
            ]),
        }
    }
}

/// 伦理网关（Ethics Gateway）
///
/// 三层校验机制：
/// 1. 第一层：快速伦理签名检查（低伦理签名直接拦截）
/// 2. 第二层：多流派交叉验证（不同流派评估一致性）
/// 3. 第三层：维度深度检查（各维度是否达到最低阈值）
pub struct EthicsGateway {
    /// 配置参数
    config: EthicsGatewayConfig,
}

impl EthicsGateway {
    /// 创建伦理网关
    pub fn new(config: EthicsGatewayConfig) -> Self {
        Self { config }
    }

    /// 默认实例
    pub fn default() -> Self {
        Self::new(EthicsGatewayConfig::default())
    }

    /// 三层校验：检查决策是否符合伦理要求
    ///
    /// 返回 EthicsSignature，包含是否通过和拦截原因
    pub fn verify(&self, decision: &DecisionContext) -> EthicsSignature {
        if !self.config.triple_check_enabled {
            return self.single_check(decision);
        }

        // 第一层：快速伦理签名检查
        let layer1_result = self.layer1_quick_check(decision);
        if !layer1_result.passed {
            log::warn!("伦理网关 第一层拦截: {}", 
                layer1_result.rejection_reason.unwrap_or_default());
            return layer1_result;
        }

        // 第二层：多流派交叉验证
        let layer2_result = self.layer2_cross_validation(decision, &layer1_result);
        if !layer2_result.passed {
            log::warn!("伦理网关 第二层拦截: {}", 
                layer2_result.rejection_reason.unwrap_or_default());
            return layer2_result;
        }

        // 第三层：维度深度检查
        let layer3_result = self.layer3_dimension_check(decision, &layer2_result);
        if !layer3_result.passed {
            log::warn!("伦理网关 第三层拦截: {}", 
                layer3_result.rejection_reason.unwrap_or_default());
            return layer3_result;
        }

        log::info!("伦理网关 三层校验通过 ✓ 评分={:.2}", layer3_result.overall_score);
        layer3_result
    }

    /// 第一层：快速伦理签名检查
    ///
    /// 检查是否有明显的低伦理签名（安全性和隐私维度极低）
    fn layer1_quick_check(&self, decision: &DecisionContext) -> EthicsSignature {
        let mut sig = EthicsSignature::empty();
        sig.school = EthicsSchool::Utilitarianism;

        // 安全性评分：是否可能造成伤害
        sig.scores.insert(EthicsDimension::Safety, decision.safety_score);
        
        // 隐私评分：是否涉及隐私风险
        sig.scores.insert(EthicsDimension::Privacy, decision.privacy_score);

        // 快速判断：安全性或隐私低于0.1 → 直接拦截
        if decision.safety_score < 0.1 {
            sig.passed = false;
            sig.rejection_reason = Some("安全性评分过低（可能造成伤害）".to_string());
            sig.overall_score = 0.0;
            return sig;
        }

        if decision.privacy_score < 0.1 {
            sig.passed = false;
            sig.rejection_reason = Some("隐私评分过低（隐私风险极高）".to_string());
            sig.overall_score = 0.0;
            return sig;
        }

        sig.overall_score = (decision.safety_score + decision.privacy_score) / 2.0;
        sig.passed = true;
        sig
    }

    /// 第二层：多流派交叉验证
    ///
    /// 不同伦理流派分别评估，检查一致性
    fn layer2_cross_validation(&self, decision: &DecisionContext, prev: &EthicsSignature) -> EthicsSignature {
        let mut sig = prev.clone();
        sig.school = EthicsSchool::Deontology; // 第二层使用义务论

        // 功利主义评分：后果的好坏
        let utilitarian_score = self.utilitarian_eval(decision);
        sig.scores.insert(EthicsDimension::Fairness, utilitarian_score);

        // 义务论评分：是否符合规则
        let deontology_score = self.deontology_eval(decision);
        sig.scores.insert(EthicsDimension::Autonomy, deontology_score);

        // 德性伦理评分：品格层面
        let virtue_score = self.virtue_ethics_eval(decision);
        sig.scores.insert(EthicsDimension::Trustworthiness, virtue_score);

        // 计算综合评分
        let weighted_scores = self.config.school_weights.iter()
            .map(|(school, weight)| {
                let score = match school {
                    EthicsSchool::Utilitarianism => utilitarian_score,
                    EthicsSchool::Deontology => deontology_score,
                    EthicsSchool::VirtueEthics => virtue_score,
                    EthicsSchool::Confucian => (utilitarian_score + deontology_score) / 2.0,
                    EthicsSchool::CareEthics => (virtue_score + deontology_score) / 2.0,
                };
                score * weight
            })
            .sum::<f64>();

        sig.overall_score = weighted_scores / self.config.school_weights.values().sum::<f64>();

        // 各流派评分差异过大 → 不一致性 → 拦截
        let scores = vec![utilitarian_score, deontology_score, virtue_score];
        let variance = self.compute_variance(&scores);
        if variance > 0.3 {
            sig.passed = false;
            sig.rejection_reason = Some(format!(
                "伦理评估不一致（方差={:.2}），各流派评分差异过大", variance));
            return sig;
        }

        sig.passed = true;
        sig
    }

    /// 第三层：维度深度检查
    ///
    /// 检查每个伦理维度是否都达到最低阈值
    fn layer3_dimension_check(&self, decision: &DecisionContext, prev: &EthicsSignature) -> EthicsSignature {
        let mut sig = prev.clone();

        // 补充缺失维度评分
        sig.scores.insert(EthicsDimension::Transparency, decision.transparency_score);

        // 检查所有维度是否达到最低阈值
        for (dimension, threshold) in &self.config.dimension_thresholds {
            let score = sig.scores.get(dimension).unwrap_or(&0.0);
            if *score < *threshold {
                sig.passed = false;
                sig.rejection_reason = Some(format!(
                    "{}维度评分 {:.2} 低于阈值 {:.2}",
                    dimension.name(), score, threshold));
                return sig;
            }
        }

        // 总体评分是否达标
        if sig.overall_score < self.config.pass_threshold {
            sig.passed = false;
            sig.rejection_reason = Some(format!(
                "总体伦理评分 {:.2} 低于阈值 {:.2}",
                sig.overall_score, self.config.pass_threshold));
            return sig;
        }

        sig.passed = true;
        sig
    }

    /// 单层检查（简化版）
    fn single_check(&self, decision: &DecisionContext) -> EthicsSignature {
        let mut sig = EthicsSignature::neutral();
        sig.scores.insert(EthicsDimension::Safety, decision.safety_score);
        sig.scores.insert(EthicsDimension::Privacy, decision.privacy_score);
        sig.scores.insert(EthicsDimension::Transparency, decision.transparency_score);
        
        sig.overall_score = sig.scores.values().sum::<f64>() / sig.scores.len() as f64;
        sig.passed = sig.overall_score >= self.config.pass_threshold;
        
        if !sig.passed {
            sig.rejection_reason = Some(format!("总体评分 {:.2} 低于阈值", sig.overall_score));
        }
        sig
    }

    /// 功利主义评估：后果的好坏
    fn utilitarian_eval(&self, decision: &DecisionContext) -> f64 {
        // 简化：基于决策的正向影响比例
        (decision.safety_score + decision.privacy_score + decision.transparency_score) / 3.0
    }

    /// 义务论评估：是否符合规则和义务
    fn deontology_eval(&self, decision: &DecisionContext) -> f64 {
        // 简化：基于合规性
        if decision.is_compliant { 0.8 } else { 0.2 }
    }

    /// 德性伦理评估：品格层面
    fn virtue_ethics_eval(&self, decision: &DecisionContext) -> f64 {
        // 简化：基于决策的道德品质
        (decision.transparency_score + decision.safety_score) / 2.0
    }

    /// 计算方差
    fn compute_variance(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() { return 0.0; }
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64
    }
}

/// 决策上下文（提交给伦理网关审核）
#[derive(Debug, Clone)]
pub struct DecisionContext {
    /// 安全性评分
    pub safety_score: f64,
    /// 隐私评分
    pub privacy_score: f64,
    /// 透明度评分
    pub transparency_score: f64,
    /// 是否合规
    pub is_compliant: bool,
    /// 决策描述
    pub description: String,
    /// 决策类别
    pub category: String,
    /// 目标（路由目标）
    pub route_target: String,
}

impl DecisionContext {
    /// 创建安全的决策上下文
    pub fn safe(description: String) -> Self {
        Self {
            safety_score: 0.9,
            privacy_score: 0.8,
            transparency_score: 0.7,
            is_compliant: true,
            description,
            category: "安全查询".to_string(),
            route_target: "textcnn".to_string(),
        }
    }

    /// 创建危险的决策上下文
    pub fn dangerous(description: String) -> Self {
        Self {
            safety_score: 0.05,
            privacy_score: 0.1,
            transparency_score: 0.2,
            is_compliant: false,
            description,
            category: "高危操作".to_string(),
            route_target: "unknown".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ethics_gateway_default() {
        let gateway = EthicsGateway::default();
        assert_eq!(gateway.config.pass_threshold, 0.6);
        assert!(gateway.config.triple_check_enabled);
    }

    #[test]
    fn test_safe_decision() {
        let gateway = EthicsGateway::default();
        let decision = DecisionContext::safe("安全查询".to_string());
        let result = gateway.verify(&decision);
        assert!(result.passed);
        assert!(result.overall_score > 0.6);
    }

    #[test]
    fn test_dangerous_decision() {
        let gateway = EthicsGateway::default();
        let decision = DecisionContext::dangerous("高危操作".to_string());
        let result = gateway.verify(&decision);
        assert!(!result.passed);
        assert!(result.rejection_reason.is_some());
    }

    #[test]
    fn test_ethics_dimensions() {
        let dims = EthicsDimension::all();
        assert_eq!(dims.len(), 6);
        assert_eq!(EthicsDimension::Safety.name(), "安全性");
    }
}
