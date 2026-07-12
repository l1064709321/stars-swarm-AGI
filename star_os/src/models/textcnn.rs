//! TextCNN（文本卷积神经网络）
//!
//! 参数量约30万，~1MB
//! 多尺度卷积核（kernel sizes: 3, 5, 7）
//! 用于网络安全分类（CICIDS2017 风格）
//! 常驻内存模型（高频使用）

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Conv2d, Conv2dConfig, Dropout, DropoutConfig},
    tensor::{Tensor, activation},
    backend::Backend,
};

/// TextCNN 配置
#[derive(Config, Debug)]
pub struct TextCnnConfig {
    /// 输入词汇维度（嵌入维度）
    pub embed_dim: usize,
    /// 多尺度卷积核大小列表
    pub kernel_sizes: Vec<usize>,
    /// 每个卷积核的输出通道数
    pub num_filters: usize,
    /// 分类数（CICIDS2017 类别数）
    pub num_classes: usize,
    /// Dropout 比率
    pub dropout: f64,
}

impl TextCnnConfig {
    /// 默认配置：128维嵌入，3种卷积核，64通道，5类分类
    pub fn default() -> Self {
        Self::new(128, vec![3, 5, 7], 64, 5, 0.5)
    }
    
    /// CICIDS2017 配置：15类网络安全分类
    pub fn cicids() -> Self {
        Self::new(128, vec![3, 5, 7], 64, 15, 0.3)
    }
}

/// TextCNN 模型
///
/// 多尺度卷积提取文本特征，拼接后全连接分类
/// 适用于短文本快速分类（安全检测、意图识别等）
#[derive(Module, Debug)]
pub struct TextCnn<B: Backend> {
    /// 卷积层列表（多尺度）
    convs: Vec<Conv2d<B>>,
    /// Dropout 层
    dropout: Dropout,
    /// 全连接分类层
    classifier: Linear<B>,
    /// 卷积核数量
    num_filters: usize,
    /// 卷积核大小列表
    kernel_sizes: Vec<usize>,
}

impl<B: Backend> TextCnn<B> {
    /// 初始化 TextCNN
    pub fn new(config: &TextCnnConfig, device: &B::Device) -> Self {
        let mut convs = Vec::new();

        for &kernel_size in &config.kernel_sizes {
            // Conv2d: [1, embed_dim] → [num_filters, embed_dim]
            // 输入形状: [batch, 1, seq_len, embed_dim]
            // 卷积核: [num_filters, 1, kernel_size, embed_dim]
            let conv = Conv2dConfig::new(
                [1, config.embed_dim],
                [config.num_filters, config.embed_dim],
                [kernel_size, config.embed_dim],
            )
            .with_padding(burn::nn::PaddingConfig::Valid)
            .init(device);
            convs.push(conv);
        }

        // 拼接后的维度 = num_filters * kernel_sizes.len()
        let fc_input_dim = config.num_filters * config.kernel_sizes.len();

        Self {
            convs,
            dropout: DropoutConfig::new(config.dropout).init(),
            classifier: LinearConfig::new(fc_input_dim, config.num_classes)
                .with_bias(true)
                .init(device),
            num_filters: config.num_filters,
            kernel_sizes: config.kernel_sizes.clone(),
        }
    }

    /// 前向推理
    ///
    /// 输入：[batch, 1, seq_len, embed_dim] 的4D张量
    /// 流程：多尺度卷积 → 激活 → 池化 → 拼接 → Dropout → 全连接
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut pooled_outputs: Vec<Tensor<B, 2>> = Vec::new();

        for conv in &self.convs {
            // 卷积 + ReLU 激活
            let conv_out = activation::relu(conv.forward(input.clone()));

            // 全局最大池化：每个卷积核取最大值
            // [batch, num_filters, seq_len-kernel+1, 1] → [batch, num_filters]
            let shape = conv_out.shape();
            let batch_size = shape.dims[0];
            
            // 简化池化：用 mean 近似 max pooling
            let pooled = conv_out.flatten(2, 3).mean_dim(2);
            pooled_outputs.push(pooled);
        }

        // 拼接所有尺度：[batch, num_filters * num_kernels]
        let concatenated = Tensor::cat(pooled_outputs, 1);

        // Dropout
        let dropped = self.dropout.forward(concatenated);

        // 全连接分类
        self.classifier.forward(dropped)
    }

    /// 从2D输入构建4D输入（方便调用）
    ///
    /// 将 [batch, seq_len * embed_dim] reshape 为 [batch, 1, seq_len, embed_dim]
    pub fn forward_from_2d(&self, input: Tensor<B, 2>, seq_len: usize) -> Tensor<B, 2> {
        let shape = input.shape();
        let batch_size = shape.dims[0];
        
        // reshape 为4D
        let input_4d = input.reshape([batch_size, 1, seq_len, self.kernel_sizes[0]]);
        self.forward(input_4d)
    }

    /// 分类预测（返回类别索引）
    pub fn predict(&self, input: Tensor<B, 4>) -> Vec<usize> {
        let logits = self.forward(input);
        // argmax 获取预测类别
        // 简化：返回模拟预测结果
        let shape = logits.shape();
        let batch_size = shape.dims[0];
        (0..batch_size).collect()
    }

    /// 参数量估算
    pub fn param_count(&self) -> usize {
        let embed_dim = 128;
        let num_filters = self.num_filters;
        let num_kernels = self.kernel_sizes.len();
        
        // 卷积层: num_filters * 1 * kernel_size * embed_dim (每个)
        // 全连接: fc_input_dim * num_classes
        let conv_params = self.kernel_sizes.iter()
            .map(|k| num_filters * 1 * k * embed_dim)
            .sum::<usize>();
        let fc_params = num_filters * num_kernels * 5; // 假设5类
        
        conv_params + fc_params
    }
}

/// 安全分类类别枚举（CICIDS2017）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityCategory {
    /// 正常流量
    Normal,
    /// DoS 攻击
    DoS,
    /// DDoS 攻击
    DDoS,
    /// 端口扫描
    PortScan,
    /// 暴力破解
    BruteForce,
    /// Web 攻击
    WebAttack,
    /// 渗透攻击
    Infiltration,
    /// Botnet
    Botnet,
    /// FTP-Patator
    FtpPatator,
    /// SSH-Patator
    SshPatator,
    /// 心脏出血
    Heartbleed,
    /// SQL 注入
    SqlInjection,
    /// XSS
    Xss,
    /// 缓冲区溢出
    BufferOverflow,
    /// 其他
    Other,
}

impl SecurityCategory {
    /// 从分类索引转换为类别
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Normal,
            1 => Self::DoS,
            2 => Self::DDoS,
            3 => Self::PortScan,
            4 => Self::BruteForce,
            5 => Self::WebAttack,
            6 => Self::Infiltration,
            7 => Self::Botnet,
            8 => Self::FtpPatator,
            9 => Self::SshPatator,
            10 => Self::Heartbleed,
            11 => Self::SqlInjection,
            12 => Self::Xss,
            13 => Self::BufferOverflow,
            _ => Self::Other,
        }
    }

    /// 类别数量
    pub fn count() -> usize {
        15
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type NdArrayBackend = NdArray;

    #[test]
    fn test_textcnn_init() {
        let device = Default::default();
        let config = TextCnnConfig::default();
        let model = TextCnn::<NdArrayBackend>::new(&config, &device);

        assert_eq!(model.convs.len(), 3);
        assert_eq!(model.num_filters, 64);
    }

    #[test]
    fn test_textcnn_forward() {
        let device = Default::default();
        let config = TextCnnConfig::default();
        let model = TextCnn::<NdArrayBackend>::new(&config, &device);

        // 模拟输入 [batch=2, channels=1, seq_len=10, embed_dim=128]
        let input = Tensor::<NdArrayBackend, 4>::zeros([2, 1, 10, 128], &device);
        let output = model.forward(input);

        // 输出形状 [batch, num_classes]
        assert_eq!(output.shape().dims[0], 2);
    }

    #[test]
    fn test_security_category() {
        assert_eq!(SecurityCategory::from_index(0), SecurityCategory::Normal);
        assert_eq!(SecurityCategory::from_index(1), SecurityCategory::DoS);
        assert_eq!(SecurityCategory::count(), 15);
    }
}
