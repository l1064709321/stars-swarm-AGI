# STARS-Gateway 路线图

## v0.1.0（已完成）
- 单节点 HTTP 慢速陷阱引擎
- 基于 Tokio 的异步并发处理
- 验证不对称消耗可行性

## v0.2.0（开发中）
- 强制读取客户端请求（防空连接攻击）
- 协议指纹随机化（防测绘）
- 时间扰动（Jitter）
- Prometheus 指标暴露

## v0.3.0（计划中）
- 多协议伪装（SSH/MySQL/SMTP）
- 行为分析与智能分流
- 结构化 JSON 日志
- HTTP API 联动接口

## v1.0.0（远期）
- eBPF 内核分流
- Gossip 威胁同步
- stars-AGI系统接入。
