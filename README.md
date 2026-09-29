# 智能客服 Agent 系统

面向电商场景的客服 Agent：用自然语言查询订单 / 物流 / 退款 / 开票 / 优惠券 / 库存，替代传统关键字机器人。

> **在线演示**：https://415502dd7ee94bec9421df87bffc1743.app.workbuddy.host
> **零依赖**：仅使用 Node.js 内置模块，`node server.js` 即可运行，无需安装任何第三方包、无需数据库/Redis/MQ。

## 功能与架构

```
用户输入
   │
   ├─ DFA 敏感词过滤 ── 命中 → 直接拦截
   ├─ 令牌桶限流     ── 超限 → 429
   │
   ▼
Agent 推理循环（最多 5 轮）
   ① 意图识别      关键词加权打分，9 类意图
   ② RAG 检索      BM25（中文二元切词）+ 关键词字段加权 = 混合检索
   ③ LLM 决策     规划器判断「直接作答 / 检索作答 / 调用工具 / 转人工」
   ④ 工具执行      @Tool 注解式注册，参数缺失自动兜底
   ⑤ 结果回填     工具结果 + 检索资料 → 生成回复
```

## 核心实现要点

| 能力 | 实现方式 |
| --- | --- |
| **Function Calling** | `Tool({name, description, params, run})` 声明式注册，等价于 Java 的 `@Tool` 注解 + 反射展开为 JSON Schema；`/api/tools` 可查看自动生成的工具清单 |
| **RAG 混合检索** | 中文按字符二元组 + 英文按词切分；BM25 相似度 + 关键词字段加权，返回带分数的召回结果 |
| **DFA 敏感词** | Trie 树构建 + 单次扫描命中，O(n) 与词库规模无关 |
| **令牌桶限流** | 惰性补充令牌（无定时器），容量 5、速率 0.5/s，按会话维度 |
| **降级** | 未配置 Redis/MQ 时全部走进程内实现；未配置大模型密钥时走本地规则引擎 |

## 已注册工具（10+ 业务工具中的核心 8 个）

`query_order`、`query_logistics`、`apply_refund`、`issue_invoice`、`query_coupon`、`query_stock`、`change_address`、`handoff_human`

## 运行

```bash
node server.js
# 打开 http://localhost:3000
```

监听 `process.env.PORT`（默认 3000），绑定 `0.0.0.0`。

## 可选：接入真实大模型

默认使用内置本地推理引擎（无需密钥，开箱即用）。若要走真实大模型，设置环境变量即可，服务会自动切换到 OpenAI 兼容协议：

```bash
LLM_API_KEY=sk-xxx LLM_BASE_URL=https://api.openai.com/v1 LLM_MODEL=gpt-4o-mini node server.js
```

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | `{sessionId, message}` → `{reply, trace}`，trace 含意图、检索、工具调用、耗时 |
| GET | `/api/tools` | 自动生成的工具 JSON Schema 清单 |
| GET | `/api/kb` | 知识库条目 |
| GET | `/api/stats` | 限流 / 敏感词 / 会话统计 |
| GET | `/api/health` | 健康检查 |

## 说明

- 数据（订单、物流、商品、知识库）均为**演示用模拟数据**，不接真实业务库。
- 本项目为求职作品集演示，重点展示 Agent 推理链路与工程化组件，而非线上真实流量。
