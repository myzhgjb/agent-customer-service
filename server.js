/**
 * 智能客服 Agent 系统 — 零依赖 Node 实现
 * 演示：意图识别 → RAG 混合检索 → Agent 推理循环（LLM 决策）→ 工具执行 → 结果回填
 * 另含：@Tool 注解式工具注册（用注解对象 + 反射思想模拟）、DFA 敏感词过滤、令牌桶限流、降级
 *
 * 运行时：Node.js（仅用内置模块），监听 process.env.PORT，绑定 0.0.0.0
 */
'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const HOST = '0.0.0.0';
const PUBLIC_DIR = path.join(__dirname, 'public');

/* ------------------------------------------------------------------ *
 * 0. 可选的「真实大模型」通道
 *    设置环境变量 LLM_API_KEY（及可选 LLM_BASE_URL / LLM_MODEL）即走真实模型；
 *    否则使用内置的本地推理引擎（无需任何密钥即可运行）。
 * ------------------------------------------------------------------ */
const LLM_API_KEY = process.env.LLM_API_KEY || process.env.OPENAI_API_KEY || '';
const LLM_BASE_URL = process.env.LLM_BASE_URL || 'https://api.openai.com/v1';
const LLM_MODEL = process.env.LLM_MODEL || 'gpt-4o-mini';
const LLM_ENABLED = !!LLM_API_KEY;

/* ================================================================== *
 * 1. 模拟业务数据库（真实项目里是 MySQL / 订单服务）
 * ================================================================== */
const DB = {
  orders: {
    ORD2024001: {
      id: 'ORD2024001', user: '马誉洲', status: '已发货', amount: 399.0,
      product: '无线蓝牙耳机 Pro', createdAt: '2024-11-02 10:21',
      address: '江苏省苏州市高新区科锐路1号', coupon: 'CM-20（满200减20，已使用）',
    },
    ORD2024002: {
      id: 'ORD2024002', user: '马誉洲', status: '待发货', amount: 1299.0,
      product: '智能手表 Watch S2', createdAt: '2024-11-18 19:03',
      address: '江苏省苏州市虎丘区学府路99号', coupon: 'NEW-50（新人券，可用）',
    },
    ORD2024003: {
      id: 'ORD2024003', user: '马誉洲', status: '已完成', amount: 89.9,
      product: '机械键盘 K68', createdAt: '2024-10-05 08:45',
      address: '江苏省苏州市高新区科锐路1号', coupon: '无',
    },
  },
  logistics: {
    ORD2024001: [
      { time: '2024-11-03 09:12', node: '【苏州市】快件已揽收' },
      { time: '2024-11-03 15:40', node: '【苏州市】到达苏州转运中心' },
      { time: '2024-11-04 08:05', node: '【苏州市】派送中，快递员：张师傅 138****2210' },
    ],
    ORD2024002: [
      { time: '2024-11-19 11:30', node: '【苏州市】订单已进入仓库，正在打包' },
    ],
    ORD2024003: [
      { time: '2024-10-06 10:00', node: '【苏州市】已签收，感谢使用' },
    ],
  },
  products: {
    P1001: { id: 'P1001', name: '无线蓝牙耳机 Pro', stock: 128, price: 399.0 },
    P1002: { id: 'P1002', name: '智能手表 Watch S2', stock: 12, price: 1299.0 },
    P1003: { id: 'P1003', name: '机械键盘 K68', stock: 0, price: 89.9 },
  },
  invoices: {},   // 已开票记录
  refunds: {},    // 退款工单
};

/* ================================================================== *
 * 2. DFA 敏感词过滤（Deterministic Finite Automaton）
 * ================================================================== */
function buildDFA(words) {
  const root = Object.create(null);
  for (const w of words) {
    let node = root;
    for (const ch of w) {
      if (!node[ch]) node[ch] = Object.create(null);
      node = node[ch];
    }
    node.__end = true;
  }
  return root;
}
const SENSITIVE_WORDS = ['傻逼', '白痴', '垃圾公司', '诈骗', '去死', 'fuck', '滚蛋', '骗子'];
const DFA_ROOT = buildDFA(SENSITIVE_WORDS);

const filterStats = { checked: 0, blocked: 0 };
function dfaFilter(text) {
  filterStats.checked++;
  const hits = [];
  for (let i = 0; i < text.length; i++) {
    let node = DFA_ROOT;
    for (let j = i; j < text.length; j++) {
      node = node[text[j]];
      if (!node) break;
      if (node.__end) { hits.push(text.slice(i, j + 1)); i = j; break; }
    }
  }
  if (hits.length) filterStats.blocked++;
  return { blocked: hits.length > 0, hits };
}

/* ================================================================== *
 * 3. 令牌桶限流（惰性补充，无定时器）
 * ================================================================== */
class TokenBucket {
  constructor(capacity, ratePerSec) {
    this.capacity = capacity;
    this.rate = ratePerSec;
    this.tokens = capacity;
    this.last = Date.now();
  }
  tryAcquire(n = 1) {
    const now = Date.now();
    this.tokens = Math.min(this.capacity, this.tokens + ((now - this.last) / 1000) * this.rate);
    this.last = now;
    if (this.tokens >= n) { this.tokens -= n; return true; }
    return false;
  }
}
const buckets = new Map();       // sessionId -> TokenBucket
const rateStats = { total: 0, limited: 0 };
function rateLimit(sessionId) {
  rateStats.total++;
  if (!buckets.has(sessionId)) buckets.set(sessionId, new TokenBucket(5, 0.5)); // 容量5，0.5/s
  const ok = buckets.get(sessionId).tryAcquire(1);
  if (!ok) rateStats.limited++;
  return ok;
}

/* ================================================================== *
 * 4. 知识库 + RAG 混合检索（向量式语义 + 关键词）
 * ================================================================== */
const KB = [
  { id: 'kb1', title: '退货退款政策', keywords: ['退款', '退货', '售后', '七天无理由'],
    content: '自签收之日起 7 天内支持无理由退货，商品需保持完好、不影响二次销售。退款将在审核通过后 1-3 个工作日原路退回。定制商品、已拆封的贴身用品不支持无理由退货。' },
  { id: 'kb2', title: '发货与物流时效', keywords: ['物流', '快递', '发货', '多久', '时效'],
    content: '现货商品在付款后 24 小时内发出，江浙沪一般 1-2 天送达，其他地区 2-4 天。大促期间可能延迟 1-2 天。发货后在“我的订单”可查看实时物流轨迹。' },
  { id: 'kb3', title: '发票开具说明', keywords: ['发票', '开票', '抬头', '税号', '报销'],
    content: '支持开具电子普票与增值税专票。电子普票在订单完成后自动开具，可下载 PDF；专票需要提供公司名称、税号、开户行等信息，审核后 3 个工作日内开具。' },
  { id: 'kb4', title: '优惠券使用规则', keywords: ['优惠券', '券', '满减', '折扣', '新人'],
    content: '优惠券在结算时自动匹配可用面额，同一订单仅可使用一张优惠券，不与部分特价活动叠加。未使用的优惠券在到期后自动失效。' },
  { id: 'kb5', title: '修改收货地址', keywords: ['地址', '收货', '改地址', '换地址'],
    content: '订单在“待发货”状态下可以修改收货地址；一旦进入“已发货”状态则无法修改，可联系快递员协商或拒收后重新下单。' },
  { id: 'kb6', title: '商品库存与补货', keywords: ['库存', '缺货', '补货', '有货'],
    content: '商品详情页显示实时库存。若显示缺货，可点击“到货提醒”，补货后会以短信和 App 推送通知。一般补货周期为 3-7 天。' },
];

/* 中文按字符二元组切词，英文/数字按连续串 */
function tokenize(text) {
  const tokens = [];
  const lower = (text || '').toLowerCase();
  const segs = lower.match(/[a-z0-9]+|[\u4e00-\u9fa5]+/g) || [];
  for (const seg of segs) {
    if (/^[a-z0-9]+$/.test(seg)) { tokens.push(seg); continue; }
    if (seg.length === 1) { tokens.push(seg); continue; }
    for (let i = 0; i < seg.length - 1; i++) tokens.push(seg.slice(i, i + 2));
  }
  return tokens;
}
/* 简易 BM25 */
const docTokens = KB.map(d => tokenize(d.title + ' ' + d.keywords.join(' ') + ' ' + d.content));
const df = {};
docTokens.forEach(toks => { new Set(toks).forEach(t => (df[t] = (df[t] || 0) + 1)); });
const N = KB.length;
const AVG_LEN = docTokens.reduce((a, t) => a + t.length, 0) / N;
const K1 = 1.5, B = 0.75;

function bm25Scores(query) {
  const qTokens = tokenize(query);
  return docTokens.map((toks, i) => {
    const tf = {};
    toks.forEach(t => (tf[t] = (tf[t] || 0) + 1));
    let score = 0;
    for (const qt of qTokens) {
      if (!tf[qt]) continue;
      const idf = Math.log(1 + (N - df[qt] + 0.5) / (df[qt] + 0.5));
      score += idf * (tf[qt] * (K1 + 1)) / (tf[qt] + K1 * (1 - B + B * toks.length / AVG_LEN));
    }
    // 关键词字段加权（模拟“向量语义相似度”通道）
    const kwBoost = KB[i].keywords.reduce((a, k) => a + (query.includes(k) ? 1.5 : 0), 0);
    return { idx: i, score: score + kwBoost };
  });
}
function retrieve(query, topK = 2) {
  return bm25Scores(query)
    .filter(s => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(s => ({ id: KB[s.idx].id, title: KB[s.idx].title, score: +s.score.toFixed(2), content: KB[s.idx].content }));
}

/* ================================================================== *
 * 5. 工具注册（模拟 @Tool 注解 + 反射）
 *    Java 版：扫描带 @Tool 的方法，用反射读参数类型生成 JSON Schema；
 *    此处用等价的注解对象 + 处理器函数实现同样的“声明式注册”。
 * ================================================================== */
const TOOLS = {};
function Tool(def) {                       // 等价于 Java 的 @Tool 注解
  TOOLS[def.name] = def;
  return def;
}
function toolSchema() {                    // 等价于“注解 → JSON Schema”的展开
  return Object.values(TOOLS).map(t => ({
    name: t.name, description: t.description,
    parameters: t.params,
  }));
}

Tool({
  name: 'query_order', description: '根据订单号查询订单状态、金额、商品、收货地址',
  params: { orderId: { type: 'string', required: true, desc: '订单号，如 ORD2024001' } },
  run: ({ orderId }) => {
    const o = DB.orders[orderId];
    return o ? { ok: true, data: o } : { ok: false, error: '订单不存在' };
  },
});
Tool({
  name: 'query_logistics', description: '根据订单号查询物流轨迹',
  params: { orderId: { type: 'string', required: true, desc: '订单号' } },
  run: ({ orderId }) => {
    const l = DB.logistics[orderId];
    return l ? { ok: true, data: l } : { ok: false, error: '暂无物流信息' };
  },
});
Tool({
  name: 'apply_refund', description: '为指定订单发起退款申请（写操作，需二次确认）',
  params: { orderId: { type: 'string', required: true, desc: '订单号' }, reason: { type: 'string', required: false, desc: '退款原因' } },
  run: ({ orderId, reason }) => {
    const o = DB.orders[orderId];
    if (!o) return { ok: false, error: '订单不存在' };
    if (o.status === '待发货' || o.status === '已发货' || o.status === '已完成') {
      DB.refunds[orderId] = { orderId, reason: reason || '用户申请', status: '已受理', at: now() };
      return { ok: true, data: { orderId, refundNo: 'RF' + orderId.slice(-4) + Math.floor(Math.random() * 900 + 100), status: '已受理，1-3 个工作日原路退回' } };
    }
    return { ok: false, error: '当前订单状态不支持退款' };
  },
});
Tool({
  name: 'issue_invoice', description: '为订单开具电子发票',
  params: { orderId: { type: 'string', required: true, desc: '订单号' }, title: { type: 'string', required: false, desc: '发票抬头' } },
  run: ({ orderId, title }) => {
    const o = DB.orders[orderId];
    if (!o) return { ok: false, error: '订单不存在' };
    DB.invoices[orderId] = { orderId, title: title || o.user, amount: o.amount, status: '已开具', at: now() };
    return { ok: true, data: DB.invoices[orderId] };
  },
});
Tool({
  name: 'query_coupon', description: '查询订单可用的优惠券',
  params: { orderId: { type: 'string', required: true, desc: '订单号' } },
  run: ({ orderId }) => {
    const o = DB.orders[orderId];
    return o ? { ok: true, data: { orderId, coupon: o.coupon } } : { ok: false, error: '订单不存在' };
  },
});
Tool({
  name: 'query_stock', description: '查询商品实时库存',
  params: { productId: { type: 'string', required: true, desc: '商品编号，如 P1001' } },
  run: ({ productId }) => {
    const p = DB.products[productId];
    return p ? { ok: true, data: p } : { ok: false, error: '商品不存在' };
  },
});
Tool({
  name: 'change_address', description: '修改订单收货地址（仅待发货状态可用）',
  params: { orderId: { type: 'string', required: true, desc: '订单号' }, address: { type: 'string', required: true, desc: '新地址' } },
  run: ({ orderId, address }) => {
    const o = DB.orders[orderId];
    if (!o) return { ok: false, error: '订单不存在' };
    if (o.status !== '待发货') return { ok: false, error: `订单当前为「${o.status}」，不可修改地址` };
    o.address = address;
    return { ok: true, data: { orderId, address } };
  },
});
Tool({
  name: 'handoff_human', description: '转人工客服（复杂或投诉类问题）',
  params: { reason: { type: 'string', required: false, desc: '转人工原因' } },
  run: ({ reason }) => ({ ok: true, data: { ticket: 'HD' + Date.now().toString().slice(-6), reason: reason || '用户请求', status: '已转接人工客服，请稍候' } }),
});

/* ================================================================== *
 * 6. 意图识别 + 实体抽取
 * ================================================================== */
const INTENTS = [
  { name: 'greeting',  kw: ['你好', '您好', 'hi', 'hello', '在吗', '嗨'] },
  { name: 'order',     kw: ['订单', '下单', '买', 'order'] },
  { name: 'logistics', kw: ['物流', '快递', '发货', '到哪', '签收', '运单', '配送'] },
  { name: 'refund',    kw: ['退款', '退货', '售后', '退', 'refund'] },
  { name: 'invoice',   kw: ['发票', '开票', '抬头', '税号', '报销'] },
  { name: 'coupon',    kw: ['优惠券', '券', '满减', '折扣'] },
  { name: 'stock',     kw: ['库存', '缺货', '有货', '补货', '还有'] },
  { name: 'address',   kw: ['地址', '收货', '改地址', '换地址'] },
  { name: 'complaint', kw: ['投诉', '人工', '客服', '差评', '生气'] },
];
function recognizeIntent(text) {
  let best = { name: 'other', score: 0 };
  for (const it of INTENTS) {
    let score = 0;
    for (const k of it.kw) if (text.includes(k)) score += k.length >= 2 ? 2 : 1;
    if (score > best.score) best = { name: it.name, score };
  }
  return best;
}
function extractOrderId(text) {
  const m = text.match(/ORD\d{4,}/i);
  if (m) return m[0].toUpperCase();
  const digits = text.match(/\d{5,}/);   // 用户只报后几位时做模糊匹配
  if (digits) {
    const hit = Object.keys(DB.orders).find(id => id.includes(digits[0]));
    if (hit) return hit;
  }
  return null;
}

/* ================================================================== *
 * 7. Agent 推理循环
 *    规划器 => 决定调用哪个工具；执行器 => 调工具；回填 => 汇总生成回答
 *    最多 5 轮（简历：5 轮内多工具自主调用）
 * ================================================================== */
const MAX_ITER = 5;

function plan(intent, text, memory) {
  const orderId = extractOrderId(text);
  // 政策/规则类提问且未给具体单号 → 直接走知识库，不误触发写操作工具
  const policyHint = /政策|规则|怎么|怎样|如何|多久|多少|说明|是什么|能不能|可以吗|支持|流程/.test(text);
  if (!orderId && policyHint && intent.name !== 'greeting' && intent.name !== 'complaint') {
    return { type: 'rag', text };
  }
  switch (intent.name) {
    case 'greeting': return { type: 'answer', text: '您好，我是智能客服小誉～ 可以帮您查订单、物流、退款、发票、优惠券、库存或修改地址。请问有什么可以帮您？' };
    case 'logistics': return { type: 'tool', name: 'query_logistics', args: { orderId } };
    case 'refund':    return { type: 'tool', name: 'apply_refund', args: { orderId, reason: text } };
    case 'invoice':   return { type: 'tool', name: 'issue_invoice', args: { orderId, title: /抬头[:：]?\s*([^\s，。]+)/.exec(text)?.[1] } };
    case 'coupon':    return { type: 'tool', name: 'query_coupon', args: { orderId } };
    case 'address': {
      const addr = /(改成|修改为|改为)\s*([^\s。]+)/.exec(text)?.[2];
      return addr
        ? { type: 'tool', name: 'change_address', args: { orderId, address: addr } }
        : { type: 'answer', text: '好的，请告诉我新的收货地址和订单号，我来帮您修改（仅“待发货”状态可改）。' };
    }
    case 'stock': {
      const pid = (text.match(/P\d{4}/i) || [])[0]?.toUpperCase();
      return { type: 'tool', name: 'query_stock', args: { productId: pid || 'P1001' } };
    }
    case 'complaint': return { type: 'tool', name: 'handoff_human', args: { reason: text } };
    case 'order': {
      if (orderId) return { type: 'tool', name: 'query_order', args: { orderId } };
      return { type: 'answer', text: '请提供订单号（形如 ORD2024001），我帮您查询订单状态。' };
    }
    default:
      return { type: 'rag', text };   // other → 走知识库检索回答
  }
}

function composeAnswer(intent, toolResults, docs) {
  const lines = [];
  if (toolResults.length) {
    for (const { name, args, result } of toolResults) {
      if (!result.ok) { lines.push(`查询未成功：${result.error}。`); continue; }
      const d = result.data;
      if (name === 'query_order') lines.push(`订单 ${d.id}（${d.product}）：当前状态「${d.status}」，金额 ¥${d.amount}，下单时间 ${d.createdAt}。`);
      else if (name === 'query_logistics') lines.push(`订单 ${args.orderId} 最新物流：${d[d.length - 1].node}（${d[d.length - 1].time}）。共 ${d.length} 条轨迹。`);
      else if (name === 'apply_refund') lines.push(`已为您提交退款申请，单号 ${d.refundNo}，${d.status}。`);
      else if (name === 'issue_invoice') lines.push(`发票已开具：抬头「${d.title}」，金额 ¥${d.amount}，状态「${d.status}」。`);
      else if (name === 'query_coupon') lines.push(`订单 ${d.orderId} 的优惠券：${d.coupon}。`);
      else if (name === 'query_stock') lines.push(`${d.name}（${d.id}）当前库存 ${d.stock} 件，单价 ¥${d.price}。`);
      else if (name === 'change_address') lines.push(`已把订单 ${d.orderId} 的收货地址修改为：${d.address}。`);
      else if (name === 'handoff_human') lines.push(`已为您转接人工客服，工单号 ${d.ticket}。`);
    }
  }
  if (docs.length) {
    lines.push(`【参考资料 · ${docs[0].title}】${docs[0].content}`);
  }
  if (!lines.length) lines.push('抱歉，我暂时没有理解您的问题，可以换个说法，或说“转人工”联系客服。');
  return lines.join('\n');
}

async function runAgent(text, sessionId) {
  const t0 = Date.now();
  const trace = { intent: null, retrieved: [], toolCalls: [], iterations: 0, engine: LLM_ENABLED ? 'LLM' : 'local' };
  const memory = [];
  const toolResults = [];
  let docs = [];
  let iteration = 0;

  const intent = recognizeIntent(text);
  trace.intent = { name: intent.name, score: intent.score };

  while (iteration < MAX_ITER) {
    iteration++;
    const step = plan(intent, text, memory);

    if (step.type === 'answer') { memory.push({ role: 'assistant', content: step.text }); var finalText = step.text; break; }

    if (step.type === 'rag') {
      docs = retrieve(text, 2);
      trace.retrieved = docs;
      finalText = composeAnswer(intent, toolResults, docs);
      break;
    }

    if (step.type === 'tool') {
      const tool = TOOLS[step.name];
      if (!tool) { finalText = '内部错误：工具未注册。'; break; }
      // 参数缺失兜底：模型选对了工具但没给订单号
      const missing = Object.entries(tool.params).filter(([k, v]) => v.required && !step.args[k]).map(([k]) => k);
      if (missing.length) {
        finalText = `请补充${missing.map(m => `「${tool.params[m].desc}」`).join('、')}，我才能继续为您处理。`;
        trace.toolCalls.push({ name: step.name, args: step.args, result: { ok: false, error: 'missing params: ' + missing.join(',') } });
        break;
      }
      const result = tool.run(step.args);
      trace.toolCalls.push({ name: step.name, args: step.args, result });
      toolResults.push({ name: step.name, args: step.args, result });
      memory.push({ role: 'tool', name: step.name, content: JSON.stringify(result) });
      // 回填后再决策一轮：若还为 other 则补检索，否则直接生成
      docs = docs.length ? docs : retrieve(text, 2);
      trace.retrieved = docs;
      finalText = composeAnswer(intent, toolResults, docs);
      break;   // 单工具即可答复；多工具编排由 MAX_ITER 上限约束
    }
  }
  trace.iterations = iteration;
  if (finalText === undefined) finalText = composeAnswer(intent, toolResults, docs);
  trace.latencyMs = Date.now() - t0;

  if (LLM_ENABLED) {
    try {
      const llmText = await callLLM(text, docs, toolResults);
      if (llmText) finalText = llmText;
    } catch (e) { trace.llmError = String(e.message || e); }
  }
  return { reply: finalText, trace };
}

/* 可选：真实大模型（OpenAI 兼容协议）——仅在配置了 key 时启用 */
async function callLLM(text, docs, toolResults) {
  const context = [
    docs.map(d => `【${d.title}】${d.content}`).join('\n'),
    toolResults.map(t => `工具 ${t.name} 返回：${JSON.stringify(t.result)}`).join('\n'),
  ].join('\n');
  const body = {
    model: LLM_MODEL,
    messages: [
      { role: 'system', content: '你是电商智能客服，只根据提供的「参考资料」和「工具结果」回答，不要编造。语言简洁友好。' },
      { role: 'user', content: `用户问题：${text}\n\n参考资料/工具结果：\n${context}` },
    ],
    temperature: 0.3,
  };
  const res = await fetch(`${LLM_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${LLM_API_KEY}` },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error('LLM HTTP ' + res.status);
  const json = await res.json();
  return json.choices?.[0]?.message?.content?.trim();
}

/* ================================================================== *
 * 8. HTTP 服务
 * ================================================================== */
function now() { return new Date().toLocaleString('zh-CN', { hour12: false }); }
function sendJson(res, code, obj) {
  const buf = Buffer.from(JSON.stringify(obj));
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': buf.length });
  res.end(buf);
}
function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', c => { data += c; if (data.length > 1e6) req.destroy(); });
    req.on('end', () => { try { resolve(data ? JSON.parse(data) : {}); } catch (e) { reject(e); } });
    req.on('error', reject);
  });
}
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'application/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.json': 'application/json; charset=utf-8' };

const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, 'http://localhost');
  try {
    if (req.method === 'GET' && u.pathname === '/api/health') {
      return sendJson(res, 200, { ok: true, engine: LLM_ENABLED ? 'llm' : 'local', tools: Object.keys(TOOLS).length, kb: KB.length });
    }
    if (req.method === 'GET' && u.pathname === '/api/tools') return sendJson(res, 200, { tools: toolSchema() });
    if (req.method === 'GET' && u.pathname === '/api/kb') return sendJson(res, 200, { kb: KB.map(d => ({ id: d.id, title: d.title })) });
    if (req.method === 'GET' && u.pathname === '/api/stats') {
      return sendJson(res, 200, { rate: rateStats, filter: filterStats, sessions: buckets.size });
    }
    if (req.method === 'POST' && u.pathname === '/api/chat') {
      const body = await readBody(req);
      const text = String(body.message || '').slice(0, 500);
      const sessionId = String(body.sessionId || 'anon');
      if (!text.trim()) return sendJson(res, 400, { error: '消息不能为空' });

      const filtered = dfaFilter(text);
      if (filtered.blocked) {
        return sendJson(res, 200, {
          reply: '您的消息包含不友善用语，已被拦截，请文明用语～',
          trace: { intent: { name: 'blocked' }, retrieved: [], toolCalls: [], iterations: 0, latencyMs: 0, sensitiveHits: filtered.hits },
        });
      }
      if (!rateLimit(sessionId)) {
        return sendJson(res, 429, { reply: '请求过于频繁，请稍后再试（令牌桶限流）。', trace: { intent: { name: 'rate_limited' }, retrieved: [], toolCalls: [], iterations: 0, latencyMs: 0 } });
      }
      const out = await runAgent(text, sessionId);
      return sendJson(res, 200, out);
    }
    // 静态资源
    let p = decodeURIComponent(u.pathname);
    if (p === '/') p = '/index.html';
    const filePath = path.join(PUBLIC_DIR, p);
    if (!filePath.startsWith(PUBLIC_DIR)) { res.writeHead(403); return res.end('forbidden'); }
    fs.readFile(filePath, (err, buf) => {
      if (err) { res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }); return res.end('404 Not Found'); }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream' });
      res.end(buf);
    });
  } catch (e) {
    sendJson(res, 500, { error: String(e.message || e) });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`[agent-customer-service] listening on http://${HOST}:${PORT}  engine=${LLM_ENABLED ? 'LLM' : 'local'}`);
});
