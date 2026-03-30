# BYOKG-RAG 模块化改造说明（Graph Evidence Provider）

本次改造将 BYOKG-RAG 从“端到端 KGQA 流程”收敛为可插拔的**图证据获取器**，以便直接挂到 DeepResearch 的工具层。

## 1. 目标接口

### 输入
- `question`
- `schema`（必须）
- `graph_context`（可选）

### 输出
- `graph_evidence`
  - `triples`
  - `paths`
  - `query_results`
- `linking_artifacts`
  - 最终实体、路径、DSL、OpenCypher、reranker 参数
  - refinement 历史
- `confidence`
- `coverage`

---

## 2. 模块落地

新增文件：`byokg_rag.py`

核心类：
- `BYOKGRAGProvider`
- `BYOKGValidator`
- `CypherDSLCompiler`
- `RerankerConfig`

并在 `tools.py` 中新增工具路由：
- `retrieve_graph_evidence`

这样上层工作流只需把它当“KG 证据检索工具”，无需暴露内部每一步。

---

## 3. 按要求实现的关键改动

### 3.1 保留多策略 + 迭代 refinement

`BYOKGRAGProvider.retrieve_graph_evidence()` 保留“多轮 refinement”模式，支持：
- 首轮 `mode=full`
- 校验失败后 `mode=repair-only`

并记录每轮 `refinement_history`。

### 3.2 新增工程 validator（可恢复分支）

`BYOKGValidator.validate()` 增加以下校验：
1. `entities` 空/重复/明显噪声过滤
2. `paths.relation` 必须属于 `schema.relation_types`
3. OpenCypher 基础检查：至少包含 `MATCH` 与 `RETURN`
4. 可选 dry-run：注入 `cypher_dry_run` 回调时，执行 `LIMIT 1` 检查

失败后触发 repair-only，输入包含：
- 错误列表
- schema 摘要
- 上一轮 artifacts
- 上一轮 graph_context

### 3.3 Graph Reranker 动态参数（L/k）

`_dynamic_reranker()` 根据：
- KG 稠密度（relation_types 数量）
- 问题是否多跳（关键词）

动态设置：
- `L`（hop）
- `k`（top_k）

### 3.4 OpenCypher 改为 DSL 约束输出

不强依赖“自由生成 Cypher”，而是：
1. linker 生成 schema-constrained DSL(JSON)
2. `CypherDSLCompiler.compile()` 编译成 Cypher
3. validator 再做检查/修复

DSL 结构支持：
- `match`
- `edges`
- `where`
- `return`
- `sort`
- `limit`

### 3.5 多策略候选 + 集成重排（新增）

`BYOKGRAGProvider` 现在支持多种 linking strategy 并进行候选集成：
- `entity_path`
- `relation_path`
- `query_shape`

每个策略产出实体/路径/DSL 后会进行打分排序，输出：
- `linking_artifacts.strategy_ranking`
- `linking_artifacts.selected_strategy`

并将 top 候选进行实体和路径合并，提升复杂问题的召回稳定性。

### 3.6 Schema Focus/Pruning（新增）

针对大 schema 场景，在调用 linker 前会基于问题词与关系名的匹配构造：
- `focus_relation_types`
- `schema_pruned=true`

用于减少搜索空间和噪声路径。

### 3.7 查询执行与证据回填（新增）

新增可选 `cypher_executor` 注入能力，provider 可在生成 Cypher 后执行查询并回填：
- `graph_evidence.query_results`

同时将查询命中数量纳入 `coverage` 估计。

---

## 4. 与原功能兼容性

- `ToolRouter` 原有工具均保留。
- 新工具 `retrieve_graph_evidence` 为增量能力，不破坏现有 `search/open_webpage/retrieve_lexical_graph/analyze_image`。
- 默认 linker 缺省时有 fallback，不会导致调用崩溃。

---

## 5. 建议接入方式（上层）

上层只需在某轮工具调用中传入：

```json
{
  "tool_name": "retrieve_graph_evidence",
  "args": {
    "question": "...",
    "schema": {"relation_types": ["..."], "node_types": ["..."], "summary": "..."},
    "graph_context": "..."
  }
}
```

并基于 `confidence/coverage` 决定是否继续：
- 低置信/低覆盖：继续外层检索或追问
- 高置信/高覆盖：直接进入总结/回答
