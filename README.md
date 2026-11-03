# SentinelMCP

[![CI](https://github.com/kdt523/Sentinel-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/kdt523/Sentinel-mcp/actions/workflows/ci.yml)

**A session-aware, bidirectional security gateway for AI agents that use the Model Context Protocol (MCP).**

> **Status: early development (Phase 0 — project setup).** Nothing below the architecture section works yet.

---

## The problem

AI agents no longer just chat — they read files, run shell commands, fetch web pages, query databases and send email through tools. That opens new attack paths:

- **Indirect prompt injection** — malicious instructions hidden in a tool's *output* (a web page, a document, a DB row).
- **Tool poisoning** — hidden instructions inside a tool's *description*.
- **Multi-step attacks** — each call looks harmless alone (read a secret… later send it out), so single-call checks miss them.
- **Obfuscation** — payloads disguised with base64, unicode tricks, synonyms or argument splitting slip past keyword filters.

## The approach

SentinelMCP is a drop-in MCP proxy between the agent and its tools. For every tool **request** and every tool **output**, with the **whole session** in view, it chooses one of six actions:

| Action | Meaning |
|---|---|
| `ALLOW` | Forward as-is |
| `BLOCK` | Reject this call; session continues |
| `SANITIZE` | Strip injected content from a tool output, then forward |
| `REQUIRE_APPROVAL` | Pause until a human approves or denies |
| `REVOKE_TOOL` | Disable the tool for the rest of the session |
| `TERMINATE_SESSION` | End the session |

Decisions come from a **two-speed defence**: a YAML policy engine and a fast ONNX classifier handle clear-cut cases; uncertain cases go to a reasoning guard (Qwen2.5-3B + LoRA, trained with GRPO against an adaptive LLM attacker). Every decision carries a risk score and a short reason, and is logged and monitored.

## Architecture

```
 Agent (LangGraph / any MCP client)
        │  MCP (JSON-RPC: tools/list, tools/call)
        ▼
 ┌──────────────────── SentinelMCP Gateway (FastAPI + MCP SDK) ────────────────────┐
 │ 1. Session Tracker (Redis): rolling state — files read, secrets touched,         │
 │    taint flags, network egress, tool history, risk accumulator                   │
 │ 2. Policy Engine (YAML): hard allow/deny rules per tool                          │
 │ 3. Fast Tier (ONNX classifier): risk score in ~ms                                │
 │ 4. Router: confident → decide; uncertain band → Reasoning Guard                  │
 │ 5. Reasoning Guard (vLLM, Qwen2.5-3B + GRPO LoRA): JSON decision                 │
 │ 6. Enforcer: apply action; shadow mode = log only; fail-open/closed on error     │
 │ 7. Output Scanner: same pipeline on tool RESULTS before they reach the agent     │
 │ 8. Decision Log (Postgres) + Metrics (Prometheus) + Feedback API                 │
 └──────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 MCP tool servers: filesystem (sandbox), shell (sandboxed container), http_fetch,
 sql_db (SQLite), email_send (mock), plus deliberately "poisoned" test servers
```

_A rendered diagram will replace this once the system is built._

## Quickstart (development)

Requirements: [uv](https://docs.astral.sh/uv/), GNU Make, Docker (for the local stack).

```bash
make install    # create .venv with dev dependencies
cp .env.example .env
make lint       # ruff + mypy
make test       # unit tests
make dev        # start redis + postgres (docker compose)
```

## Status

- [ ] Phase 0 — Project setup _(in progress)_
- [ ] Phase 1 — MCP tool servers + gateway (policy only)
- [ ] Phase 2 — Dataset & benchmark
- [ ] Phase 3 — Baselines
- [ ] Phase 4 — RL environment + GRPO training
- [ ] Phase 5 — Two-speed serving
- [ ] Phase 6 — Production layer
- [ ] Phase 7 — Evaluation & write-up

## Results

**TBD.** This section will contain only numbers produced by real runs, each reproducible by a documented command.

## Safety note

All tools run against a mock sandbox with clearly fake secrets. Attack samples in this repository exist only to test defences and target the sandbox only.
