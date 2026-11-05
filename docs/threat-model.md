# SentinelMCP threat model

Living document. Each entry states an assumption or limitation, why it exists, and what
would change it. Last updated: 2026-11-04.

## 1. What we protect

- **Secrets and personal data** reachable through tools (`.env`, keys, customer records).
- **Integrity** of the workspace, databases and outbound channels (email, HTTP).
- **The agent's instructions**: tool outputs and tool descriptions must not be able to
  take over the agent (indirect injection, tool poisoning, rug-pulls).

## 2. Trust boundaries

| Component | Trusted? | Notes |
|---|---|---|
| Gateway, policy file, Redis, Postgres | yes | operator-controlled |
| Agent / MCP client | **no** | may be steered by injected content |
| Upstream tool servers | **no** | may be poisoned or compromised |
| Tool outputs and descriptions | **no** | scanned on every response (`direction=output`) |
| Human reviewers (approvals) | yes | authenticated in Phase 6 |

## 3. Session identity (decided 2026-11-04)

Session-level detection — taint tracking, multi-step exfiltration, per-session description
hashes, revoked tools, termination — only works if the gateway can tell which requests
belong to the same agent session.

**Current design.** Sessions are resolved by a `SessionResolver`. The first and only
implementation uses the **MCP transport session id** (`Mcp-Session-Id`) of the stateful
MCP protocol (`2025-11-25`), which the server assigns per connection.

**Protocol dependency.** The newer MCP protocol (`2026-07-28`) — negotiated by default by
mcp v2 clients — is **stateless**: no `Mcp-Session-Id`, no per-connection server session.
Under it the gateway cannot link "read `.env`" to a later "send it out".
Therefore the gateway **declines the stateless protocol**: mcp v2 clients in `auto` mode
fall back to the stateful protocol automatically (pinned by a test). Clients that only
speak the stateless protocol are **refused with an explicit reason** (fail closed) rather
than served without session tracking.

*What would change it:* Phase 6 adds a gateway-issued per-agent token resolver (agent
authentication), which identifies sessions without relying on the transport and allows
the stateless protocol to be supported safely.

**Not built: client-supplied sub-sessions.** A header that lets one connection carry
several logical sessions is deliberately absent. If ever added it must be: behind a config
flag that is **off by default**; accepted **only from trusted clients**; the sub-session id
**derived from the connection session id** (never chosen freely by the client); and taint
**also tracked at the connection level**, so switching sub-sessions can never clear it.
Otherwise an attacker could read a secret in one sub-session and exfiltrate it from a
"clean" one.

## 4. Known limitations

| # | Limitation | Mitigation / plan |
|---|---|---|
| L1 | Content fingerprints catch verbatim copies and URL / base64 / hex encodings up to two layers deep (bounded: 256 KB input, 64 segments per layer, 512 KB decoded; updated 2026-11-05), not paraphrases, three or more layers, or other transformations (compression, encryption, ciphers). | Reasoning guard (Phase 4/5) for semantic leaks. |
| L2 | PII detection is threshold based (>= 3 e-mail addresses or >= 2 phone numbers). Single values in free text are not tainted. | Dataset-driven tuning in Phase 2. |
| L3 | `http_fetch` allowlists hosts by name only; no resolved-IP / DNS-rebinding checks. | Phase 6 hardening item. |
| L4 | Rule-based sanitizer: weak signals (comments, role-play blocks, pseudo-tags) fall back to SANITIZE and are logged `escalated` until the guard exists. | Phase 5 routes escalations to the guard. |
| L5 | `http_fetch` text mode drops HTML comments, so comment-borne injections never reach the agent in that mode. | Labelled `neutralised_by_tool` in Phase 2; not credited to the gateway. |
| L6 | Approvals hold the call open up to 50 s because many clients do not extend timeouts on progress notifications. | Raise only for clients that reset on progress (see `gateway/README.md`). |
| L7 | Raw capture mode writes unredacted events to disk. | Refused outside the sandbox; startup warning; files gitignored. |
| L9 | Encoded key-material paths in shell commands (2026-11-05). Filesystem tools take literal paths, so every spelling (`./.ssh/id_rsa`, `docs/../.ssh/id_rsa`) is blocked by `key-material-paths`. In the shell an encoded path needs `python3` (no shell, no pipes). `key-material-shell` (`decode: true`) also matches base64 / hex / percent-encoded quoted literals of any length, so these are BLOCKED. Paths *computed* at run time (character codes, string arithmetic, any program logic) cannot be decoded statically. | Those fall to `python3-escalate` (never auto-allowed: REQUIRE_APPROVAL, logged `escalated`) until the guard (Phase 5) judges them. If approved, the key appears in the output, which taints the session, so any later egress of it is still terminated. |
| L8 | A description change can surface through an SDK-internal `tools/list` that the mcp v2 client issues *inside* `call_tool` (tool schema not cached), i.e. after the call's request was already approved and sent upstream. | **Tightened 2026-11-04:** the change is detected during the call; the call's OUTPUT is blocked (`builtin:description-changed-mid-call`) and the changed tool(s) are revoked regardless of the per-tool description rule, so the upstream result never reaches the agent. Residual risk: the upstream already *received* the request, so side effects of that one call (e.g. a write) are not prevented. Shadow mode logs only. |
