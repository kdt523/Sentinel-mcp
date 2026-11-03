# gateway

MCP proxy gateway: session tracker, policy engine, decider, enforcer, decision log
(Phases 1, 5, 6). Configuration: `gateway/config.py` (env prefix `SENTINEL_`).

## Human approvals (`REQUIRE_APPROVAL`)

- The agent's tool call is **held open** while a reviewer approves or denies it
  (statuses: `pending`, `approved`, `denied`, `expired`, `cancelled`).
- **Timeout = deny**, whatever the tool's fail mode. The agent receives a message saying the
  request *timed out* and was treated as denied (status `expired`).
- If the agent disconnects or cancels the call, the approval becomes `cancelled`.
- While waiting, the gateway sends MCP **progress notifications** every
  `SENTINEL_APPROVAL_PROGRESS_INTERVAL_S` — but only to clients that sent a `progressToken`.

### Why the default timeout is 50 s

`SENTINEL_APPROVAL_TIMEOUT_S` defaults to **50 s** on purpose. Progress notifications do
**not** extend a request's timeout in the mcp v2 Python client (and many other clients),
and 60 s is a common client default request timeout. A longer approval window would make
the client give up first and surface a transport error instead of a clear denial.

Raise the timeout only if every client you serve either has a longer request timeout or
resets its timeout on progress notifications.
