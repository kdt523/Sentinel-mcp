"""SessionTracker against the real Redis from docker compose (`make dev`)."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator

import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from gateway.config import Settings
from gateway.models import Action, Decision, Direction, Event, Tier, sha256_text
from gateway.session import KEY_PREFIX, SessionTracker, create_redis

pytestmark = pytest.mark.integration


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    client = create_redis(Settings())
    try:
        await client.ping()
    except (RedisConnectionError, RedisTimeoutError, OSError):
        await client.aclose()
        pytest.skip("redis not reachable (run `make dev`)")
    yield client
    await client.aclose()


@pytest.fixture
async def sid(redis: Redis) -> AsyncIterator[str]:
    session_id = f"itest-{uuid.uuid4().hex}"
    yield session_id
    keys = [k async for k in redis.scan_iter(f"{KEY_PREFIX}:{session_id}:*")]
    if keys:
        await redis.delete(*keys)


def _req(sid: str, path: str) -> Event:
    return Event(
        session_id=sid, direction=Direction.REQUEST, server="filesystem", tool="read_file",
        args={"path": path},
    )  # fmt: skip


def _dec(risk: float) -> Decision:
    return Decision(action=Action.ALLOW, risk=risk, reason="t", tier=Tier.POLICY, model_version="t")


async def test_lua_scripts_and_transactions_on_real_redis(redis: Redis, sid: str) -> None:
    tracker = SessionTracker(redis, ttl_s=120, history_len=5, risk_half_life_s=3600)
    await asyncio.gather(*(tracker.record(_req(sid, f"f{i}"), _dec(0.1)) for i in range(40)))
    st = await tracker.get(sid)
    assert st.event_count == 40
    assert len(st.files_read) == 40
    assert len(st.history) == 5
    assert st.risk == pytest.approx(4.0, rel=1e-3)

    first = await tracker.observe_description(sid, "weather.get_forecast", sha256_text("a"))
    changed = await tracker.observe_description(sid, "weather.get_forecast", sha256_text("b"))
    assert first.first_seen and changed.changed

    keys = [k async for k in redis.scan_iter(f"{KEY_PREFIX}:{sid}:*")]
    ttls = [await redis.ttl(k) for k in keys]
    assert keys and all(0 < t <= 120 for t in ttls)


async def test_unreachable_redis_fails_fast() -> None:
    settings = Settings(
        redis_url="redis://127.0.0.1:1/0", redis_connect_timeout_s=0.5, redis_socket_timeout_s=0.5
    )
    client = create_redis(settings)
    t0 = time.perf_counter()
    with pytest.raises((RedisConnectionError, RedisTimeoutError, OSError)):
        await SessionTracker(client).is_terminated("x")
    await client.aclose()
    assert time.perf_counter() - t0 < 5.0


async def test_approval_transitions_on_real_redis(redis: Redis) -> None:
    from gateway.approvals import ApprovalStore
    from gateway.models import ApprovalStatus

    store = ApprovalStore(redis, ttl_s=120)
    ev = _req(f"itest-{uuid.uuid4().hex}", "x")
    a = await store.create(ev, _dec(0.5), preview="{}")
    try:
        assert (await store.transition(a.approval_id, ApprovalStatus.DENIED, by="kd")).ok
        late = await store.transition(a.approval_id, ApprovalStatus.EXPIRED)
        assert not late.ok and late.status is ApprovalStatus.DENIED
        assert a.approval_id not in {p.approval_id for p in await store.list_pending()}
        assert 0 < await redis.ttl(f"sentinel:approval:{a.approval_id}") <= 120
    finally:
        await redis.delete(f"sentinel:approval:{a.approval_id}")
