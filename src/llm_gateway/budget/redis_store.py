"""Redis budget store: one Lua script per operation, so reserve and settle are atomic (ADR-3, ADR-0004).

Budget keys never get a TTL: under `maxmemory-policy volatile-lfu` only keys with a TTL (cache entries) are
evicted. Reservations expire by a deadline in a sorted set, checked inside `reserve` with Redis `TIME`."""

from redis.exceptions import RedisError

from llm_gateway.budget.store import BudgetUnavailable, Caps, Scopes

RES_ZSET = "budget:reservations"
RES_HASH = "budget:reservation_meta"

# KEYS: run, client_day, global_day, zset, hash.
# ARGV: amount, ttl_s, id, run_cap, client_cap, global_cap (-1 = no cap)
RESERVE = """
local now = tonumber(redis.call('TIME')[1])
local expired = redis.call('ZRANGEBYSCORE', KEYS[4], '-inf', now)
for _, rid in ipairs(expired) do
  local meta = redis.call('HGET', KEYS[5], rid)
  if meta then
    local amt, k1, k2, k3 = string.match(meta, '^(%d+)|([^|]*)|([^|]*)|([^|]*)$')
    for _, k in ipairs({k1, k2, k3}) do
      if k ~= '' then redis.call('HINCRBY', k, 'reserved', -tonumber(amt)) end
    end
    redis.call('HDEL', KEYS[5], rid)
  end
  redis.call('ZREM', KEYS[4], rid)
end
local amount = tonumber(ARGV[1])
for i = 1, 3 do
  local cap = tonumber(ARGV[3 + i])
  if cap >= 0 then
    local spent = tonumber(redis.call('HGET', KEYS[i], 'spent') or '0')
    local reserved = tonumber(redis.call('HGET', KEYS[i], 'reserved') or '0')
    if spent + reserved + amount > cap then return 0 end
  end
end
for i = 1, 3 do redis.call('HINCRBY', KEYS[i], 'reserved', amount) end
redis.call('ZADD', KEYS[4], now + tonumber(ARGV[2]), ARGV[3])
redis.call('HSET', KEYS[5], ARGV[3], ARGV[1] .. '|' .. KEYS[1] .. '|' .. KEYS[2] .. '|' .. KEYS[3])
return 1
"""

# KEYS: run, client_day, global_day, zset, hash.  ARGV: real, id.  Returns 1 when the reservation had expired.
SETTLE = """
local late = 1
local meta = redis.call('HGET', KEYS[5], ARGV[2])
if meta then
  late = 0
  local amt, k1, k2, k3 = string.match(meta, '^(%d+)|([^|]*)|([^|]*)|([^|]*)$')
  for _, k in ipairs({k1, k2, k3}) do
    if k ~= '' then redis.call('HINCRBY', k, 'reserved', -tonumber(amt)) end
  end
  redis.call('HDEL', KEYS[5], ARGV[2])
  redis.call('ZREM', KEYS[4], ARGV[2])
end
for i = 1, 3 do redis.call('HINCRBY', KEYS[i], 'spent', tonumber(ARGV[1])) end
return late
"""


class RedisBudget:
    def __init__(self, redis) -> None:  # redis.asyncio.Redis (or fakeredis in tests)
        self._r = redis
        self._reserve = redis.register_script(RESERVE)
        self._settle = redis.register_script(SETTLE)

    def _keys(self, scopes: Scopes) -> list[str]:
        return [*scopes.as_list(), RES_ZSET, RES_HASH]

    async def reserve(self, rid: str, amount: int, ttl_s: float, scopes: Scopes, caps: Caps) -> bool:
        cap = lambda c: -1 if c is None else c  # noqa: E731
        try:
            ok = await self._reserve(
                keys=self._keys(scopes),
                args=[amount, int(ttl_s) + 1, rid, cap(caps.run), cap(caps.client_day), cap(caps.global_day)],
            )
        except RedisError as exc:
            raise BudgetUnavailable(str(exc)) from exc
        return bool(ok)

    async def settle(self, rid: str, real: int, scopes: Scopes) -> bool:
        try:
            return bool(await self._settle(keys=self._keys(scopes), args=[real, rid]))
        except RedisError as exc:
            raise BudgetUnavailable(str(exc)) from exc

    async def snapshot(self, scopes: Scopes) -> dict:
        try:
            out = {}
            for name, key in zip(("run", "client_day", "global_day"), scopes.as_list(), strict=True):
                h = await self._r.hgetall(key)
                get = lambda f, h=h: int(h.get(f.encode(), h.get(f, 0)))  # noqa: E731
                out[name] = {"spent": get("spent"), "reserved": get("reserved")}
            return out
        except RedisError as exc:
            raise BudgetUnavailable(str(exc)) from exc
