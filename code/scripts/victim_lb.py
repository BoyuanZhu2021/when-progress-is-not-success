"""Round-robin TCP balancer that fans one trainee across several victim servers.

Why this exists
---------------
Measured on the 9B x Qwen3.8-27B campaign: **rollout is 98% of step time** and the victim GPU sits
at 100% utilisation while the trainee GPU sits at 0%. The workload is victim-bound, so adding
trainee lanes against a single victim is worth **1.02x** -- they simply queue. The only way a third
card buys anything is to add victim CAPACITY, which means one trainee talking to two victims.

``a3_multiturn_train.py`` takes a single ``--base-url``, so rather than teach every call site about
a server list, this sits at that URL and spreads connections over the backends. The trainee is
unchanged and unaware.

Connection-level, not request-level
-----------------------------------
The OpenAI client keeps HTTP connections alive, so one TCP connection carries many sequential
requests. Balancing per CONNECTION is therefore the right granularity here: with
``--concurrency 32`` the client opens ~32 sockets, which round-robin to ~16 per backend. A
request-level HTTP proxy would have to parse and re-frame every response, adding latency to the
exact path that is already the bottleneck.

This changes wall-clock only. Each victim samples independently at temperature 0.7, exactly as the
single-server setup did, so the sampled distribution the DV is computed from is untouched.

Usage
-----
    python code/scripts/victim_lb.py --listen 8000 --backends 8001,8002
    python code/scripts/victim_lb.py --check --backends 8001,8002
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
import urllib.request


def parse_backends(spec: str, host: str = "127.0.0.1") -> list[tuple[str, int]]:
    """'8001,8002' or 'h:8001,h:8002' -> [(host, port), ...]."""
    out: list[tuple[str, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            h, _, p = part.rpartition(":")
            out.append((h or host, int(p)))
        else:
            out.append((host, int(part)))
    if not out:
        raise ValueError("no backends given")
    return out


class Balancer:
    def __init__(self, backends: list[tuple[str, int]]):
        self.backends = backends
        self._rr = itertools.cycle(range(len(backends)))
        self.counts = [0] * len(backends)
        self.failures = [0] * len(backends)

    def pick(self) -> int:
        return next(self._rr)

    async def _connect(self, idx: int):
        host, port = self.backends[idx]
        return await asyncio.open_connection(host, port)

    async def open_backend(self):
        """Try the next backend; on failure fall through to the others rather than dropping the
        client. A dead backend must degrade throughput, not kill a 12.8 h run."""
        first = self.pick()
        for offset in range(len(self.backends)):
            idx = (first + offset) % len(self.backends)
            try:
                reader, writer = await self._connect(idx)
                self.counts[idx] += 1
                return idx, reader, writer
            except Exception:
                self.failures[idx] += 1
        raise ConnectionError("no backend reachable")


async def _pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(bal: Balancer, cr, cw):
    try:
        _idx, br, bw = await bal.open_backend()
    except ConnectionError:
        cw.close()
        return
    # Both directions run concurrently; when either side closes, the other is torn down.
    await asyncio.gather(_pipe(cr, bw), _pipe(br, cw))


async def serve(listen: int, backends: list[tuple[str, int]], host: str = "127.0.0.1"):
    bal = Balancer(backends)
    server = await asyncio.start_server(lambda r, w: handle(bal, r, w), host, listen)
    print(f"victim_lb listening on {host}:{listen} -> {backends}", flush=True)

    async def report():
        while True:
            await asyncio.sleep(300)
            print(f"[lb] connections per backend: "
                  f"{dict(zip((f'{h}:{p}' for h, p in backends), bal.counts))} "
                  f"failures={bal.failures}", flush=True)

    asyncio.create_task(report())
    async with server:
        await server.serve_forever()


def check(backends: list[tuple[str, int]]) -> int:
    """Confirm every backend answers /v1/models before any training is launched."""
    bad = 0
    for h, p in backends:
        try:
            with urllib.request.urlopen(f"http://{h}:{p}/v1/models", timeout=10) as r:
                ok = r.status == 200
        except Exception as e:  # noqa: BLE001
            ok, e_ = False, e
        else:
            e_ = None
        print(f"  {h}:{p}  {'OK' if ok else 'UNREACHABLE ' + type(e_).__name__}")
        bad += not ok
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=int, default=8000)
    ap.add_argument("--backends", required=True, help="e.g. 8001,8002")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--check", action="store_true", help="probe the backends and exit")
    args = ap.parse_args()

    backends = parse_backends(args.backends, args.host)
    if args.check:
        return 1 if check(backends) else 0
    try:
        asyncio.run(serve(args.listen, backends, args.host))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
