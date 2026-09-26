"""Goldens for the victim load balancer, including a real end-to-end proxy round trip."""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from victim_lb import Balancer, handle, parse_backends  # noqa: E402


class TestParse(unittest.TestCase):
    def test_bare_ports_use_the_default_host(self):
        self.assertEqual(parse_backends("8001,8002"),
                         [("127.0.0.1", 8001), ("127.0.0.1", 8002)])

    def test_explicit_hosts(self):
        self.assertEqual(parse_backends("a:1,b:2"), [("a", 1), ("b", 2)])

    def test_whitespace_and_trailing_comma(self):
        self.assertEqual(parse_backends(" 8001 , 8002 ,"),
                         [("127.0.0.1", 8001), ("127.0.0.1", 8002)])

    def test_empty_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_backends("")


class TestRoundRobin(unittest.TestCase):
    def test_cycles_evenly(self):
        b = Balancer(parse_backends("1,2,3"))
        self.assertEqual([b.pick() for _ in range(7)], [0, 1, 2, 0, 1, 2, 0])


class TestProxying(unittest.IsolatedAsyncioTestCase):
    async def _echo_server(self, tag: str):
        """A backend that prefixes what it receives, so the test can tell which one served."""
        async def h(r, w):
            data = await r.read(1024)
            w.write(tag.encode() + data)
            await w.drain()
            w.close()
        s = await asyncio.start_server(h, "127.0.0.1", 0)
        return s, s.sockets[0].getsockname()[1]

    async def test_end_to_end_proxy_alternates_backends(self):
        """Drive real bytes THROUGH the proxy and check which backend answered.

        The earlier version of this test only exercised Balancer.pick(), duplicating
        test_cycles_evenly while leaving handle()/_pipe() -- the actual data path -- untested.
        The LB now sits on the critical path of a 12.8 h/run campaign, so the proxying itself
        has to be verified, not just the selection arithmetic.
        """
        s1, p1 = await self._echo_server("A")
        s2, p2 = await self._echo_server("B")
        bal = Balancer([("127.0.0.1", p1), ("127.0.0.1", p2)])
        srv = await asyncio.start_server(
            lambda r, w: handle(bal, r, w), "127.0.0.1", 0)
        lb_port = srv.sockets[0].getsockname()[1]

        seen = []
        for _ in range(4):
            r, w = await asyncio.open_connection("127.0.0.1", lb_port)
            w.write(b"ping")
            await w.drain()
            seen.append((await r.read(64)).decode())
            w.close()

        srv.close()
        await srv.wait_closed()
        for s in (s1, s2):
            s.close()

        self.assertEqual(seen, ["Aping", "Bping", "Aping", "Bping"],
                         "consecutive connections must alternate backends AND relay the payload")
        self.assertEqual(bal.counts, [2, 2])

    async def test_falls_through_when_a_backend_is_dead(self):
        """A dead victim must degrade throughput, not kill a 12.8 h run."""
        s_ok, p_ok = await self._echo_server("A")
        dead = 1          # port 1 is not listening
        bal = Balancer([("127.0.0.1", dead), ("127.0.0.1", p_ok)])
        idx, r, w = await bal.open_backend()
        w.close()
        s_ok.close()
        self.assertEqual(idx, 1, "should have fallen through to the reachable backend")
        self.assertEqual(bal.failures[0], 1)

    async def test_raises_when_no_backend_is_reachable(self):
        bal = Balancer([("127.0.0.1", 1), ("127.0.0.1", 2)])
        with self.assertRaises(ConnectionError):
            await bal.open_backend()


if __name__ == "__main__":
    unittest.main(verbosity=2)
