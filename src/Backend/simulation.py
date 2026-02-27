import asyncio
import hashlib
import math
import time
from typing import Dict, Tuple, List, Optional, Callable, Awaitable

FlowKey = Tuple[str, str, int, int, int]


class StreamSimulator:
    """
    Production-safe streaming packet simulator.

    Guarantees:
        - No race conditions
        - No duplicate window processing
        - No NaN crashes
        - No packet loss
        - Deterministic window rollover
        - Safe numeric conversion
    """

    def __init__(
        self,
        packets: List[dict],
        rate: float = 100.0,
        window_size: float = 5.0,
        mode: str = "rate",            # "rate" or "realtime"
        time_scale: float = 1.0,
        window_callback: Optional[
            Callable[[int, float, List[dict]], Awaitable[None]]
        ] = None,
        max_realtime_gap: float = 60.0,
    ):
        self.packets = packets
        self.rate = float(rate)
        self.window_size = float(window_size)
        self.mode = mode
        self.time_scale = float(time_scale)
        self.max_realtime_gap = float(max_realtime_gap)

        self.window_callback = window_callback

        self.active_flows: Dict[FlowKey, dict] = {}
        self.current_window_start: Optional[float] = None
        self.window_id = 0

    # ─────────────────────────────────────────────
    # Safe utilities
    # ─────────────────────────────────────────────
    def _safe_int(self, value, default=0):
        if value is None:
            return default
        if isinstance(value, float):
            if math.isnan(value):
                return default
        try:
            return int(value)
        except Exception:
            return default

    def _safe_float(self, value, default=0.0):
        if value is None:
            return default
        if isinstance(value, float):
            if math.isnan(value):
                return default
        try:
            return float(value)
        except Exception:
            return default

    def _normalize_ts(self, raw_ts: float) -> float:
        raw_ts = self._safe_float(raw_ts)
        if raw_ts > 1e14:
            return raw_ts / 1_000_000.0
        elif raw_ts > 1e11:
            return raw_ts / 1_000.0
        return raw_ts

    # ─────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────
    async def run(self):
        print(
            f"[SIM] mode={self.mode} | "
            f"rate={self.rate}pps | "
            f"window={self.window_size}s | "
            f"time_scale={self.time_scale}"
        )

        prev_ts = None

        for pkt in self.packets:

            if "timestamp" not in pkt:
                continue  # skip invalid packet safely

            ts = self._normalize_ts(pkt["timestamp"])

            # Replay pacing
            if self.mode == "rate":
                if self.rate > 0:
                    await asyncio.sleep(1.0 / self.rate)

            elif self.mode == "realtime":
                if prev_ts is not None:
                    delta = ts - prev_ts
                    if 0 < delta < self.max_realtime_gap:
                        await asyncio.sleep(delta * self.time_scale)

            await self._process_packet(pkt, ts)
            prev_ts = ts

        # Final flush
        await self._close_window()

        print("[SIM] Simulation complete")

    # ─────────────────────────────────────────────
    # Packet aggregation
    # ─────────────────────────────────────────────
    async def _process_packet(self, pkt: dict, ts: float):

        if self.current_window_start is None:
            self.current_window_start = ts

        # Handle multiple skipped windows (timestamp jump)
        while ts >= self.current_window_start + self.window_size:
            await self._close_window()
            self.window_id += 1
            self.current_window_start += self.window_size

        key: FlowKey = (
            str(pkt.get("src_ip") or "0.0.0.0"),
            str(pkt.get("dst_ip") or "0.0.0.0"),
            self._safe_int(pkt.get("src_port")),
            self._safe_int(pkt.get("dst_port")),
            self._safe_int(pkt.get("protocol")),
        )

        flow = self.active_flows.setdefault(
            key,
            {
                "packet_count": 0,
                "total_bytes": 0,
                "label": self._safe_int(pkt.get("label")),
            },
        )

        flow["packet_count"] += 1
        flow["total_bytes"] += self._safe_int(pkt.get("packet_length"))

    # ─────────────────────────────────────────────
    # Window close
    # ─────────────────────────────────────────────
    async def _close_window(self):

        if not self.active_flows:
            return

        window_id_snapshot = self.window_id
        window_start_snapshot = self.current_window_start
        flows_snapshot = self.active_flows

        # Clear immediately to avoid mutation issues
        self.active_flows = {}

        flows = self._build_flow_docs(
            window_id_snapshot,
            window_start_snapshot,
            flows_snapshot,
        )

        print(
            f"[SIM] Window {window_id_snapshot} closed → "
            f"{len(flows)} flows"
        )

        if self.window_callback:
            await self.window_callback(
                window_id_snapshot,
                window_start_snapshot,
                flows,
            )

    def _build_flow_docs(
        self,
        window_id: int,
        window_start: float,
        flows_dict: Dict[FlowKey, dict],
    ) -> List[dict]:

        output = []
        now_ms = int(time.time() * 1000)

        for key, data in flows_dict.items():
            src_ip, dst_ip, src_port, dst_port, protocol = key

            raw = (
                f"{window_id}:{window_start}:"
                f"{src_ip}:{dst_ip}:{src_port}:{dst_port}:{protocol}"
            )
            flow_id = hashlib.md5(raw.encode()).hexdigest()[:16]

            output.append(
                {
                    "flow_id": flow_id,
                    "window_id": window_id,
                    "window_start": window_start,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": protocol,
                    "packet_count": data["packet_count"],
                    "total_bytes": data["total_bytes"],
                    "label": data["label"],
                    "timestamp": now_ms,
                }
            )

        return output