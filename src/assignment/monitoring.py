"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def record(self, *, blocked: bool, layer: str | None = None,
               judge_checked: bool = False, judge_failed: bool = False):
        """Cập nhật bộ đếm sau mỗi request (pipeline gọi)."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
        if layer == "rate_limiter":
            self.rate_limit_hits += 1
        if judge_checked:
            self.judge_checks += 1
            if judge_failed:
                self.judge_fails += 1

    def _set_alert(self, metric: str, value: float, threshold: float, message: str):
        # Mỗi metric chỉ giữ 1 alert (giá trị mới nhất) — tránh spam alert trùng.
        self.alerts = [a for a in self.alerts if a.metric != metric]
        self.alerts.append(Alert(metric=metric, value=value, threshold=threshold, message=message))

    def check_metrics(self) -> list[Alert]:
        """Tính các tỉ lệ và tạo Alert khi vượt ngưỡng."""
        snap = self.snapshot()
        if self.total_requests and snap["block_rate"] > self.block_rate_threshold:
            self._set_alert(
                "block_rate", round(snap["block_rate"], 3), self.block_rate_threshold,
                f"Block rate {snap['block_rate']:.0%} vượt ngưỡng "
                f"{self.block_rate_threshold:.0%} — có thể đang bị tấn công hoặc guardrail chặn nhầm.",
            )
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            self._set_alert(
                "rate_limit_hits", self.rate_limit_hits, self.rate_limit_hit_threshold,
                f"{self.rate_limit_hits} request bị rate-limit — nghi ngờ spam / flooding.",
            )
        if self.judge_checks and snap["judge_fail_rate"] > self.judge_fail_rate_threshold:
            self._set_alert(
                "judge_fail_rate", round(snap["judge_fail_rate"], 3), self.judge_fail_rate_threshold,
                f"LLM-Judge đánh UNSAFE {snap['judge_fail_rate']:.0%} câu trả lời.",
            )
        for alert in self.alerts:
            print(f"[ALERT] {alert.metric}: {alert.message}")
        return self.alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        from datetime import datetime, timezone

        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot()
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload["thresholds"] = {
            "block_rate": self.block_rate_threshold,
            "rate_limit_hits": self.rate_limit_hit_threshold,
            "judge_fail_rate": self.judge_fail_rate_threshold,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
