"""Three stream-specific research agents with deep iterative analysis."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sailgp_analysis.analytics import add_foiling
from sailgp_analysis.data_loader import load_all_boats, load_metadata
from sailgp_analysis.deep_agents.shared_memory import Evidence, SharedMemory

# Re-export load - marks as dict from data_loader
from sailgp_analysis import data_loader


def _load_marks_dict(data_root):
    return data_loader.load_marks(data_root)


class StreamResearchAgent(ABC):
    stream_id: str
    stream_name: str

    def __init__(self, data_root, memory: SharedMemory, run_id: int):
        self.data_root = data_root
        self.memory = memory
        self.run_id = run_id
        self.df: pd.DataFrame | None = None
        self.df_racing: pd.DataFrame | None = None
        self.meta: pd.DataFrame | None = None
        self.marks: dict | None = None

    def load(self) -> None:
        self.df = load_all_boats(self.data_root)
        self.df_racing = add_foiling(self.df[self.df["TRK_BOAT_RACE_STATUS_unk"] == 2])
        self.meta = load_metadata(self.data_root)
        self.marks = _load_marks_dict(self.data_root)

    @abstractmethod
    def research_cycle(self) -> list[dict]:
        """One deep research pass; returns findings posted this cycle."""

    def _post_finding(self, statement: str, source: str, metric: str, value, strength: float, novelty: float = 0.6) -> dict:
        h = self.memory.find_hypothesis(statement[:40], self.stream_id)
        if not h:
            h = self.memory.add_hypothesis(self.stream_id, statement, novelty=novelty)
            h.status = "testing"
        ev = Evidence(
            stream=self.stream_id,
            source=source,
            metric=metric,
            value=value if isinstance(value, (int, float, str)) else str(value),
            strength=strength,
            run_id=self.run_id,
        )
        self.memory.add_evidence(h.id, ev)
        self.memory.post_message(
            self.stream_id, "coordinator", "finding",
            {"hypothesis_id": h.id, "statement": statement, "confidence": h.confidence},
        )
        return {"hypothesis_id": h.id, "statement": statement, "confidence": h.confidence}


class AboveWaterAgent(StreamResearchAgent):
    stream_id = "above"
    stream_name = "Above the Water"

    def research_cycle(self) -> list[dict]:
        findings = []
        if self.marks is None or self.df_racing is None:
            return findings

        for (venue, race_label), mdf in self.marks.items():
            wg = mdf[mdf["MARK"].astype(str).str.startswith("WG", na=False)]
            lg = mdf[mdf["MARK"].astype(str).str.startswith("LG", na=False)]
            if len(wg) < 5 or len(lg) < 5:
                continue
            delta_tws = float(wg["TWS_km_h_1"].mean() - lg["TWS_km_h_1"].mean())
            delta_twd = float(wg["TWD_deg"].mean() - lg["TWD_deg"].mean())
            key = f"{venue}/{race_label}"
            strength = min(0.95, abs(delta_tws) / 8)
            if abs(delta_tws) > 1.5:
                findings.append(self._post_finding(
                    f"Windward-leeward TWS gradient in {key}: WG−LG = {delta_tws:.1f} km/h",
                    source="marks.csv", metric="wg_minus_lg_tws", value=round(delta_tws, 2),
                    strength=strength, novelty=0.7 if abs(delta_tws) > 4 else 0.5,
                ))
            if abs(delta_twd) > 15:
                findings.append(self._post_finding(
                    f"Directional shear {key}: ΔTWD(WG−LG) = {delta_twd:.0f}°",
                    source="marks.csv", metric="wg_minus_lg_twd", value=round(delta_twd, 1),
                    strength=min(0.9, abs(delta_twd) / 40), novelty=0.75,
                ))

        # Dirty air proxy: close following + speed drop
        sub = self.df_racing.dropna(subset=["PC_DTB_m", "GPS_SOG_km_h_1"])
        close = sub[sub["PC_DTB_m"] < 80]
        if len(close) > 100:
            far = sub[sub["PC_DTB_m"] > 200]
            if len(far) > 100:
                close_spd = close["GPS_SOG_km_h_1"].mean()
                far_spd = far["GPS_SOG_km_h_1"].mean()
                drop = far_spd - close_spd
                if drop > 2:
                    findings.append(self._post_finding(
                        f"Boats within 80m of leader average {drop:.1f} km/h slower SOG (dirty-air proxy)",
                        source="boats PC_DTB_m", metric="speed_drop_close_following",
                        value=round(drop, 2), strength=min(0.85, drop / 10), novelty=0.8,
                    ))

        # Boat vs mark wind bias
        for venue in sub["venue"].unique():
            vsub = sub[sub["venue"] == venue].dropna(subset=["TWS_SGP_km_h_1"])
            if len(vsub) < 500:
                continue
            boat_mean = float(vsub["TWS_SGP_km_h_1"].mean())
            mark_means = []
            for (v, r), mdf in self.marks.items():
                if v != venue:
                    continue
                mark_means.append(float(mdf["TWS_km_h_1"].mean()))
            if mark_means:
                bias = boat_mean - float(np.mean(mark_means))
                if abs(bias) > 1:
                    findings.append(self._post_finding(
                        f"{venue}: onboard TWS bias vs mark average = {bias:+.1f} km/h",
                        source="TWS_SGP vs marks", metric="tws_bias", value=round(bias, 2),
                        strength=min(0.8, abs(bias) / 5), novelty=0.65,
                    ))
        return findings


class OnWaterAgent(StreamResearchAgent):
    stream_id = "on_water"
    stream_name = "On the Water"

    def research_cycle(self) -> list[dict]:
        findings = []
        if self.df_racing is None:
            return findings

        # Leader vs mid-fleet foil/wing signature
        for (venue, race_label), grp in self.df_racing.groupby(["venue", "race_label"]):
            leaders = grp[grp["TRK_RACE_RANK_unk"] == 1]
            mid = grp[grp["TRK_RACE_RANK_unk"].between(4, 7)]
            if len(leaders) < 30 or len(mid) < 30:
                continue
            foil_l = leaders["foiling"].mean() * 100
            foil_m = mid["foiling"].mean() * 100
            if foil_l - foil_m > 8:
                findings.append(self._post_finding(
                    f"{venue} {race_label}: leaders foiling {foil_l - foil_m:.0f}% more than mid-fleet",
                    source="foiling flag", metric="leader_foil_gap", value=round(foil_l - foil_m, 1),
                    strength=min(0.9, (foil_l - foil_m) / 25), novelty=0.85,
                ))
            wing_l = leaders["ANGLE_WING_ROT_deg"].mean()
            wing_m = mid["ANGLE_WING_ROT_deg"].mean()
            if abs(wing_l - wing_m) > 3:
                findings.append(self._post_finding(
                    f"{venue} {race_label}: leaders wing rot differs by {wing_l - wing_m:+.1f}° vs mid-fleet",
                    source="ANGLE_WING_ROT_deg", metric="wing_rot_leader_gap",
                    value=round(wing_l - wing_m, 2), strength=0.7, novelty=0.78,
                ))

        # VMG efficiency by TWA bucket
        sub = self.df_racing.dropna(subset=["TWA_SGP_deg", "VMG_km_h_1", "TRK_RACE_RANK_unk"])
        sub = sub.assign(twa_bin=(sub["TWA_SGP_deg"] // 20) * 20)
        for twa_bin, g in sub.groupby("twa_bin"):
            if len(g) < 200:
                continue
            lead_vmg = g[g["TRK_RACE_RANK_unk"] <= 2]["VMG_km_h_1"].mean()
            tail_vmg = g[g["TRK_RACE_RANK_unk"] >= 8]["VMG_km_h_1"].mean()
            if pd.notna(lead_vmg) and pd.notna(tail_vmg) and lead_vmg - tail_vmg > 3:
                findings.append(self._post_finding(
                    f"TWA {int(twa_bin)}°: top-2 VMG exceeds tail by {lead_vmg - tail_vmg:.1f} km/h",
                    source="VMG by TWA", metric=f"vmg_gap_twa_{int(twa_bin)}",
                    value=round(lead_vmg - tail_vmg, 2), strength=0.75, novelty=0.72,
                ))

        # Speed consistency (std) leaders vs others
        for team in sub["team"].unique()[:12]:
            t = sub[sub["team"] == team]
            if len(t) < 100:
                continue
            spd_std = t["BOAT_SPEED_km_h_1"].std()
            if spd_std < 8:
                findings.append(self._post_finding(
                    f"{team} maintains low speed variance (σ={spd_std:.1f} km/h) — consistent performance signature",
                    source="BOAT_SPEED_km_h_1", metric="speed_std", value=round(spd_std, 2),
                    strength=0.6, novelty=0.55,
                ))
        return findings[:12]  # cap per cycle


class AroundWaterAgent(StreamResearchAgent):
    stream_id = "around"
    stream_name = "Around the Water"

    def research_cycle(self) -> list[dict]:
        findings = []
        if self.df_racing is None:
            return findings

        # Narrative metrics: rank volatility, comeback rate
        for (venue, race_label), grp in self.df_racing.groupby(["venue", "race_label"]):
            pivots = 0
            for team, t in grp.groupby("team"):
                ranks = t["TRK_RACE_RANK_unk"].dropna()
                if len(ranks) < 20:
                    continue
                pivots += int((ranks.diff().abs() > 0).sum())
            if pivots > 40:
                findings.append(self._post_finding(
                    f"{venue} {race_label}: high rank volatility ({pivots} position changes) — fan narrative: lead changes",
                    source="TRK_RACE_RANK_unk", metric="rank_changes", value=pivots,
                    strength=min(0.85, pivots / 80), novelty=0.7,
                ))

        # Speed gaps for storytelling
        for (venue, race_label), grp in self.df_racing.groupby(["venue", "race_label"]):
            last = grp.sort_index().groupby("team").last()
            if "PC_DTL_m" not in last.columns or len(last) < 4:
                continue
            gap = last["PC_DTL_m"].max()
            if gap > 200:
                findings.append(self._post_finding(
                    f"{venue} {race_label}: final gap to leader {gap:.0f}m — quantifies broadcast 'distance behind'",
                    source="PC_DTL_m", metric="final_dtl_max", value=round(gap, 0),
                    strength=min(0.8, gap / 500), novelty=0.68,
                ))

        # Pre-start drama: speed variance last minute
        pre = self.df
        if pre is not None:
            pre = pre[pre["TRK_BOAT_RACE_STATUS_unk"] == 1]
            if len(pre) > 200:
                spd_range = pre.groupby(["venue", "race_label"])["GPS_SOG_km_h_1"].agg(lambda s: s.max() - s.min())
                for idx, val in spd_range.nlargest(3).items():
                    if val > 15:
                        findings.append(self._post_finding(
                            f"Pre-start speed swing {idx[0]} {idx[1]}: {val:.0f} km/h range — start-line tension metric",
                            source="prestart GPS_SOG", metric="prestart_speed_range",
                            value=round(val, 1), strength=0.65, novelty=0.74,
                        ))
        return findings


STREAM_AGENTS = {
    "above": AboveWaterAgent,
    "on_water": OnWaterAgent,
    "around": AroundWaterAgent,
}
