#!/usr/bin/env python3
"""
predict_graph_groups.py

Job-level runtime prediction using:
  (1) a transport-aware stage-duration predictor (calibrated stage model), and
  (2) dependency-aware aggregation using precomputed stage-dependency groups.

For a given job, this module:
  • predicts hybrid stage durations from on-prem stage baselines + target network regime,
  • aggregates stage predictions with serialized dependency-aware groups,
  • reports two reference aggregations:
      - cumulative stage summation (transport-aware, dependency-agnostic),
      - transport-agnostic baseline cumulative sum (+ scheduler overhead from the group graph).

Conventions aligned with the paper:
  • T_base        : on-prem observed stage duration (baseline)
  • T_hat_hybrid  : predicted hybrid stage duration (transport-aware stage model)
  • num_tasks     : number of tasks in the stage

Notes:
  • Scheduler overhead is read from the group JSON and added to job-level predictions.
  • Group graphs are expected under:
        data/graph/<fileformat>_<compression>/<query_id>_groups.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import transport_aware_model.model.stage_model as stage_model


# =============================================================================
# Graph I/O
# =============================================================================
def _get_one_hot_suffix(row: dict[str, Any], prefix: str) -> str | None:
    for k, v in row.items():
        if k.startswith(prefix) and v is not None and int(v) == 1:
            return k[len(prefix) :]
    return None


def get_graph_subdir(job: dict[str, Any]) -> str:
    compression = _get_one_hot_suffix(job, "compression_codec_")
    fileformat = _get_one_hot_suffix(job, "file_format_")
    if compression is None or fileformat is None:
        raise ValueError("Job is missing compression_codec_* or file_format_* one-hot fields")
    return f"{fileformat}_{compression}"


def load_group_graph(graph_path: str | Path) -> tuple[list[dict[str, Any]], list[tuple[int, int]], dict[int, int], float]:
    with open(graph_path, "r") as f:
        g = json.load(f)

    groups: list[dict[str, Any]] = g["groups"]
    group_edges: list[tuple[int, int]] = [tuple(e) for e in g.get("group_edges", [])]
    stage_to_group: dict[int, int] = {int(k): int(v) for k, v in g["stage_to_group"].items()}
    scheduler_overhead_s: float = float(g.get("scheduler_overhead_s", 0.0))

    return groups, group_edges, stage_to_group, scheduler_overhead_s


# =============================================================================
# Dependency-aware aggregation (group model)
# =============================================================================
def topo_groups(groups: list[dict[str, Any]], group_edges: list[tuple[int, int]]) -> tuple[list[int], dict[int, list[int]]]:
    indeg: dict[int, int] = defaultdict(int)
    succ: dict[int, list[int]] = defaultdict(list)

    gids = [int(g["gid"]) for g in groups]
    for u, v in group_edges:
        indeg[v] += 1
        indeg.setdefault(u, 0)
        succ[u].append(v)

    q = deque([g for g in gids if indeg[g] == 0])
    order: list[int] = []
    while q:
        x = q.popleft()
        order.append(x)
        for y in succ[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                q.append(y)

    if not group_edges:
        return sorted(gids), succ

    return order, succ


def resource_filling_time(members: list[int], feat: dict[int, dict[str, float]], E: int) -> float:
    """
    Resource-filling approximation for a parallel group:
      • approximate per-stage wave duration by T_hat_hybrid / ceil(num_tasks / E)
      • create a task-duration multiset and "pack" into waves of size E (descending).
    """
    task_durations: list[float] = []
    for sid in members:
        num_tasks = int(feat[sid]["num_tasks"])
        waves = max(1, math.ceil(num_tasks / E))
        tau = float(feat[sid]["T_hat_hybrid"]) / waves
        task_durations.extend([float(tau)] * num_tasks)

    n = len(task_durations)
    task_durations.sort(reverse=True)  # descending

    if n <= E:
        return float(task_durations[0])

    t = 0.0
    for i in range(0, n, E):
        t += float(task_durations[i])
    return float(t)


def group_time(group: dict[str, Any], feat: dict[int, dict[str, float]], E: int) -> float:
    members: list[int] = [int(x) for x in group["members"]]

    if len(members) == 1:
        return float(feat[members[0]]["T_hat_hybrid"])

    n_tasks = sum(int(feat[sid]["num_tasks"]) for sid in members)
    if n_tasks <= E:
        return float(max(float(feat[sid]["T_hat_hybrid"]) for sid in members))

    return resource_filling_time(members, feat, E)


def predict_job_time_sequential(
    groups: list[dict[str, Any]],
    group_edges: list[tuple[int, int]],
    feat: dict[int, dict[str, float]],
    E: int,
) -> float:
    """
    Serialize groups in topological order (as used in the paper's aggregation),
    and sum predicted group runtimes.
    """
    order, _ = topo_groups(groups, group_edges)
    t_group = {int(g["gid"]): group_time(g, feat, E) for g in groups}

    t = 0.0
    for gid in order:
        t += float(t_group[gid])

    return float(t)


# =============================================================================
# Public API used by transport_aware_model.predict_job
# =============================================================================
def predict_single_job(
    job: dict[str, Any],
    onprem_stages: list[dict[str, Any]],
    graph_root: Path = Path("data/graph"),
) -> tuple[float, float, float, float]:
    """
    Predict a single job runtime.

    Returns:
        observed_job_s,
        transport_aware_with_dependency_grouping_s,
        transport_aware_cumulative_sum_s,
        transport_agnostic_baseline_cumulative_sum_s
    """
    query_id = job.get("query_id")
    if query_id is None:
        raise ValueError("Job is missing query_id")

    # --- 1) Transport-aware stage predictions (T_hat_hybrid) from on-prem baselines (T_base) ---
    # Keep the stage list as a simple list, and also build a dict keyed by stage_id for grouping.
    stage_rows: list[dict[str, float]] = []
    feat_by_stage_id: dict[int, dict[str, float]] = {}

    for s in onprem_stages:
        if s.get("query_id") != query_id:
            raise ValueError(f"Query mismatch: job={query_id} vs stage={s.get('query_id')}")

        stage_id = int(s.get("stage_id"))
        T_base = float(s.get("stage_duration_s"))
        num_tasks = int(s.get("num_tasks"))

        T_hat_hybrid, _ = stage_model.predict_stage_duration(
            T_base=T_base,
            N_tasks=num_tasks,
            N_partitions=int(s.get("number_of_partitions_read")),
            input_read_gb=float(s.get("input_read_gb")),
            window_max_bytes=int(s.get("window_max_bytes")),
            mtu_bytes=int(s.get("mtu")),
            spark_num_executors=int(s.get("spark_num_executors")),
            cores_per_executor=int(s.get("spark_executor_cores")),
            network_latency_ms=float(job.get("network_latency_ms")),
            B_link=float(job.get("tsg_max_bw_gbs")),
        )

        row = {
            "stage_id": float(stage_id),
            "T_hat_hybrid": float(T_hat_hybrid),
            "T_base": float(T_base),
            "num_tasks": float(num_tasks),
        }
        stage_rows.append(row)
        feat_by_stage_id[stage_id] = {
            "T_hat_hybrid": float(T_hat_hybrid),
            "T_base": float(T_base),
            "num_tasks": float(num_tasks),
        }

    # --- 2) Dependency-aware aggregation (group model) + scheduler overhead from graph ---
    graph_subdir = get_graph_subdir(job)
    graph_path = graph_root / graph_subdir / f"{query_id}_groups.json"
    groups, group_edges, _, scheduler_overhead_s = load_group_graph(graph_path)

    # Executor core budget (E)
    e1 = job.get("spark_num_executors")
    e2 = job.get("spark_executor_cores")
    if e1 is not None and e2 is not None:
        E = int(e1) * int(e2)
    else:
        E = int(onprem_stages[0].get("spark_num_executors")) * int(onprem_stages[0].get("spark_executor_cores"))

    # Validate grouping references exist in stage features
    missing = [sid for g in groups for sid in g["members"] if int(sid) not in feat_by_stage_id]
    if missing:
        raise KeyError(f"Missing stage features for stage ids referenced by the group graph: {sorted(set(map(int, missing)))}")

    t_groups = predict_job_time_sequential(groups, group_edges, feat_by_stage_id, E)
    transport_aware_with_dependency_grouping_s = float(t_groups + scheduler_overhead_s)

    # --- 3) Reference aggregations ---
    transport_aware_cumulative_sum_s = float(sum(r["T_hat_hybrid"] for r in stage_rows) + scheduler_overhead_s)
    transport_agnostic_baseline_cumulative_sum_s = float(sum(r["T_base"] for r in stage_rows) + scheduler_overhead_s)

    observed_job_s = float(job.get("exec_time_s"))
    return (
        observed_job_s,
        transport_aware_with_dependency_grouping_s,
        transport_aware_cumulative_sum_s,
        transport_agnostic_baseline_cumulative_sum_s,
    )
