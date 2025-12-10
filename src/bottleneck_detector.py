"""
Bottleneck Detection Engine

Identifies the current operational bottleneck using multiple signals:
1. Queue accumulation (orders waiting)
2. High utilization (>85%)
3. Cycle time degradation
4. Upstream starvation / downstream blocking
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .metrics import (
    calculate_stage_metrics,
    calculate_utilization,
    calculate_hourly_metrics,
    calculate_flow_efficiency
)


@dataclass
class BottleneckAnalysis:
    """Results of bottleneck detection."""
    bottleneck_stage: str
    confidence: float  # 0-1 score
    primary_reason: str
    contributing_factors: list[str]
    metrics: dict
    recommendations: list[str]


# Default worker counts per stage
DEFAULT_WORKERS = {
    "receiving": 4,
    "picking": 10,
    "packing": 6,
    "sorting": 4,
    "shipping": 5
}


def detect_bottleneck(
    event_log: pd.DataFrame,
    stage_workers: dict[str, int] = DEFAULT_WORKERS
) -> BottleneckAnalysis:
    """
    Identify the primary bottleneck in the warehouse.

    Uses a scoring system based on:
    - Queue time score (40%): High queue times indicate backup
    - Utilization score (30%): High utilization means near capacity
    - Cycle time variability (15%): Inconsistency signals problems
    - Downstream impact (15%): Does this stage starve downstream?
    """
    stage_metrics = calculate_stage_metrics(event_log)
    utilization = calculate_utilization(event_log, stage_workers)

    # Merge metrics
    combined = stage_metrics.merge(utilization, on="stage")

    # Calculate bottleneck scores
    scores = []

    for _, row in combined.iterrows():
        stage = row["stage"]

        # Queue time score (normalized)
        queue_score = min(row["avg_queue_time"] / 10, 1.0) * 40

        # Utilization score
        util_score = (row["utilization_pct"] / 100) * 30

        # Cycle time variability (coefficient of variation)
        cv = row["std_cycle_time"] / row["avg_cycle_time"] if row["avg_cycle_time"] > 0 else 0
        variability_score = min(cv, 1.0) * 15

        # P95 vs median ratio (indicates tail issues)
        tail_ratio = row["p95_cycle_time"] / row["median_cycle_time"] if row["median_cycle_time"] > 0 else 1
        tail_score = min((tail_ratio - 1) / 2, 1.0) * 15

        total_score = queue_score + util_score + variability_score + tail_score

        scores.append({
            "stage": stage,
            "total_score": total_score,
            "queue_score": queue_score,
            "util_score": util_score,
            "variability_score": variability_score,
            "tail_score": tail_score,
            "avg_queue_time": row["avg_queue_time"],
            "utilization_pct": row["utilization_pct"],
            "avg_cycle_time": row["avg_cycle_time"]
        })

    scores_df = pd.DataFrame(scores)

    # Identify bottleneck
    bottleneck_row = scores_df.loc[scores_df["total_score"].idxmax()]
    bottleneck_stage = bottleneck_row["stage"]

    # Calculate confidence (how much higher than second place)
    sorted_scores = scores_df.sort_values("total_score", ascending=False)
    if len(sorted_scores) > 1:
        score_gap = sorted_scores.iloc[0]["total_score"] - sorted_scores.iloc[1]["total_score"]
        confidence = min(0.5 + (score_gap / 50), 0.95)
    else:
        confidence = 0.9

    # Determine primary reason
    component_scores = {
        "queue_score": bottleneck_row["queue_score"],
        "util_score": bottleneck_row["util_score"],
        "variability_score": bottleneck_row["variability_score"],
        "tail_score": bottleneck_row["tail_score"]
    }
    primary_component = max(component_scores, key=component_scores.get)

    reason_map = {
        "queue_score": f"High queue buildup (avg {bottleneck_row['avg_queue_time']:.1f} min wait)",
        "util_score": f"Near capacity ({bottleneck_row['utilization_pct']:.1f}% utilization)",
        "variability_score": "Inconsistent processing times",
        "tail_score": "Frequent long-tail delays"
    }
    primary_reason = reason_map[primary_component]

    # Contributing factors
    contributing = []
    if bottleneck_row["utilization_pct"] > 80:
        contributing.append(f"Utilization at {bottleneck_row['utilization_pct']:.1f}%")
    if bottleneck_row["avg_queue_time"] > 5:
        contributing.append(f"Average queue time of {bottleneck_row['avg_queue_time']:.1f} minutes")
    if bottleneck_row["avg_cycle_time"] > 7:
        contributing.append(f"Long average cycle time ({bottleneck_row['avg_cycle_time']:.1f} min)")

    # Generate recommendations
    recommendations = _generate_recommendations(
        bottleneck_stage,
        bottleneck_row,
        stage_workers
    )

    return BottleneckAnalysis(
        bottleneck_stage=bottleneck_stage,
        confidence=confidence,
        primary_reason=primary_reason,
        contributing_factors=contributing,
        metrics={
            "all_scores": scores_df.to_dict("records"),
            "bottleneck_score": bottleneck_row["total_score"],
            "stage_metrics": stage_metrics.to_dict("records"),
            "utilization": utilization.to_dict("records")
        },
        recommendations=recommendations
    )


def _generate_recommendations(
    stage: str,
    metrics: pd.Series,
    workers: dict[str, int]
) -> list[str]:
    """Generate actionable recommendations for addressing the bottleneck."""
    recs = []
    current_workers = workers.get(stage, 5)

    # High utilization → add workers
    if metrics["utilization_pct"] > 85:
        additional = max(1, int(current_workers * 0.25))
        recs.append(
            f"Add {additional} worker(s) to {stage} to reduce utilization below 80%"
        )

    # High queue time → process reengineering or workers
    if metrics["avg_queue_time"] > 5:
        recs.append(
            f"Investigate {stage} process for efficiency improvements"
        )

    # High variability → standardization
    if metrics["variability_score"] > 8:
        recs.append(
            f"Standardize {stage} procedures to reduce processing time variability"
        )

    # Specific stage recommendations
    if stage == "picking":
        recs.append("Consider zone-based picking or batch picking to reduce travel time")
    elif stage == "packing":
        recs.append("Pre-stage packing materials and optimize workstation layout")
    elif stage == "sorting":
        recs.append("Evaluate automated sorting options for high-volume periods")

    # Always suggest load balancing during peaks
    recs.append(f"Cross-train workers to shift to {stage} during peak hours")

    return recs[:5]  # Limit to top 5 recommendations


def analyze_bottleneck_shifts(
    event_log: pd.DataFrame,
    stage_workers: dict[str, int] = DEFAULT_WORKERS
) -> pd.DataFrame:
    """
    Analyze how the bottleneck shifts throughout the day.

    Returns hourly bottleneck identification.
    """
    df = event_log.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["hour"] = df["start_time"].dt.hour

    results = []

    for hour in sorted(df["hour"].unique()):
        hour_data = df[df["hour"] == hour]

        if len(hour_data) < 10:  # Skip hours with little data
            continue

        try:
            analysis = detect_bottleneck(hour_data, stage_workers)
            results.append({
                "hour": hour,
                "bottleneck_stage": analysis.bottleneck_stage,
                "confidence": analysis.confidence,
                "primary_reason": analysis.primary_reason
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def calculate_bottleneck_cost(
    event_log: pd.DataFrame,
    hourly_cost_per_delay_minute: float = 0.5
) -> dict:
    """
    Estimate the cost impact of the bottleneck.

    Uses queue time as a proxy for delay costs.
    """
    flow = calculate_flow_efficiency(event_log)

    # Total queue time across all orders
    total_queue_minutes = flow["total_queue_time"].sum()

    # Estimated cost
    delay_cost = total_queue_minutes * hourly_cost_per_delay_minute

    # Throughput loss estimate (orders that could have been processed)
    avg_cycle = flow["total_cycle_time"].mean()
    potential_additional_orders = int(total_queue_minutes / avg_cycle)

    return {
        "total_queue_minutes": total_queue_minutes,
        "estimated_delay_cost": delay_cost,
        "potential_additional_orders": potential_additional_orders,
        "avg_lead_time_minutes": flow["lead_time_minutes"].mean(),
        "avg_flow_efficiency_pct": flow["flow_efficiency_pct"].mean()
    }


if __name__ == "__main__":
    from data_generator import generate_dataset

    # Generate test data with packing bottleneck
    print("Generating test data with packing bottleneck...")
    df = generate_dataset(
        num_days=7,
        bottleneck_stage="packing",
        bottleneck_severity=1.5,
        random_seed=42
    )

    print("\n=== Bottleneck Detection ===")
    analysis = detect_bottleneck(df)

    print(f"\nBottleneck Stage: {analysis.bottleneck_stage.upper()}")
    print(f"Confidence: {analysis.confidence:.1%}")
    print(f"Primary Reason: {analysis.primary_reason}")

    print("\nContributing Factors:")
    for factor in analysis.contributing_factors:
        print(f"  • {factor}")

    print("\nRecommendations:")
    for i, rec in enumerate(analysis.recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n=== Bottleneck Cost Analysis ===")
    cost = calculate_bottleneck_cost(df)
    print(f"Total queue time: {cost['total_queue_minutes']:,.0f} minutes")
    print(f"Estimated delay cost: ${cost['estimated_delay_cost']:,.2f}")
    print(f"Potential additional orders: {cost['potential_additional_orders']:,}")
