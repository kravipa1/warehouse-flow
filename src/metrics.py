"""
Warehouse Metrics Calculator

Computes key operational metrics from event logs:
- Stage-level cycle times
- Queue lengths over time
- Utilization rates
- Throughput analysis
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import timedelta


def calculate_stage_metrics(event_log: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregate metrics for each warehouse stage.

    Returns:
        DataFrame with metrics per stage:
        - avg_cycle_time: Mean processing time
        - std_cycle_time: Variability in processing
        - avg_queue_time: Mean waiting time before processing
        - max_queue_time: Worst-case wait
        - throughput_per_hour: Orders processed per hour
        - total_orders: Count of orders processed
    """
    # Ensure datetime types
    event_log = event_log.copy()
    event_log["start_time"] = pd.to_datetime(event_log["start_time"])
    event_log["end_time"] = pd.to_datetime(event_log["end_time"])

    # Calculate time span
    total_hours = (
        event_log["end_time"].max() - event_log["start_time"].min()
    ).total_seconds() / 3600

    metrics = event_log.groupby("stage").agg(
        avg_cycle_time=("cycle_time_minutes", "mean"),
        std_cycle_time=("cycle_time_minutes", "std"),
        median_cycle_time=("cycle_time_minutes", "median"),
        p95_cycle_time=("cycle_time_minutes", lambda x: x.quantile(0.95)),
        avg_queue_time=("queue_time_minutes", "mean"),
        max_queue_time=("queue_time_minutes", "max"),
        total_orders=("order_id", "count")
    ).reset_index()

    metrics["throughput_per_hour"] = metrics["total_orders"] / total_hours

    # Add stage order for proper sorting
    stage_order = {"receiving": 1, "picking": 2, "packing": 3, "sorting": 4, "shipping": 5}
    metrics["stage_order"] = metrics["stage"].map(stage_order)
    metrics = metrics.sort_values("stage_order").drop(columns=["stage_order"])

    return metrics.round(3)


def calculate_hourly_metrics(
    event_log: pd.DataFrame,
    time_column: str = "start_time"
) -> pd.DataFrame:
    """
    Calculate metrics aggregated by hour for trend analysis.

    Returns hourly breakdown of:
    - Orders started per stage
    - Average cycle and queue times
    - Worker utilization estimates
    """
    df = event_log.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["hour"] = df[time_column].dt.floor("h")

    hourly = df.groupby(["hour", "stage"]).agg(
        order_count=("order_id", "count"),
        avg_cycle_time=("cycle_time_minutes", "mean"),
        avg_queue_time=("queue_time_minutes", "mean"),
        total_cycle_minutes=("cycle_time_minutes", "sum")
    ).reset_index()

    return hourly


def calculate_queue_over_time(
    event_log: pd.DataFrame,
    interval_minutes: int = 15
) -> pd.DataFrame:
    """
    Calculate queue length at each stage over time.

    Uses event-based calculation: queue increases when order arrives at stage,
    decreases when processing starts.
    """
    df = event_log.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # Create time index
    min_time = df["start_time"].min().floor("h")
    max_time = df["end_time"].max().ceil("h")
    time_index = pd.date_range(min_time, max_time, freq=f"{interval_minutes}min")

    results = []

    for stage in df["stage"].unique():
        stage_df = df[df["stage"] == stage]

        for t in time_index:
            t_next = t + timedelta(minutes=interval_minutes)

            # Orders in queue: arrived but not started processing
            # Approximation: orders that started within the interval
            in_progress = stage_df[
                (stage_df["start_time"] <= t) &
                (stage_df["end_time"] > t)
            ]

            # Orders waiting (queue time > 0 indicates waiting occurred)
            waiting = stage_df[
                (stage_df["start_time"] > t) &
                (stage_df["start_time"] <= t_next) &
                (stage_df["queue_time_minutes"] > 1)  # More than 1 min wait
            ]

            results.append({
                "timestamp": t,
                "stage": stage,
                "in_progress": len(in_progress),
                "queue_length": len(waiting),
                "avg_queue_time": waiting["queue_time_minutes"].mean() if len(waiting) > 0 else 0
            })

    return pd.DataFrame(results)


def calculate_utilization(
    event_log: pd.DataFrame,
    stage_workers: dict[str, int]
) -> pd.DataFrame:
    """
    Calculate worker utilization rate per stage.

    Utilization = (Total processing time) / (Available worker-hours)
    """
    df = event_log.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # Total time span
    total_hours = (df["end_time"].max() - df["start_time"].min()).total_seconds() / 3600

    utilization = []

    for stage in df["stage"].unique():
        stage_df = df[df["stage"] == stage]

        # Total processing minutes
        total_processing = stage_df["cycle_time_minutes"].sum()

        # Available capacity (workers * hours * 60 min/hour)
        workers = stage_workers.get(stage, 5)
        available_minutes = workers * total_hours * 60

        util_rate = (total_processing / available_minutes) * 100 if available_minutes > 0 else 0

        utilization.append({
            "stage": stage,
            "workers": workers,
            "total_processing_minutes": total_processing,
            "available_minutes": available_minutes,
            "utilization_pct": min(util_rate, 100),  # Cap at 100%
            "is_overloaded": util_rate > 85
        })

    return pd.DataFrame(utilization)


def calculate_flow_efficiency(event_log: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate flow efficiency per order.

    Flow efficiency = Value-added time / Total lead time
    Where value-added time = sum of cycle times
    And total lead time = end of shipping - start of receiving
    """
    df = event_log.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    order_metrics = df.groupby("order_id").agg(
        total_cycle_time=("cycle_time_minutes", "sum"),
        total_queue_time=("queue_time_minutes", "sum"),
        start=("start_time", "min"),
        end=("end_time", "max"),
        item_count=("item_count", "first"),
        priority=("priority", "first")
    ).reset_index()

    order_metrics["lead_time_minutes"] = (
        order_metrics["end"] - order_metrics["start"]
    ).dt.total_seconds() / 60

    order_metrics["flow_efficiency_pct"] = (
        order_metrics["total_cycle_time"] / order_metrics["lead_time_minutes"] * 100
    )

    return order_metrics


def get_peak_hour_analysis(event_log: pd.DataFrame) -> pd.DataFrame:
    """
    Identify peak hours and their impact on each stage.
    """
    df = event_log.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["hour_of_day"] = df["start_time"].dt.hour

    peak_analysis = df.groupby(["hour_of_day", "stage"]).agg(
        order_volume=("order_id", "count"),
        avg_queue_time=("queue_time_minutes", "mean"),
        avg_cycle_time=("cycle_time_minutes", "mean")
    ).reset_index()

    # Identify if hour is typically "peak"
    hourly_totals = df.groupby("hour_of_day")["order_id"].count()
    peak_threshold = hourly_totals.quantile(0.75)
    peak_hours = hourly_totals[hourly_totals >= peak_threshold].index.tolist()

    peak_analysis["is_peak_hour"] = peak_analysis["hour_of_day"].isin(peak_hours)

    return peak_analysis


if __name__ == "__main__":
    from data_generator import generate_dataset

    # Generate test data
    df = generate_dataset(num_days=7, bottleneck_stage="packing", random_seed=42)

    print("=== Stage Metrics ===")
    print(calculate_stage_metrics(df))

    print("\n=== Utilization ===")
    workers = {"receiving": 4, "picking": 10, "packing": 6, "sorting": 4, "shipping": 5}
    print(calculate_utilization(df, workers))

    print("\n=== Flow Efficiency Summary ===")
    flow = calculate_flow_efficiency(df)
    print(f"Average flow efficiency: {flow['flow_efficiency_pct'].mean():.1f}%")
    print(f"Average lead time: {flow['lead_time_minutes'].mean():.1f} minutes")
