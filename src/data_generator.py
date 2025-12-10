"""
Synthetic Warehouse Event Log Generator

Generates realistic warehouse operation data with configurable bottlenecks.
Each order flows through: Receiving → Picking → Packing → Sorting → Shipping
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


# Warehouse stage configuration with realistic parameters
STAGE_CONFIG = {
    "receiving": {
        "base_cycle_time": 3.0,      # minutes
        "cycle_time_std": 1.0,
        "workers": 4,
        "order": 1
    },
    "picking": {
        "base_cycle_time": 8.0,      # minutes - depends on items
        "cycle_time_std": 3.0,
        "workers": 10,
        "order": 2
    },
    "packing": {
        "base_cycle_time": 5.0,      # minutes
        "cycle_time_std": 1.5,
        "workers": 6,
        "order": 3
    },
    "sorting": {
        "base_cycle_time": 2.0,      # minutes
        "cycle_time_std": 0.5,
        "workers": 4,
        "order": 4
    },
    "shipping": {
        "base_cycle_time": 4.0,      # minutes
        "cycle_time_std": 1.0,
        "workers": 5,
        "order": 5
    }
}


def generate_order_arrival_times(
    start_date: datetime,
    num_days: int,
    base_orders_per_hour: float = 30,
    peak_multiplier: float = 2.5,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate realistic order arrival patterns with daily/hourly variations.

    Peak hours: 9-11 AM and 2-4 PM (typical warehouse patterns)
    Weekend reduction: 60% of weekday volume
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    orders = []
    order_id = 1000

    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        is_weekend = current_date.weekday() >= 5

        for hour in range(6, 22):  # Warehouse operates 6 AM to 10 PM
            # Determine hourly demand multiplier
            if hour in [9, 10, 11, 14, 15, 16]:
                hour_mult = peak_multiplier
            elif hour in [6, 7, 20, 21]:
                hour_mult = 0.5
            else:
                hour_mult = 1.0

            # Weekend adjustment
            day_mult = 0.6 if is_weekend else 1.0

            # Calculate orders for this hour
            expected_orders = base_orders_per_hour * hour_mult * day_mult
            actual_orders = np.random.poisson(expected_orders)

            # Generate arrival times within the hour
            for _ in range(actual_orders):
                minute = np.random.randint(0, 60)
                second = np.random.randint(0, 60)

                arrival_time = current_date.replace(
                    hour=hour, minute=minute, second=second, microsecond=0
                )

                # Random item count (affects processing time)
                item_count = max(1, int(np.random.lognormal(mean=1.5, sigma=0.8)))

                orders.append({
                    "order_id": f"ORD-{order_id:06d}",
                    "arrival_time": arrival_time,
                    "item_count": min(item_count, 50),  # Cap at 50 items
                    "priority": np.random.choice(
                        ["standard", "express", "priority"],
                        p=[0.7, 0.2, 0.1]
                    )
                })
                order_id += 1

    return pd.DataFrame(orders)


def simulate_warehouse_flow(
    orders_df: pd.DataFrame,
    stage_config: dict = STAGE_CONFIG,
    bottleneck_stage: Optional[str] = None,
    bottleneck_severity: float = 1.5,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate order flow through warehouse stages.

    Args:
        orders_df: DataFrame with order arrivals
        stage_config: Configuration for each stage
        bottleneck_stage: Stage to artificially slow down (for testing)
        bottleneck_severity: Multiplier for bottleneck stage cycle time
        random_seed: For reproducibility

    Returns:
        Event log with start/end times for each stage
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Sort stages by order
    stages = sorted(stage_config.keys(), key=lambda x: stage_config[x]["order"])

    # Track when each worker becomes available at each stage
    worker_availability = {
        stage: [datetime.min] * config["workers"]
        for stage, config in stage_config.items()
    }

    events = []

    # Process each order
    for _, order in orders_df.sort_values("arrival_time").iterrows():
        order_ready_time = order["arrival_time"]

        for stage in stages:
            config = stage_config[stage]

            # Find earliest available worker
            earliest_worker_idx = np.argmin(worker_availability[stage])
            earliest_available = worker_availability[stage][earliest_worker_idx]

            # Start time is max of (order ready, worker available)
            start_time = max(order_ready_time, earliest_available)

            # Calculate cycle time
            base_time = config["base_cycle_time"]
            std_time = config["cycle_time_std"]

            # Item count affects picking and packing time
            if stage in ["picking", "packing"]:
                item_multiplier = 1 + (order["item_count"] - 1) * 0.15
            else:
                item_multiplier = 1.0

            # Apply bottleneck if specified
            if stage == bottleneck_stage:
                base_time *= bottleneck_severity

            # Priority orders are 20% faster
            if order["priority"] == "priority":
                base_time *= 0.8

            # Generate actual cycle time (log-normal for realistic skew)
            cycle_time = max(0.5, np.random.lognormal(
                mean=np.log(base_time * item_multiplier),
                sigma=std_time / base_time
            ))

            end_time = start_time + timedelta(minutes=cycle_time)

            # Record event
            events.append({
                "order_id": order["order_id"],
                "stage": stage,
                "start_time": start_time,
                "end_time": end_time,
                "cycle_time_minutes": cycle_time,
                "worker_id": f"{stage.upper()}-W{earliest_worker_idx + 1:02d}",
                "item_count": order["item_count"],
                "priority": order["priority"],
                "queue_time_minutes": (start_time - order_ready_time).total_seconds() / 60
            })

            # Update worker availability
            worker_availability[stage][earliest_worker_idx] = end_time

            # Order is ready for next stage when this stage completes
            order_ready_time = end_time

    return pd.DataFrame(events)


def generate_dataset(
    start_date: str = "2024-01-01",
    num_days: int = 30,
    base_orders_per_hour: float = 25,
    bottleneck_stage: Optional[str] = "packing",
    bottleneck_severity: float = 1.4,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate a complete warehouse dataset.

    Default configuration creates a realistic packing bottleneck scenario.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")

    # Generate order arrivals
    orders = generate_order_arrival_times(
        start_date=start,
        num_days=num_days,
        base_orders_per_hour=base_orders_per_hour,
        random_seed=random_seed
    )

    # Simulate flow through warehouse
    event_log = simulate_warehouse_flow(
        orders_df=orders,
        bottleneck_stage=bottleneck_stage,
        bottleneck_severity=bottleneck_severity,
        random_seed=random_seed
    )

    return event_log


def generate_multi_scenario_dataset(
    random_seed: int = 42
) -> dict[str, pd.DataFrame]:
    """
    Generate multiple datasets with different bottleneck scenarios.
    Useful for training ML models.
    """
    scenarios = {
        "balanced": {"bottleneck_stage": None, "bottleneck_severity": 1.0},
        "picking_bottleneck": {"bottleneck_stage": "picking", "bottleneck_severity": 1.5},
        "packing_bottleneck": {"bottleneck_stage": "packing", "bottleneck_severity": 1.6},
        "sorting_bottleneck": {"bottleneck_stage": "sorting", "bottleneck_severity": 2.0},
        "shipping_bottleneck": {"bottleneck_stage": "shipping", "bottleneck_severity": 1.7},
    }

    datasets = {}
    for name, params in scenarios.items():
        datasets[name] = generate_dataset(
            num_days=14,
            random_seed=random_seed,
            **params
        )
        datasets[name]["scenario"] = name

    return datasets


if __name__ == "__main__":
    # Generate sample dataset
    df = generate_dataset(num_days=7, random_seed=42)
    print(f"Generated {len(df)} events")
    print(f"\nSample events:")
    print(df.head(10))
    print(f"\nStage summary:")
    print(df.groupby("stage").agg({
        "cycle_time_minutes": ["mean", "std"],
        "queue_time_minutes": ["mean", "max"]
    }).round(2))
