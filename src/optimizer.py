"""
What-If Scenario Analyzer & Recommendation Engine

Simulates operational changes and quantifies their impact on throughput.
Examples:
- Adding workers to a stage
- Reducing cycle time through process improvement
- Shifting workers between stages during peak hours
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .metrics import calculate_stage_metrics, calculate_utilization, calculate_flow_efficiency
from .bottleneck_detector import detect_bottleneck, DEFAULT_WORKERS


@dataclass
class ScenarioResult:
    """Result of a what-if simulation."""
    scenario_name: str
    baseline_throughput: float
    projected_throughput: float
    throughput_change_pct: float
    baseline_avg_lead_time: float
    projected_avg_lead_time: float
    lead_time_change_pct: float
    baseline_bottleneck: str
    projected_bottleneck: str
    bottleneck_resolved: bool
    implementation_notes: str


@dataclass
class OptimizationRecommendation:
    """A ranked optimization recommendation."""
    rank: int
    action: str
    expected_improvement_pct: float
    implementation_effort: str  # "low", "medium", "high"
    reasoning: str
    scenario_result: ScenarioResult


class WhatIfAnalyzer:
    """
    Analyzes the impact of operational changes on warehouse throughput.

    Uses lightweight simulation/extrapolation rather than full discrete-event sim.
    """

    def __init__(self, event_log: pd.DataFrame, stage_workers: dict[str, int] = None):
        self.event_log = event_log.copy()
        self.event_log["start_time"] = pd.to_datetime(self.event_log["start_time"])
        self.event_log["end_time"] = pd.to_datetime(self.event_log["end_time"])
        self.stage_workers = stage_workers or DEFAULT_WORKERS.copy()

        # Calculate baseline metrics
        self.baseline_metrics = calculate_stage_metrics(event_log)
        self.baseline_utilization = calculate_utilization(event_log, self.stage_workers)
        self.baseline_flow = calculate_flow_efficiency(event_log)
        self.baseline_bottleneck = detect_bottleneck(event_log, self.stage_workers)

    def _calculate_throughput(self, event_log: pd.DataFrame) -> float:
        """Calculate orders per hour throughput."""
        df = event_log.copy()
        df["start_time"] = pd.to_datetime(df["start_time"])
        df["end_time"] = pd.to_datetime(df["end_time"])

        # Count completed orders (reached shipping)
        completed = df[df["stage"] == "shipping"]
        total_hours = (
            df["end_time"].max() - df["start_time"].min()
        ).total_seconds() / 3600

        return len(completed) / total_hours if total_hours > 0 else 0

    def simulate_add_workers(
        self,
        stage: str,
        additional_workers: int
    ) -> ScenarioResult:
        """
        Simulate adding workers to a stage.

        Approximation: More workers → proportionally reduced queue time,
        with diminishing returns after 80% utilization resolved.
        """
        current_workers = self.stage_workers.get(stage, 5)
        new_workers = current_workers + additional_workers

        # Get current stage metrics
        stage_util = self.baseline_utilization[
            self.baseline_utilization["stage"] == stage
        ].iloc[0]

        current_util = stage_util["utilization_pct"]

        # New utilization (assuming linear scaling)
        new_util = current_util * (current_workers / new_workers)

        # Estimate queue time reduction
        stage_metrics = self.baseline_metrics[
            self.baseline_metrics["stage"] == stage
        ].iloc[0]

        current_queue_time = stage_metrics["avg_queue_time"]

        # Queue time reduction model:
        # If over 85% util, queue grows exponentially
        # Adding workers brings util down, reducing queue
        if current_util > 85:
            queue_reduction_factor = (current_util - new_util) / current_util
        else:
            queue_reduction_factor = 0.3 * (additional_workers / current_workers)

        new_queue_time = current_queue_time * (1 - queue_reduction_factor)

        # Estimate throughput impact
        # Throughput limited by bottleneck capacity
        baseline_throughput = self._calculate_throughput(self.event_log)

        if stage == self.baseline_bottleneck.bottleneck_stage:
            # Relieving bottleneck has bigger impact
            capacity_increase = additional_workers / current_workers
            throughput_increase = min(capacity_increase * 0.8, 0.5)  # Cap at 50%
        else:
            throughput_increase = 0.05  # Minor impact on non-bottleneck

        projected_throughput = baseline_throughput * (1 + throughput_increase)

        # Lead time impact
        baseline_lead_time = self.baseline_flow["lead_time_minutes"].mean()
        lead_time_reduction = current_queue_time - new_queue_time
        projected_lead_time = baseline_lead_time - lead_time_reduction

        # Determine if bottleneck shifts
        if stage == self.baseline_bottleneck.bottleneck_stage and new_util < 75:
            # Find next bottleneck
            other_stages = self.baseline_utilization[
                self.baseline_utilization["stage"] != stage
            ]
            next_bottleneck = other_stages.loc[
                other_stages["utilization_pct"].idxmax()
            ]["stage"]
            bottleneck_resolved = True
        else:
            next_bottleneck = self.baseline_bottleneck.bottleneck_stage
            bottleneck_resolved = False

        return ScenarioResult(
            scenario_name=f"Add {additional_workers} worker(s) to {stage}",
            baseline_throughput=baseline_throughput,
            projected_throughput=projected_throughput,
            throughput_change_pct=(projected_throughput / baseline_throughput - 1) * 100,
            baseline_avg_lead_time=baseline_lead_time,
            projected_avg_lead_time=max(projected_lead_time, baseline_lead_time * 0.5),
            lead_time_change_pct=(projected_lead_time / baseline_lead_time - 1) * 100,
            baseline_bottleneck=self.baseline_bottleneck.bottleneck_stage,
            projected_bottleneck=next_bottleneck,
            bottleneck_resolved=bottleneck_resolved,
            implementation_notes=f"Requires hiring/training {additional_workers} workers for {stage}"
        )

    def simulate_cycle_time_reduction(
        self,
        stage: str,
        reduction_pct: float
    ) -> ScenarioResult:
        """
        Simulate process improvement reducing cycle time.

        Examples: Better tools, improved layout, automation assistance.
        """
        stage_metrics = self.baseline_metrics[
            self.baseline_metrics["stage"] == stage
        ].iloc[0]

        current_cycle = stage_metrics["avg_cycle_time"]
        new_cycle = current_cycle * (1 - reduction_pct / 100)

        # Utilization drops proportionally
        stage_util = self.baseline_utilization[
            self.baseline_utilization["stage"] == stage
        ].iloc[0]
        current_util = stage_util["utilization_pct"]
        new_util = current_util * (1 - reduction_pct / 100)

        # Throughput impact (cycle time reduction = capacity increase)
        baseline_throughput = self._calculate_throughput(self.event_log)

        if stage == self.baseline_bottleneck.bottleneck_stage:
            capacity_increase = reduction_pct / 100
            throughput_increase = min(capacity_increase * 0.7, 0.4)
        else:
            throughput_increase = 0.03

        projected_throughput = baseline_throughput * (1 + throughput_increase)

        # Lead time impact
        baseline_lead_time = self.baseline_flow["lead_time_minutes"].mean()
        # Cycle time reduction directly reduces lead time
        lead_time_reduction = current_cycle * (reduction_pct / 100)
        projected_lead_time = baseline_lead_time - lead_time_reduction

        # Bottleneck resolution
        bottleneck_resolved = (
            stage == self.baseline_bottleneck.bottleneck_stage and
            new_util < 75
        )

        return ScenarioResult(
            scenario_name=f"Reduce {stage} cycle time by {reduction_pct}%",
            baseline_throughput=baseline_throughput,
            projected_throughput=projected_throughput,
            throughput_change_pct=(projected_throughput / baseline_throughput - 1) * 100,
            baseline_avg_lead_time=baseline_lead_time,
            projected_avg_lead_time=max(projected_lead_time, baseline_lead_time * 0.5),
            lead_time_change_pct=(projected_lead_time / baseline_lead_time - 1) * 100,
            baseline_bottleneck=self.baseline_bottleneck.bottleneck_stage,
            projected_bottleneck=self.baseline_bottleneck.bottleneck_stage if not bottleneck_resolved else "balanced",
            bottleneck_resolved=bottleneck_resolved,
            implementation_notes=f"Process improvement, training, or equipment upgrade for {stage}"
        )

    def simulate_worker_reallocation(
        self,
        from_stage: str,
        to_stage: str,
        worker_count: int
    ) -> ScenarioResult:
        """
        Simulate moving workers from one stage to another.

        Zero-cost change (no hiring), but requires cross-training.
        """
        # Check feasibility
        from_workers = self.stage_workers.get(from_stage, 5)
        if from_stage == to_stage or worker_count >= from_workers:
            return ScenarioResult(
                scenario_name=f"Move {worker_count} workers from {from_stage} to {to_stage}",
                baseline_throughput=self._calculate_throughput(self.event_log),
                projected_throughput=self._calculate_throughput(self.event_log),
                throughput_change_pct=0,
                baseline_avg_lead_time=self.baseline_flow["lead_time_minutes"].mean(),
                projected_avg_lead_time=self.baseline_flow["lead_time_minutes"].mean(),
                lead_time_change_pct=0,
                baseline_bottleneck=self.baseline_bottleneck.bottleneck_stage,
                projected_bottleneck=self.baseline_bottleneck.bottleneck_stage,
                bottleneck_resolved=False,
                implementation_notes="ERROR: Cannot move that many workers"
            )

        # Simulate impact on both stages
        add_result = self.simulate_add_workers(to_stage, worker_count)

        # Account for reduced capacity at from_stage
        from_util = self.baseline_utilization[
            self.baseline_utilization["stage"] == from_stage
        ].iloc[0]["utilization_pct"]

        new_from_util = from_util * (from_workers / (from_workers - worker_count))

        # Adjust throughput if from_stage becomes bottleneck
        if new_from_util > 95:
            # This creates a new bottleneck
            throughput_penalty = (new_from_util - 85) / 100 * 0.5
            projected_throughput = add_result.projected_throughput * (1 - throughput_penalty)
            new_bottleneck = from_stage
        else:
            projected_throughput = add_result.projected_throughput
            new_bottleneck = add_result.projected_bottleneck

        baseline_throughput = self._calculate_throughput(self.event_log)

        return ScenarioResult(
            scenario_name=f"Move {worker_count} worker(s) from {from_stage} to {to_stage}",
            baseline_throughput=baseline_throughput,
            projected_throughput=projected_throughput,
            throughput_change_pct=(projected_throughput / baseline_throughput - 1) * 100,
            baseline_avg_lead_time=add_result.baseline_avg_lead_time,
            projected_avg_lead_time=add_result.projected_avg_lead_time,
            lead_time_change_pct=add_result.lead_time_change_pct,
            baseline_bottleneck=self.baseline_bottleneck.bottleneck_stage,
            projected_bottleneck=new_bottleneck,
            bottleneck_resolved=add_result.bottleneck_resolved and new_from_util < 85,
            implementation_notes=f"Requires cross-training {worker_count} workers. Zero hiring cost."
        )

    def generate_recommendations(
        self,
        max_recommendations: int = 5
    ) -> list[OptimizationRecommendation]:
        """
        Generate ranked optimization recommendations.

        Tests multiple scenarios and ranks by improvement potential.
        """
        bottleneck = self.baseline_bottleneck.bottleneck_stage
        scenarios = []

        # Scenario 1: Add workers to bottleneck
        for add_count in [1, 2, 3]:
            result = self.simulate_add_workers(bottleneck, add_count)
            scenarios.append((
                f"Add {add_count} worker(s) to {bottleneck}",
                result,
                "medium" if add_count <= 2 else "high",
                f"Directly addresses {bottleneck} capacity constraint"
            ))

        # Scenario 2: Cycle time improvements
        for reduction in [10, 15, 20]:
            result = self.simulate_cycle_time_reduction(bottleneck, reduction)
            scenarios.append((
                f"Reduce {bottleneck} cycle time by {reduction}%",
                result,
                "medium" if reduction <= 15 else "high",
                f"Process improvement at {bottleneck} increases capacity"
            ))

        # Scenario 3: Worker reallocation (find underutilized stages)
        underutilized = self.baseline_utilization[
            self.baseline_utilization["utilization_pct"] < 60
        ]["stage"].tolist()

        for stage in underutilized:
            if stage != bottleneck:
                result = self.simulate_worker_reallocation(stage, bottleneck, 1)
                scenarios.append((
                    f"Move 1 worker from {stage} to {bottleneck}",
                    result,
                    "low",
                    f"{stage} has spare capacity ({self.baseline_utilization[self.baseline_utilization['stage']==stage]['utilization_pct'].iloc[0]:.0f}% util)"
                ))

        # Rank by improvement percentage
        scenarios.sort(key=lambda x: x[1].throughput_change_pct, reverse=True)

        recommendations = []
        for i, (action, result, effort, reasoning) in enumerate(scenarios[:max_recommendations], 1):
            recommendations.append(OptimizationRecommendation(
                rank=i,
                action=action,
                expected_improvement_pct=result.throughput_change_pct,
                implementation_effort=effort,
                reasoning=reasoning,
                scenario_result=result
            ))

        return recommendations

    def compare_scenarios(
        self,
        scenarios: list[ScenarioResult]
    ) -> pd.DataFrame:
        """
        Create a comparison table of multiple scenarios.
        """
        data = []
        for s in scenarios:
            data.append({
                "Scenario": s.scenario_name,
                "Throughput Change": f"{s.throughput_change_pct:+.1f}%",
                "Lead Time Change": f"{s.lead_time_change_pct:+.1f}%",
                "Resolves Bottleneck": "Yes" if s.bottleneck_resolved else "No",
                "New Bottleneck": s.projected_bottleneck,
                "Notes": s.implementation_notes
            })

        return pd.DataFrame(data)


if __name__ == "__main__":
    from data_generator import generate_dataset

    print("Generating test data with packing bottleneck...")
    df = generate_dataset(
        num_days=14,
        bottleneck_stage="packing",
        bottleneck_severity=1.5,
        random_seed=42
    )

    print("\nInitializing What-If Analyzer...")
    analyzer = WhatIfAnalyzer(df)

    print(f"Current bottleneck: {analyzer.baseline_bottleneck.bottleneck_stage.upper()}")
    print(f"Current throughput: {analyzer._calculate_throughput(df):.1f} orders/hour")

    print("\n=== Scenario Analysis ===")

    # Test scenarios
    scenarios = [
        analyzer.simulate_add_workers("packing", 2),
        analyzer.simulate_cycle_time_reduction("packing", 15),
        analyzer.simulate_worker_reallocation("receiving", "packing", 1)
    ]

    for s in scenarios:
        print(f"\n{s.scenario_name}")
        print(f"  Throughput: {s.throughput_change_pct:+.1f}%")
        print(f"  Lead Time: {s.lead_time_change_pct:+.1f}%")
        print(f"  Resolves Bottleneck: {'Yes' if s.bottleneck_resolved else 'No'}")

    print("\n=== Top Recommendations ===")
    recommendations = analyzer.generate_recommendations(5)

    for rec in recommendations:
        print(f"\n{rec.rank}. {rec.action}")
        print(f"   Expected Improvement: {rec.expected_improvement_pct:+.1f}%")
        print(f"   Effort: {rec.implementation_effort}")
        print(f"   Reasoning: {rec.reasoning}")
