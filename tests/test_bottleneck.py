"""
Tests for bottleneck detection and optimization modules.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import generate_dataset, STAGE_CONFIG
from src.metrics import calculate_stage_metrics, calculate_utilization
from src.bottleneck_detector import detect_bottleneck, DEFAULT_WORKERS
from src.optimizer import WhatIfAnalyzer


class TestDataGenerator:
    """Tests for synthetic data generation."""

    def test_generates_correct_stages(self):
        """All five stages should be present in generated data."""
        df = generate_dataset(num_days=3, random_seed=42)
        stages = set(df["stage"].unique())
        expected = {"receiving", "picking", "packing", "sorting", "shipping"}
        assert stages == expected

    def test_generates_consistent_orders(self):
        """Same seed should produce same data."""
        df1 = generate_dataset(num_days=3, random_seed=42)
        df2 = generate_dataset(num_days=3, random_seed=42)
        assert len(df1) == len(df2)
        assert df1["order_id"].iloc[0] == df2["order_id"].iloc[0]

    def test_bottleneck_stage_has_higher_queue(self):
        """Artificially bottlenecked stage should show higher queue times."""
        df = generate_dataset(
            num_days=7,
            bottleneck_stage="packing",
            bottleneck_severity=1.8,
            random_seed=42
        )
        metrics = calculate_stage_metrics(df)

        packing_queue = metrics[metrics["stage"] == "packing"]["avg_queue_time"].iloc[0]
        other_queue = metrics[metrics["stage"] != "packing"]["avg_queue_time"].mean()

        assert packing_queue > other_queue, "Bottleneck stage should have higher queue time"


class TestBottleneckDetection:
    """Tests for bottleneck detection algorithm."""

    def test_detects_artificial_bottleneck(self):
        """Detector should identify the stage we artificially constrained."""
        df = generate_dataset(
            num_days=10,
            bottleneck_stage="packing",
            bottleneck_severity=1.6,
            random_seed=42
        )

        analysis = detect_bottleneck(df)
        assert analysis.bottleneck_stage == "packing"

    def test_returns_confidence_score(self):
        """Detection should include a confidence score between 0 and 1."""
        df = generate_dataset(num_days=7, random_seed=42)
        analysis = detect_bottleneck(df)

        assert 0 <= analysis.confidence <= 1

    def test_provides_recommendations(self):
        """Detection should include actionable recommendations."""
        df = generate_dataset(num_days=7, bottleneck_stage="picking", random_seed=42)
        analysis = detect_bottleneck(df)

        assert len(analysis.recommendations) > 0
        assert any("worker" in r.lower() or "picking" in r.lower()
                  for r in analysis.recommendations)


class TestMetrics:
    """Tests for metrics calculation."""

    def test_stage_metrics_structure(self):
        """Stage metrics should include required columns."""
        df = generate_dataset(num_days=3, random_seed=42)
        metrics = calculate_stage_metrics(df)

        required_cols = ["stage", "avg_cycle_time", "avg_queue_time", "throughput_per_hour"]
        for col in required_cols:
            assert col in metrics.columns

    def test_utilization_calculation(self):
        """Utilization should be between 0 and 100%."""
        df = generate_dataset(num_days=7, random_seed=42)
        utilization = calculate_utilization(df, DEFAULT_WORKERS)

        assert (utilization["utilization_pct"] >= 0).all()
        assert (utilization["utilization_pct"] <= 100).all()


class TestOptimizer:
    """Tests for what-if scenario analyzer."""

    def test_add_workers_improves_throughput(self):
        """Adding workers to bottleneck should improve throughput."""
        df = generate_dataset(
            num_days=10,
            bottleneck_stage="packing",
            bottleneck_severity=1.5,
            random_seed=42
        )

        analyzer = WhatIfAnalyzer(df)
        result = analyzer.simulate_add_workers("packing", 2)

        assert result.throughput_change_pct > 0

    def test_recommendations_are_ranked(self):
        """Recommendations should be ranked by improvement potential."""
        df = generate_dataset(num_days=10, random_seed=42)
        analyzer = WhatIfAnalyzer(df)
        recommendations = analyzer.generate_recommendations(5)

        improvements = [r.expected_improvement_pct for r in recommendations]
        assert improvements == sorted(improvements, reverse=True)

    def test_reallocation_from_same_stage_fails(self):
        """Cannot reallocate workers to the same stage."""
        df = generate_dataset(num_days=7, random_seed=42)
        analyzer = WhatIfAnalyzer(df)

        result = analyzer.simulate_worker_reallocation("packing", "packing", 1)
        # Should have minimal or zero change since it's invalid
        assert "ERROR" in result.implementation_notes or result.throughput_change_pct == 0


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline(self):
        """Test the complete analysis pipeline."""
        # Generate data
        df = generate_dataset(
            num_days=7,
            bottleneck_stage="sorting",
            bottleneck_severity=1.7,
            random_seed=123
        )

        # Detect bottleneck
        analysis = detect_bottleneck(df)
        assert analysis.bottleneck_stage is not None

        # Get recommendations
        analyzer = WhatIfAnalyzer(df)
        recommendations = analyzer.generate_recommendations(3)
        assert len(recommendations) > 0

        # Verify recommendation addresses bottleneck
        top_rec = recommendations[0]
        assert analysis.bottleneck_stage in top_rec.action.lower() or \
               top_rec.expected_improvement_pct > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
