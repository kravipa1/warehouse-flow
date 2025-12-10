"""
Warehouse Flow Optimizer - Interactive Dashboard

A decision-support tool for warehouse managers to:
1. Identify current bottlenecks
2. Predict future constraints
3. Simulate operational changes
4. Get actionable recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Import our modules
from src.data_generator import generate_dataset, STAGE_CONFIG
from src.metrics import (
    calculate_stage_metrics,
    calculate_utilization,
    calculate_hourly_metrics,
    calculate_flow_efficiency,
    get_peak_hour_analysis
)
from src.bottleneck_detector import detect_bottleneck, calculate_bottleneck_cost, DEFAULT_WORKERS
from src.predictor import BottleneckPredictor
from src.optimizer import WhatIfAnalyzer

# Page config
st.set_page_config(
    page_title="Warehouse Flow Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .bottleneck-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 5px;
    }
    .recommendation-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(num_days: int, bottleneck_stage: str, severity: float, seed: int):
    """Generate and cache synthetic data."""
    return generate_dataset(
        num_days=num_days,
        bottleneck_stage=bottleneck_stage if bottleneck_stage != "None" else None,
        bottleneck_severity=severity,
        random_seed=seed
    )


@st.cache_resource
def train_predictor(_data: pd.DataFrame):
    """Train and cache the prediction model."""
    predictor = BottleneckPredictor(model_type="random_forest")
    metrics = predictor.train(_data, lookahead_hours=1)
    return predictor, metrics


def main():
    st.title("📦 Warehouse Flow Optimizer")
    st.markdown("*Bottleneck Detection & Throughput Optimization*")

    # Sidebar - Data Configuration
    st.sidebar.header("📊 Data Configuration")

    with st.sidebar.expander("Generate Synthetic Data", expanded=True):
        num_days = st.slider("Days of data", 7, 30, 14)
        bottleneck_stage = st.selectbox(
            "Simulate bottleneck at",
            ["None", "picking", "packing", "sorting", "shipping"]
        )
        severity = st.slider("Bottleneck severity", 1.0, 2.0, 1.4, 0.1)
        seed = st.number_input("Random seed", value=42, step=1)

        if st.button("Generate Data", type="primary"):
            st.session_state["data_generated"] = True

    # Generate data
    df = load_data(num_days, bottleneck_stage, severity, seed)

    # Sidebar - Worker Configuration
    st.sidebar.header("👷 Worker Configuration")
    workers = {}
    for stage in ["receiving", "picking", "packing", "sorting", "shipping"]:
        workers[stage] = st.sidebar.number_input(
            f"{stage.capitalize()} workers",
            min_value=1,
            max_value=20,
            value=DEFAULT_WORKERS[stage]
        )

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Current Status",
        "🔮 Predictions",
        "🧪 What-If Analysis",
        "📋 Recommendations"
    ])

    # Tab 1: Current Status
    with tab1:
        st.header("Current Operational Status")

        # Detect bottleneck
        analysis = detect_bottleneck(df, workers)
        cost_analysis = calculate_bottleneck_cost(df)

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_orders = df["order_id"].nunique()
            st.metric("Total Orders", f"{total_orders:,}")

        with col2:
            throughput = total_orders / (num_days * 16)  # 16 operating hours
            st.metric("Throughput", f"{throughput:.1f} orders/hr")

        with col3:
            flow = calculate_flow_efficiency(df)
            avg_lead = flow["lead_time_minutes"].mean()
            st.metric("Avg Lead Time", f"{avg_lead:.0f} min")

        with col4:
            flow_eff = flow["flow_efficiency_pct"].mean()
            st.metric("Flow Efficiency", f"{flow_eff:.1f}%")

        st.markdown("---")

        # Bottleneck Alert
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🚨 Bottleneck Detection")
            st.markdown(f"""
            <div class="bottleneck-alert">
                <h3>Current Bottleneck: {analysis.bottleneck_stage.upper()}</h3>
                <p><strong>Confidence:</strong> {analysis.confidence:.1%}</p>
                <p><strong>Primary Reason:</strong> {analysis.primary_reason}</p>
            </div>
            """, unsafe_allow_html=True)

            if analysis.contributing_factors:
                st.markdown("**Contributing Factors:**")
                for factor in analysis.contributing_factors:
                    st.markdown(f"- {factor}")

        with col2:
            st.subheader("💰 Cost Impact")
            st.metric(
                "Total Queue Time",
                f"{cost_analysis['total_queue_minutes']:,.0f} min"
            )
            st.metric(
                "Est. Delay Cost",
                f"${cost_analysis['estimated_delay_cost']:,.0f}"
            )
            st.metric(
                "Potential Extra Orders",
                f"{cost_analysis['potential_additional_orders']:,}"
            )

        st.markdown("---")

        # Stage Metrics Visualization
        st.subheader("Stage Performance Metrics")

        metrics = calculate_stage_metrics(df)
        utilization = calculate_utilization(df, workers)

        col1, col2 = st.columns(2)

        with col1:
            # Cycle Time Chart
            fig = px.bar(
                metrics,
                x="stage",
                y="avg_cycle_time",
                error_y="std_cycle_time",
                title="Average Cycle Time by Stage",
                labels={"avg_cycle_time": "Minutes", "stage": "Stage"},
                color="stage",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Utilization Gauge
            fig = go.Figure()

            for i, row in utilization.iterrows():
                color = "red" if row["utilization_pct"] > 85 else "orange" if row["utilization_pct"] > 70 else "green"
                fig.add_trace(go.Bar(
                    name=row["stage"],
                    x=[row["stage"]],
                    y=[row["utilization_pct"]],
                    marker_color=color,
                    text=f"{row['utilization_pct']:.0f}%",
                    textposition="outside"
                ))

            fig.update_layout(
                title="Worker Utilization by Stage",
                yaxis_title="Utilization %",
                yaxis_range=[0, 110],
                showlegend=False
            )
            fig.add_hline(y=85, line_dash="dash", line_color="red",
                         annotation_text="Overload threshold")
            st.plotly_chart(fig, use_container_width=True)

        # Queue Time Heatmap
        st.subheader("Queue Time Analysis")

        hourly = calculate_hourly_metrics(df)
        pivot = hourly.pivot_table(
            index="stage",
            columns=hourly["hour"].dt.hour,
            values="avg_queue_time",
            aggfunc="mean"
        )

        fig = px.imshow(
            pivot,
            labels=dict(x="Hour of Day", y="Stage", color="Avg Queue (min)"),
            title="Queue Time Heatmap (Minutes)",
            color_continuous_scale="RdYlGn_r",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Predictions
    with tab2:
        st.header("Bottleneck Predictions")

        st.info("The ML model predicts which stage will become the bottleneck in the next hour based on current operational patterns.")

        # Train model
        with st.spinner("Training prediction model..."):
            predictor, train_metrics = train_predictor(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy", f"{train_metrics['test_accuracy']:.1%}")
        with col2:
            st.metric("Cross-Validation", f"{train_metrics['cv_mean']:.1%} ± {train_metrics['cv_std']:.1%}")
        with col3:
            st.metric("Model Type", "Random Forest")

        st.markdown("---")

        # Make prediction
        st.subheader("Current Prediction")

        prediction = predictor.predict(df)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>Predicted Next Bottleneck: {prediction.predicted_bottleneck.upper()}</h3>
                <p><strong>Probability:</strong> {prediction.probability:.1%}</p>
                <p><strong>Confidence Level:</strong> {prediction.confidence_level.upper()}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**All Stage Probabilities:**")
            prob_df = pd.DataFrame([
                {"Stage": k, "Probability": v}
                for k, v in prediction.all_probabilities.items()
            ]).sort_values("Probability", ascending=False)

            fig = px.bar(
                prob_df,
                x="Stage",
                y="Probability",
                color="Stage",
                title="Bottleneck Probability by Stage"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Current Risk Factors:**")
            if prediction.risk_factors:
                for risk in prediction.risk_factors:
                    st.warning(risk)
            else:
                st.success("No significant risk factors detected")

            # Feature importance
            st.markdown("**Top Predictive Features:**")
            importance = sorted(
                train_metrics["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:8]

            imp_df = pd.DataFrame(importance, columns=["Feature", "Importance"])
            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: What-If Analysis
    with tab3:
        st.header("What-If Scenario Analysis")

        st.info("Simulate operational changes to see their impact on throughput and lead time.")

        analyzer = WhatIfAnalyzer(df, workers)

        scenario_type = st.selectbox(
            "Scenario Type",
            ["Add Workers", "Improve Cycle Time", "Reallocate Workers"]
        )

        if scenario_type == "Add Workers":
            col1, col2 = st.columns(2)
            with col1:
                target_stage = st.selectbox("Target Stage", list(workers.keys()))
            with col2:
                add_count = st.slider("Workers to Add", 1, 5, 2)

            if st.button("Run Simulation", type="primary"):
                result = analyzer.simulate_add_workers(target_stage, add_count)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Throughput Change",
                        f"{result.throughput_change_pct:+.1f}%",
                        delta=f"{result.projected_throughput:.1f} orders/hr"
                    )
                with col2:
                    st.metric(
                        "Lead Time Change",
                        f"{result.lead_time_change_pct:+.1f}%"
                    )
                with col3:
                    if result.bottleneck_resolved:
                        st.success("✅ Bottleneck Resolved!")
                    else:
                        st.warning(f"Bottleneck remains at {result.projected_bottleneck}")

                st.markdown(f"**Implementation Notes:** {result.implementation_notes}")

        elif scenario_type == "Improve Cycle Time":
            col1, col2 = st.columns(2)
            with col1:
                target_stage = st.selectbox("Target Stage", list(workers.keys()))
            with col2:
                reduction_pct = st.slider("Cycle Time Reduction %", 5, 30, 15)

            if st.button("Run Simulation", type="primary"):
                result = analyzer.simulate_cycle_time_reduction(target_stage, reduction_pct)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Throughput Change",
                        f"{result.throughput_change_pct:+.1f}%"
                    )
                with col2:
                    st.metric(
                        "Lead Time Change",
                        f"{result.lead_time_change_pct:+.1f}%"
                    )
                with col3:
                    if result.bottleneck_resolved:
                        st.success("✅ Bottleneck Resolved!")
                    else:
                        st.warning(f"Bottleneck remains at {result.projected_bottleneck}")

        else:  # Reallocate Workers
            col1, col2, col3 = st.columns(3)
            with col1:
                from_stage = st.selectbox("From Stage", list(workers.keys()))
            with col2:
                to_stage = st.selectbox("To Stage", list(workers.keys()))
            with col3:
                move_count = st.slider("Workers to Move", 1, 3, 1)

            if st.button("Run Simulation", type="primary"):
                if from_stage == to_stage:
                    st.error("From and To stages must be different")
                else:
                    result = analyzer.simulate_worker_reallocation(from_stage, to_stage, move_count)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Throughput Change",
                            f"{result.throughput_change_pct:+.1f}%"
                        )
                    with col2:
                        st.metric(
                            "Lead Time Change",
                            f"{result.lead_time_change_pct:+.1f}%"
                        )
                    with col3:
                        if result.bottleneck_resolved:
                            st.success("✅ Bottleneck Resolved!")
                        else:
                            st.warning(f"Bottleneck at {result.projected_bottleneck}")

                    st.markdown(f"**Notes:** {result.implementation_notes}")

    # Tab 4: Recommendations
    with tab4:
        st.header("Optimization Recommendations")

        st.info("Ranked recommendations based on expected throughput improvement.")

        analyzer = WhatIfAnalyzer(df, workers)
        recommendations = analyzer.generate_recommendations(5)

        for rec in recommendations:
            effort_color = {
                "low": "🟢",
                "medium": "🟡",
                "high": "🔴"
            }[rec.implementation_effort]

            with st.expander(
                f"#{rec.rank}: {rec.action} — Expected: {rec.expected_improvement_pct:+.1f}%",
                expanded=rec.rank == 1
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Expected Improvement", f"{rec.expected_improvement_pct:+.1f}%")
                with col2:
                    st.metric("Implementation Effort", f"{effort_color} {rec.implementation_effort.upper()}")
                with col3:
                    st.metric(
                        "Lead Time Impact",
                        f"{rec.scenario_result.lead_time_change_pct:+.1f}%"
                    )

                st.markdown(f"**Reasoning:** {rec.reasoning}")
                st.markdown(f"**Implementation Notes:** {rec.scenario_result.implementation_notes}")

                if rec.scenario_result.bottleneck_resolved:
                    st.success("This action is expected to resolve the current bottleneck.")

        st.markdown("---")
        st.subheader("Quick Actions Summary")

        # Create comparison table
        summary_data = []
        for rec in recommendations:
            summary_data.append({
                "Rank": rec.rank,
                "Action": rec.action,
                "Improvement": f"{rec.expected_improvement_pct:+.1f}%",
                "Effort": rec.implementation_effort,
                "Resolves Bottleneck": "Yes" if rec.scenario_result.bottleneck_resolved else "No"
            })

        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True
        )


if __name__ == "__main__":
    main()
