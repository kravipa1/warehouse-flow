"""
Warehouse Flow Optimizer - Interactive Dashboard

A decision-support tool for warehouse managers to:
1. Identify current bottlenecks
2. Predict future constraints
3. Simulate operational changes
4. Get actionable recommendations
5. Calculate cost savings
6. Export reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import base64

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
    .cost-savings {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
    }
    .upload-section {
        background-color: #fff3e0;
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


def validate_uploaded_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate uploaded CSV has required columns."""
    required_cols = ["order_id", "stage", "start_time", "end_time"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"

    # Check stage values
    valid_stages = {"receiving", "picking", "packing", "sorting", "shipping"}
    unique_stages = set(df["stage"].str.lower().unique())
    invalid_stages = unique_stages - valid_stages

    if invalid_stages:
        return False, f"Invalid stage values: {invalid_stages}. Must be: {valid_stages}"

    return True, "Valid"


def process_uploaded_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Process uploaded CSV to match expected format."""
    df = df.copy()
    df["stage"] = df["stage"].str.lower()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # Calculate cycle time if not present
    if "cycle_time_minutes" not in df.columns:
        df["cycle_time_minutes"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60

    # Calculate queue time if not present (estimate as 0 for uploaded data)
    if "queue_time_minutes" not in df.columns:
        df["queue_time_minutes"] = 0

    # Add defaults for missing optional columns
    if "worker_id" not in df.columns:
        df["worker_id"] = "UNKNOWN"
    if "item_count" not in df.columns:
        df["item_count"] = 1
    if "priority" not in df.columns:
        df["priority"] = "standard"

    return df


def generate_excel_report(df, analysis, metrics, utilization, recommendations, cost_config):
    """Generate Excel report with multiple sheets."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            "Metric": [
                "Report Generated",
                "Total Orders Analyzed",
                "Date Range",
                "Current Bottleneck",
                "Bottleneck Confidence",
                "Primary Reason",
                "Total Queue Time (min)",
                "Estimated Delay Cost",
                "Potential Additional Orders"
            ],
            "Value": [
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                df["order_id"].nunique(),
                f"{df['start_time'].min().date()} to {df['end_time'].max().date()}",
                analysis.bottleneck_stage.upper(),
                f"{analysis.confidence:.1%}",
                analysis.primary_reason,
                f"{cost_config['total_queue_minutes']:,.0f}",
                f"${cost_config['total_delay_cost']:,.2f}",
                cost_config['potential_additional_orders']
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Stage Metrics sheet
        metrics.to_excel(writer, sheet_name="Stage Metrics", index=False)

        # Utilization sheet
        utilization.to_excel(writer, sheet_name="Utilization", index=False)

        # Recommendations sheet
        rec_data = []
        for rec in recommendations:
            rec_data.append({
                "Rank": rec.rank,
                "Action": rec.action,
                "Expected Improvement": f"{rec.expected_improvement_pct:+.1f}%",
                "Implementation Effort": rec.implementation_effort,
                "Resolves Bottleneck": "Yes" if rec.scenario_result.bottleneck_resolved else "No",
                "Reasoning": rec.reasoning,
                "Implementation Notes": rec.scenario_result.implementation_notes
            })
        pd.DataFrame(rec_data).to_excel(writer, sheet_name="Recommendations", index=False)

        # Cost Analysis sheet
        cost_data = {
            "Category": [
                "Hourly Worker Wage",
                "Cost per Delay Minute",
                "Total Queue Time (minutes)",
                "Total Delay Cost",
                "Potential Additional Orders",
                "Revenue per Order",
                "Lost Revenue from Delays",
                "Annual Projected Loss (if unchanged)"
            ],
            "Value": [
                f"${cost_config['hourly_wage']:.2f}",
                f"${cost_config['delay_cost_per_min']:.2f}",
                f"{cost_config['total_queue_minutes']:,.0f}",
                f"${cost_config['total_delay_cost']:,.2f}",
                f"{cost_config['potential_additional_orders']:,}",
                f"${cost_config['revenue_per_order']:.2f}",
                f"${cost_config['lost_revenue']:,.2f}",
                f"${cost_config['annual_loss']:,.2f}"
            ]
        }
        pd.DataFrame(cost_data).to_excel(writer, sheet_name="Cost Analysis", index=False)

    return output.getvalue()


def generate_csv_report(df, analysis, metrics, recommendations, cost_config):
    """Generate simple CSV summary report."""
    report_lines = []
    report_lines.append("WAREHOUSE FLOW OPTIMIZER - ANALYSIS REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("")
    report_lines.append("=== BOTTLENECK ANALYSIS ===")
    report_lines.append(f"Current Bottleneck: {analysis.bottleneck_stage.upper()}")
    report_lines.append(f"Confidence: {analysis.confidence:.1%}")
    report_lines.append(f"Reason: {analysis.primary_reason}")
    report_lines.append("")
    report_lines.append("=== COST IMPACT ===")
    report_lines.append(f"Total Delay Cost: ${cost_config['total_delay_cost']:,.2f}")
    report_lines.append(f"Lost Revenue: ${cost_config['lost_revenue']:,.2f}")
    report_lines.append(f"Annual Projected Loss: ${cost_config['annual_loss']:,.2f}")
    report_lines.append("")
    report_lines.append("=== TOP RECOMMENDATIONS ===")
    for rec in recommendations[:3]:
        report_lines.append(f"{rec.rank}. {rec.action} (Expected: {rec.expected_improvement_pct:+.1f}%)")

    return "\n".join(report_lines)


def main():
    st.title("📦 Warehouse Flow Optimizer")
    st.markdown("*Bottleneck Detection & Throughput Optimization*")

    # Initialize session state
    if "data_source" not in st.session_state:
        st.session_state.data_source = "synthetic"
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None

    # Sidebar - Data Source Selection
    st.sidebar.header("📊 Data Source")

    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Generate Synthetic Data", "Upload CSV File"],
        index=0 if st.session_state.data_source == "synthetic" else 1
    )

    df = None
    num_days = 14  # Default for calculations

    if data_source == "Generate Synthetic Data":
        st.session_state.data_source = "synthetic"

        with st.sidebar.expander("Synthetic Data Settings", expanded=True):
            num_days = st.slider("Days of data", 7, 30, 14)
            bottleneck_stage = st.selectbox(
                "Simulate bottleneck at",
                ["None", "picking", "packing", "sorting", "shipping"]
            )
            severity = st.slider("Bottleneck severity", 1.0, 2.0, 1.4, 0.1)
            seed = st.number_input("Random seed", value=42, step=1)

        df = load_data(num_days, bottleneck_stage, severity, int(seed))

    else:  # Upload CSV
        st.session_state.data_source = "upload"

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Required CSV columns:**")
        st.sidebar.code("order_id, stage, start_time, end_time")
        st.sidebar.markdown("**Optional columns:**")
        st.sidebar.code("worker_id, item_count, priority")

        uploaded_file = st.sidebar.file_uploader(
            "Upload warehouse event log",
            type=["csv"],
            help="CSV with columns: order_id, stage, start_time, end_time"
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                is_valid, message = validate_uploaded_csv(uploaded_df)

                if is_valid:
                    df = process_uploaded_csv(uploaded_df)
                    st.session_state.uploaded_df = df
                    st.sidebar.success(f"Loaded {len(df):,} events from {df['order_id'].nunique():,} orders")

                    # Calculate num_days from data
                    df["start_time"] = pd.to_datetime(df["start_time"])
                    num_days = (df["start_time"].max() - df["start_time"].min()).days + 1
                else:
                    st.sidebar.error(message)
                    df = None
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {str(e)}")
                df = None
        elif st.session_state.uploaded_df is not None:
            df = st.session_state.uploaded_df
            num_days = (df["start_time"].max() - df["start_time"].min()).days + 1
        else:
            st.sidebar.info("Please upload a CSV file")
            # Show sample data format
            st.sidebar.markdown("**Sample format:**")
            sample_df = pd.DataFrame({
                "order_id": ["ORD-001", "ORD-001", "ORD-002"],
                "stage": ["picking", "packing", "picking"],
                "start_time": ["2024-01-01 09:00", "2024-01-01 09:15", "2024-01-01 09:05"],
                "end_time": ["2024-01-01 09:12", "2024-01-01 09:22", "2024-01-01 09:18"]
            })
            st.sidebar.dataframe(sample_df, hide_index=True)

    # If no data, show message and return
    if df is None:
        st.warning("Please select a data source and load data to continue.")
        return

    # Sidebar - Worker Configuration
    st.sidebar.markdown("---")
    st.sidebar.header("👷 Worker Configuration")
    workers = {}
    for stage in ["receiving", "picking", "packing", "sorting", "shipping"]:
        workers[stage] = st.sidebar.number_input(
            f"{stage.capitalize()} workers",
            min_value=1,
            max_value=50,
            value=DEFAULT_WORKERS[stage]
        )

    # Sidebar - Cost Configuration
    st.sidebar.markdown("---")
    st.sidebar.header("💵 Cost Settings")
    hourly_wage = st.sidebar.number_input("Hourly wage ($)", min_value=10.0, max_value=100.0, value=18.0, step=0.5)
    delay_cost_per_min = st.sidebar.number_input("Cost per delay minute ($)", min_value=0.1, max_value=10.0, value=0.50, step=0.1)
    revenue_per_order = st.sidebar.number_input("Revenue per order ($)", min_value=1.0, max_value=500.0, value=25.0, step=1.0)

    # Main content - 5 tabs now
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Current Status",
        "🔮 Predictions",
        "🧪 What-If Analysis",
        "📋 Recommendations",
        "💰 Cost Analysis & Export"
    ])

    # Pre-calculate common data
    analysis = detect_bottleneck(df, workers)
    base_cost_analysis = calculate_bottleneck_cost(df, delay_cost_per_min)
    metrics = calculate_stage_metrics(df)
    utilization = calculate_utilization(df, workers)
    flow = calculate_flow_efficiency(df)

    # Tab 1: Current Status
    with tab1:
        st.header("Current Operational Status")

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_orders = df["order_id"].nunique()
            st.metric("Total Orders", f"{total_orders:,}")

        with col2:
            operating_hours = num_days * 16
            throughput = total_orders / operating_hours if operating_hours > 0 else 0
            st.metric("Throughput", f"{throughput:.1f} orders/hr")

        with col3:
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
                f"{base_cost_analysis['total_queue_minutes']:,.0f} min"
            )
            st.metric(
                "Est. Delay Cost",
                f"${base_cost_analysis['estimated_delay_cost']:,.0f}"
            )
            st.metric(
                "Potential Extra Orders",
                f"{base_cost_analysis['potential_additional_orders']:,}"
            )

        st.markdown("---")

        # Stage Metrics Visualization
        st.subheader("Stage Performance Metrics")

        col1, col2 = st.columns(2)

        with col1:
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
                        st.success("Bottleneck Resolved!")
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
                        st.success("Bottleneck Resolved!")
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
                            st.success("Bottleneck Resolved!")
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

    # Tab 5: Cost Analysis & Export (NEW)
    with tab5:
        st.header("Cost Analysis & Report Export")

        # Calculate comprehensive costs
        total_queue_minutes = flow["total_queue_time"].sum()
        total_delay_cost = total_queue_minutes * delay_cost_per_min
        potential_extra_orders = base_cost_analysis["potential_additional_orders"]
        lost_revenue = potential_extra_orders * revenue_per_order

        # Annualize (project based on data period)
        days_in_data = num_days
        annual_multiplier = 365 / days_in_data if days_in_data > 0 else 1
        annual_loss = (total_delay_cost + lost_revenue) * annual_multiplier

        cost_config = {
            "hourly_wage": hourly_wage,
            "delay_cost_per_min": delay_cost_per_min,
            "revenue_per_order": revenue_per_order,
            "total_queue_minutes": total_queue_minutes,
            "total_delay_cost": total_delay_cost,
            "potential_additional_orders": potential_extra_orders,
            "lost_revenue": lost_revenue,
            "annual_loss": annual_loss
        }

        st.subheader("💵 Financial Impact Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="cost-savings">
                <h3>Current Cost of Bottlenecks</h3>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Total Queue/Wait Time", f"{total_queue_minutes:,.0f} minutes")
            st.metric("Delay Costs", f"${total_delay_cost:,.2f}")
            st.metric("Lost Orders", f"{potential_extra_orders:,} orders")
            st.metric("Lost Revenue", f"${lost_revenue:,.2f}")

            st.markdown("---")
            st.metric(
                "Projected Annual Loss",
                f"${annual_loss:,.0f}",
                delta=f"Based on {days_in_data} days of data",
                delta_color="inverse"
            )

        with col2:
            st.markdown("**Cost Breakdown**")

            cost_breakdown = pd.DataFrame({
                "Category": ["Delay Costs", "Lost Revenue"],
                "Amount": [total_delay_cost, lost_revenue]
            })

            fig = px.pie(
                cost_breakdown,
                values="Amount",
                names="Category",
                title="Bottleneck Cost Distribution",
                color_discrete_sequence=["#ff6b6b", "#ffa502"]
            )
            st.plotly_chart(fig, use_container_width=True)

            # ROI Calculator
            st.markdown("---")
            st.markdown("**ROI Calculator**")

            if recommendations:
                top_rec = recommendations[0]
                improvement_pct = top_rec.expected_improvement_pct / 100

                potential_savings = annual_loss * improvement_pct
                implementation_cost = st.number_input(
                    "Estimated implementation cost ($)",
                    min_value=0,
                    max_value=1000000,
                    value=5000,
                    step=1000
                )

                if implementation_cost > 0:
                    roi = ((potential_savings - implementation_cost) / implementation_cost) * 100
                    payback_months = (implementation_cost / (potential_savings / 12)) if potential_savings > 0 else float('inf')

                    st.metric("Potential Annual Savings", f"${potential_savings:,.0f}")
                    st.metric("ROI", f"{roi:,.0f}%")
                    st.metric("Payback Period", f"{payback_months:.1f} months" if payback_months < 100 else "N/A")

        st.markdown("---")
        st.subheader("📥 Export Reports")

        col1, col2, col3 = st.columns(3)

        # Generate recommendations for export
        analyzer = WhatIfAnalyzer(df, workers)
        export_recommendations = analyzer.generate_recommendations(5)

        with col1:
            # Excel Export
            excel_data = generate_excel_report(
                df, analysis, metrics, utilization,
                export_recommendations, cost_config
            )

            st.download_button(
                label="📊 Download Excel Report",
                data=excel_data,
                file_name=f"warehouse_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.caption("Full analysis with multiple sheets")

        with col2:
            # CSV Summary Export
            csv_data = generate_csv_report(
                df, analysis, metrics,
                export_recommendations, cost_config
            )

            st.download_button(
                label="📄 Download Summary (TXT)",
                data=csv_data,
                file_name=f"warehouse_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
            st.caption("Quick text summary")

        with col3:
            # Raw Data Export
            st.download_button(
                label="📁 Download Raw Data (CSV)",
                data=df.to_csv(index=False),
                file_name=f"warehouse_events_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            st.caption("Event log data")


if __name__ == "__main__":
    main()
