"""Streamlit interactive dashboard for EventOptima fest schedule optimization."""

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from cost import cost_breakdown
from main import run_pipeline
from models import Problem, Schedule
from resources import volunteer_workload

st.set_page_config(
    page_title="EventOptima Fest Scheduler",
    page_icon="📅",
    layout="wide",
)

CATEGORY_COLORS = {
    "Technical": "#1f77b4",
    "Cultural": "#ff7f0e",
    "Workshop": "#2ca02c",
    "Talk": "#d62728",
    "Competition": "#9467bd",
}


def _slot_to_time_str(slot: int) -> str:
    """Convert discrete 30-min slot index (0..23) to hh:mm string."""
    base_m = 9 * 60 + slot * config.SLOT_MINUTES
    h = base_m // 60
    m = base_m % 60
    return f"{h:02d}:{m:02d}"


def _build_timeline_df(problem: Problem, schedule: Schedule, day_filter: int) -> pd.DataFrame:
    """Build a DataFrame compatible with plotly.express.timeline for Gantt visualization."""
    rows = []
    base_dt = datetime.strptime(config.BASE_DATE, "%Y-%m-%d")

    for e_idx in range(len(problem.event_ids)):
        r_idx = schedule.room_idx[e_idx]
        if r_idx < 0:
            continue
        d = int(schedule.day[e_idx])
        if d != day_filter:
            continue

        s = int(schedule.start[e_idx])
        dur = int(problem.dur[e_idx])
        eid = problem.event_ids[e_idx]
        ev_row = problem.events.iloc[e_idx]

        start_dt = base_dt + timedelta(days=d, minutes=s * config.SLOT_MINUTES + 9 * 60)
        end_dt = start_dt + timedelta(minutes=dur * config.SLOT_MINUTES)

        rows.append(
            {
                "Event ID": eid,
                "Event Name": ev_row["name"],
                "Category": ev_row["category"],
                "Room": problem.room_ids[r_idx],
                "Start Time": start_dt,
                "End Time": end_dt,
                "Time Display": f"{_slot_to_time_str(s)} - {_slot_to_time_str(s + dur)}",
                "Attendance": int(problem.attendance[e_idx]),
                "Capacity": int(problem.room_cap[r_idx]),
                "Duration (Slots)": dur,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Event ID",
                "Event Name",
                "Category",
                "Room",
                "Start Time",
                "End Time",
                "Time Display",
                "Attendance",
                "Capacity",
                "Duration (Slots)",
            ]
        )
    return pd.DataFrame(rows)


def _build_occupancy_matrix(
    problem: Problem, schedule: Schedule, unavailable_ids: set[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a 2D binary occupancy matrix for heatmap visualization."""
    n_rooms = len(problem.room_ids)
    matrix = np.zeros((n_rooms, config.N_DAYS * config.SLOTS_PER_DAY), dtype=int)

    for e_idx in range(len(problem.event_ids)):
        r_idx = schedule.room_idx[e_idx]
        if r_idx < 0:
            continue
        d = int(schedule.day[e_idx])
        s = int(schedule.start[e_idx])
        dur = int(problem.dur[e_idx])
        if 0 <= d < config.N_DAYS:
            col_start = d * config.SLOTS_PER_DAY + s
            col_end = min(config.N_DAYS * config.SLOTS_PER_DAY, col_start + dur)
            matrix[r_idx, col_start:col_end] = 1

    time_labels = []
    for d in range(config.N_DAYS):
        for s in range(config.SLOTS_PER_DAY):
            time_labels.append(f"D{d+1} {_slot_to_time_str(s)}")

    room_labels = []
    for idx, rid in enumerate(problem.room_ids):
        if rid in unavailable_ids or not problem.room_available[idx]:
            room_labels.append(f"{rid} (down)")
        else:
            room_labels.append(rid)

    return matrix, room_labels, time_labels


def main() -> None:
    """Render the full Streamlit interface."""
    st.title("🎯 EventOptima: Two-Day Fest Schedule Optimizer")
    st.markdown(
        "Clash-free, cost-optimized timetable generation using **Simulated Annealing** and **Genetic Algorithms**."
    )

    # 1. Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=config.SEED)
    algo_label = st.sidebar.selectbox(
        "Optimization Algorithm",
        options=["Simulated Annealing", "Greedy", "Genetic Algorithm"],
        index=0,
    )
    algo_map = {
        "Simulated Annealing": "sa",
        "Greedy": "greedy",
        "Genetic Algorithm": "ga",
    }
    algo_code = algo_map[algo_label]

    iterations = st.sidebar.slider(
        "Iterations / Computational Budget",
        min_value=1000,
        max_value=50000,
        value=config.SA_ITER,
        step=1000,
    )
    compare = st.sidebar.checkbox("Compare SA vs GA", value=False)

    # Load problem to get room options
    default_room_ids = [f"R{i+1:02d}" for i in range(10)]
    unavailable_selected = st.sidebar.multiselect(
        "Rooms unavailable (what-if)",
        options=default_room_ids,
        default=[],
    )

    col_btn1, col_btn2 = st.sidebar.columns(2)
    optimize_clicked = col_btn1.button("🚀 Optimize", use_container_width=True, type="primary")
    what_if_clicked = col_btn2.button("🔄 Re-optimize with what-if", use_container_width=True)

    if optimize_clicked:
        with st.spinner("Running primary schedule optimization..."):
            res = run_pipeline(
                seed=seed,
                algo=algo_code,
                iterations=iterations,
                unavailable=(),
                compare=compare,
            )
            st.session_state["results"] = res
            st.session_state["what_if_banner"] = ""
            st.session_state["unavailable"] = set()

    if what_if_clicked:
        with st.spinner(f"Re-optimizing with {len(unavailable_selected)} unavailable room(s)..."):
            res = run_pipeline(
                seed=seed,
                algo=algo_code,
                iterations=iterations,
                unavailable=unavailable_selected,
                compare=compare,
            )
            st.session_state["results"] = res
            banner = f"What-if: {', '.join(unavailable_selected)} unavailable"
            st.session_state["what_if_banner"] = banner
            st.session_state["unavailable"] = set(unavailable_selected)

    # If nothing has run yet
    if "results" not in st.session_state:
        st.info("👆 Click **Optimize** in the sidebar to generate schedules and analytics.")
        return

    results = st.session_state["results"]
    prob: Problem = results["problem"]
    before: Schedule = results["before"]
    after: Schedule = results["after"]
    kpis_before = results["kpis_before"]
    kpis_after = results["kpis_after"]
    unavail_set: set[str] = st.session_state.get("unavailable", set())

    # What-if banner & validation check
    banner_text = st.session_state.get("what_if_banner", "")
    if banner_text:
        st.warning(f"⚠️ **{banner_text}**")
        # Ensure no event appears in an unavailable room
        for rid in unavail_set:
            if rid in prob.room_ids:
                ridx = prob.room_ids.index(rid)
                if np.any(after.room_idx == ridx):
                    st.error(f"❌ Violation: Event placed in unavailable room {rid}!")

    # 2. KPI Cards Top Row
    st.subheader("📊 Key Performance Metrics")
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.metric(
            "Total Cost",
            f"{kpis_after['total_cost']:,.0f}",
            delta=f"{kpis_after['total_cost'] - kpis_before['total_cost']:,.0f}",
            delta_color="inverse",
        )
    with kpi_cols[1]:
        st.metric(
            "Student Clashes",
            f"{kpis_after['student_clashes']:.0f}",
            delta=f"{kpis_after['student_clashes'] - kpis_before['student_clashes']:.0f}",
            delta_color="inverse",
        )
    with kpi_cols[2]:
        st.metric(
            "Room Utilization",
            f"{kpis_after['room_utilization_pct']:.1f}%",
            delta=f"{kpis_after['room_utilization_pct'] - kpis_before['room_utilization_pct']:.1f}%",
            delta_color="normal",
        )
    with kpi_cols[3]:
        st.metric(
            "Hard Violations",
            f"{kpis_after['hard_violations']:.0f}",
            delta=f"{kpis_after['hard_violations'] - kpis_before['hard_violations']:.0f}",
            delta_color="inverse",
        )
    with kpi_cols[4]:
        st.metric(
            "Late Events (≥18:00)",
            f"{kpis_after['late_events']:.0f}",
            delta=f"{kpis_after['late_events'] - kpis_before['late_events']:.0f}",
            delta_color="inverse",
        )

    # 3. Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "📅 Timetable",
            "🔥 Heatmap",
            "⚖️ Comparison",
            "📉 Convergence",
            "🚨 Violations",
            "👥 Resources",
            "📈 Forecast",
        ]
    )

    # Tab 1: Timetable
    with tab1:
        st.markdown("#### Interactive Timetable Gantt Chart")
        tcol1, tcol2 = st.columns([1, 1])
        with tcol1:
            sched_view = st.radio("Schedule View:", ["Before (Baseline)", "After (Optimized)"], horizontal=True)
        with tcol2:
            day_choice = st.radio("Day Selector:", ["Day 1", "Day 2"], horizontal=True)

        chosen_day = 0 if day_choice == "Day 1" else 1
        active_sched = before if "Before" in sched_view else after

        df_gantt = _build_timeline_df(prob, active_sched, chosen_day)
        if df_gantt.empty:
            st.info("No events scheduled for this day.")
        else:
            fig_gantt = px.timeline(
                df_gantt,
                x_start="Start Time",
                x_end="End Time",
                y="Room",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                hover_name="Event Name",
                hover_data={
                    "Event ID": True,
                    "Time Display": True,
                    "Attendance": True,
                    "Capacity": True,
                    "Start Time": False,
                    "End Time": False,
                },
                category_orders={"Room": sorted(prob.room_ids, reverse=True)},
            )
            fig_gantt.update_yaxes(autorange="reversed")
            fig_gantt.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gantt, use_container_width=True)

    # Tab 2: Utilization Heatmap
    with tab2:
        st.markdown("#### Room Occupancy Heatmap Across Fest Duration (48 Half-Hour Slots)")
        m_before, r_labels_b, time_cols = _build_occupancy_matrix(prob, before, unavail_set)
        m_after, r_labels_a, _ = _build_occupancy_matrix(prob, after, unavail_set)

        hcol1, hcol2 = st.columns(2)
        with hcol1:
            st.markdown("**Before (Baseline)**")
            fig_hb = go.Figure(
                data=go.Heatmap(
                    z=m_before,
                    x=time_cols,
                    y=r_labels_b,
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig_hb.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_hb, use_container_width=True)

        with hcol2:
            st.markdown("**After (Optimized)**")
            fig_ha = go.Figure(
                data=go.Heatmap(
                    z=m_after,
                    x=time_cols,
                    y=r_labels_a,
                    colorscale="Greens",
                    showscale=False,
                )
            )
            fig_ha.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_ha, use_container_width=True)

    # Tab 3: Comparison
    with tab3:
        st.markdown("#### Comprehensive KPI & Cost Comparison")
        comp_df = results["comparison"]
        st.dataframe(comp_df, use_container_width=True)

        bd_b = cost_breakdown(prob, before, partial=False)
        bd_a = cost_breakdown(prob, after, partial=False)
        cost_cats = ["clash", "room", "waste", "gap", "late"]

        df_bar = pd.DataFrame(
            {
                "Cost Term": [c.capitalize() for c in cost_cats] * 2,
                "Weighted Cost": [bd_b[c] for c in cost_cats] + [bd_a[c] for c in cost_cats],
                "Schedule": ["Before"] * len(cost_cats) + ["After"] * len(cost_cats),
            }
        )
        fig_bar = px.bar(
            df_bar,
            x="Cost Term",
            y="Weighted Cost",
            color="Schedule",
            barmode="group",
            title="Cost Component Breakdown: Before vs After",
            color_discrete_map={"Before": "#d62728", "After": "#2ca02c"},
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tab 4: Convergence
    with tab4:
        st.markdown("#### Cost Minimization Convergence Curves")
        histories = results["histories"]
        fig_conv = go.Figure()
        for label, h in histories.items():
            fig_conv.add_trace(go.Scatter(y=h, mode="lines", name=label, line=dict(width=2)))
        fig_conv.update_layout(
            title="Optimization Convergence History",
            xaxis_title="Iteration / Generation",
            yaxis_title="Objective Cost",
            height=450,
            template="plotly_white",
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    # Tab 5: Violations
    with tab5:
        st.markdown("#### Rule Violation Analysis")
        v_before = results["violations_before"]
        v_after = results["violations_after"]

        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.markdown(f"**Before Violations ({len(v_before)} total)**")
            if not v_before.empty:
                st.write(v_before["rule"].value_counts())
                st.dataframe(v_before, use_container_width=True)
            else:
                st.success("No violations in Before schedule.")

        with vcol2:
            st.markdown(f"**After Violations ({len(v_after)} total)**")
            if not v_after.empty:
                st.write(v_after["rule"].value_counts())
                st.dataframe(v_after, use_container_width=True)
            else:
                st.success("🎉 Zero constraint violations in Optimized schedule!")

    # Tab 6: Resources
    with tab6:
        st.markdown("#### Volunteer Workload & Equipment Allocation")
        assign_after = results["assign_after"]
        wl = volunteer_workload(prob, assign_after)

        rcol1, rcol2 = st.columns([1, 1])
        with rcol1:
            st.markdown("**Volunteer Workload Distribution (After)**")
            fig_vol = px.bar(
                wl,
                x="volunteer_id",
                y="hours",
                title="Assigned Hours per Volunteer",
                labels={"volunteer_id": "Volunteer ID", "hours": "Hours Assigned"},
                color="hours",
                color_continuous_scale="Viridis",
            )
            fig_vol.update_layout(height=380)
            st.plotly_chart(fig_vol, use_container_width=True)

            std_val = kpis_after.get("volunteer_hours_std", 0.0)
            unfilled_val = kpis_after.get("unfilled_volunteer_slots", 0.0)
            st.metric("Workload Std Dev", f"{std_val:.2f} hrs")
            st.metric("Unfilled Volunteer Slots", f"{unfilled_val:.0f}")

        with rcol2:
            st.markdown("**Equipment Allocation (After)**")
            equip_df = results["equip_after"]
            st.dataframe(equip_df, use_container_width=True, height=450)

    # Tab 7: Forecast
    with tab7:
        st.markdown("#### Attendance Trend Forecast (Target: 2026)")
        fc_df = results["forecast"]

        fcol1, fcol2 = st.columns([1, 1])
        with fcol1:
            st.markdown("**Category Projections & Regression Stats**")
            st.dataframe(fc_df, use_container_width=True)

        with fcol2:
            past_df = prob.past_attendance
            fig_fc = go.Figure()
            for cat in sorted(past_df["category"].unique()):
                sub = past_df[past_df["category"] == cat].sort_values("year")
                fig_fc.add_trace(
                    go.Scatter(
                        x=sub["year"],
                        y=sub["attendance"],
                        mode="lines+markers",
                        name=f"{cat} (Past)",
                        line=dict(color=CATEGORY_COLORS.get(cat, "#333")),
                    )
                )
                fc_row = fc_df[fc_df["category"] == cat].iloc[0]
                fig_fc.add_trace(
                    go.Scatter(
                        x=[config.FORECAST_YEAR],
                        y=[fc_row["forecast"]],
                        mode="markers",
                        marker=dict(size=12, symbol="star", color=CATEGORY_COLORS.get(cat, "#333")),
                        name=f"{cat} (2026 Forecast)",
                    )
                )

            fig_fc.update_layout(
                title="Historical vs 2026 Forecast Attendance",
                xaxis_title="Year",
                yaxis_title="Total Attendance",
                height=400,
                template="plotly_white",
            )
            st.plotly_chart(fig_fc, use_container_width=True)


if __name__ == "__main__":
    main()
