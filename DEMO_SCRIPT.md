# EventOptima: 5-Minute Demonstration Script

This script provides an exact step-by-step presentation narrative for demonstrating the **EventOptima** platform to stakeholders and evaluators.

---

### Step 1: Problem Statement (0:00 - 0:30)
- **Action:** Open terminal and run `streamlit run app.py`. Switch to the browser view.
- **Narrative:**
  > *"Welcome everyone. Organizing a college fest involves scheduling dozens of events across multiple days with limited auditoriums, classrooms, and labs. But the hardest challenge is student clashes: when popular hackathons, guest lectures, and dance battles happen at the same time, students are forced to choose, reducing attendance and frustrating participants. Add shared equipment stock, room costs, and volunteer work limits, and manual scheduling becomes nearly impossible. EventOptima solves this mathematically."*

---

### Step 2: The "Before" Baseline Schedule (0:30 - 1:15)
- **Action:** Point out the initial state. Click **Optimize** in the sidebar. When loaded, click on the **Timetable** tab and select the **Before (Baseline)** radio button.
- **Narrative:**
  > *"Notice our unoptimized Baseline schedule. A naive first-fit algorithm places events into the first free compatible room it finds. When we look at the Violations tab, we see 20 major hard violations: equipment stock is exceeded in multiple slots, and room double-bookings occur. Worst of all, 515 student registration clashes exist across 95 pairs of overlapping events. Over $16,900 is spent on room rentals, yet hundreds of seats sit empty."*

---

### Step 3: Optimization & KPI Cards (1:15 - 2:00)
- **Action:** Direct attention to the top KPI cards showing Before vs After deltas.
- **Narrative:**
  > *"When we run our Simulated Annealing optimization engine over 20,000 iterations, observe the transformation in our KPI cards:
  > - **Total Cost** plunged by **98.6%**, dropping from 213,000 down to 2,900.
  > - **Student Clashes** decreased from 515 down to just 28—a **94.6% reduction**!
  > - **Hard Violations** were completely eliminated from 20 to **0**.
  > - **Room rental costs** dropped by **14%**, saving over $2,300.
  > - **Seat utilization** improved from 33.6% to **41.2%**."*

---

### Step 4: "After" Timetable & Heatmap (2:00 - 2:45)
- **Action:** Switch radio to **After (Optimized)** in the Timetable tab. Toggle Day 1 and Day 2. Then switch to the **Heatmap** tab.
- **Narrative:**
  > *"In the After Timetable, events are distributed cleanly across the day. High-attendance events are prioritized in larger venues, while smaller workshops occupy classrooms. Notice how events with high registration overlaps are staggered to avoid conflicting times. In the Heatmap tab, you can compare room occupancy side-by-side: the optimized schedule consolidates events efficiently into 8 rooms instead of 10, freeing up unused spaces for other campus operations."*

---

### Step 5: Convergence Tab (2:45 - 3:30)
- **Action:** Click the **Convergence** tab.
- **Narrative:**
  > *"Here in the Convergence tab, we observe the optimization history. Notice how the objective cost rapidly drops within the first 3,000 iterations as hard violations are eliminated, followed by steady fine-tuning that resolves pairwise student clashes and minimizes idle gaps. With the 'Compare SA vs GA' option, we can also see the Genetic Algorithm curve, demonstrating how Simulated Annealing achieves superior local convergence."*

---

### Step 6: What-If Scenario Simulation (3:30 - 4:15)
- **Action:** In the sidebar, select **R01** (Main Auditorium) under *Rooms unavailable (what-if)*. Click **Re-optimize with what-if**.
- **Narrative:**
  > *"In real fests, unexpected problems happen—an auditorium's projector fails or a venue gets locked out. Let's select 'R01 (Main Auditorium)' as unavailable and click Re-optimize with what-if.
  > Notice the warning banner: 'What-if: R01 unavailable'. The engine immediately adapts: every event previously in the Main Auditorium is seamlessly shifted to the Mini Auditorium (R02) and Open Air Theatre without causing room double-bookings or student clashes. The app validates that zero events appear in R01."*

---

### Step 7: Resources & Attendance Forecast (4:15 - 4:45)
- **Action:** Click the **Resources** tab, then the **Forecast** tab.
- **Narrative:**
  > *"In the Resources tab, our greedy load balancer distributes event shifts across all 40 volunteers, keeping workload variance down to a standard deviation of under 1 hour with zero unfilled slots. Equipment allocations verify that projector, mic, and sound system stocks are never breached.
  > In the Forecast tab, our polynomial regression models project 2026 fest attendance based on 5-year category trends, giving the planning committee actionable data for future editions."*

---

### Step 8: Conclusion & Headline Numbers (4:45 - 5:00)
- **Action:** Switch back to the **Comparison** tab showing the full summary table.
- **Narrative:**
  > *"To summarize: EventOptima takes an uncoordinated fest timetable riddled with clashes and violations, and delivers a clash-free, cost-optimized, fully staffed schedule in seconds—cutting student clashes by 94.6% and saving 14% on venue rental. Thank you!"*
