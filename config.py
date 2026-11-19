"""Global configuration constants for EventOptima."""

SEED = 42
N_DAYS = 2
SLOT_MINUTES = 30
SLOTS_PER_DAY = 24
DAY_START = '09:00'
LATE_START_SLOT = 18
BASE_DATE = '2026-03-01'          # used only to build datetimes for charts
W_CLASH = 20.0     # per student-clash
W_ROOM = 0.1       # per currency unit of room cost
W_WASTE = 0.5      # per wasted seat
W_GAP = 3.0        # per idle gap slot
W_LATE = 40.0      # per late event
W_HARD = 10000.0   # per hard violation
SKILLS = ['tech_support', 'stage_management', 'registration', 'hospitality', 'first_aid']
CATEGORIES = ['Technical', 'Cultural', 'Workshop', 'Talk', 'Competition']
ROOM_TYPES = ['auditorium', 'open_air', 'classroom', 'lab', 'seminar']
PAST_YEARS = [2021, 2022, 2023, 2024, 2025]
FORECAST_YEAR = 2026
SA_ITER = 20000
SA_T0 = 50.0
SA_T_END = 0.05
GA_POP = 60
GA_GENERATIONS = 300
GA_ELITE = 2
GA_TOURNAMENT_K = 3
GA_MUT_RATE = 0.05
