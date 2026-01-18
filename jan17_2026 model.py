import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

st.set_page_config(page_title="Therapy Practice Financial Model V4.4", layout="wide")
st.title("🧠 Therapy Practice Financial Model V4.4")
st.markdown("Fixed churn modeling - each client cohort experiences proper churn progression")

# ==========================================
# CONSTANTS
# ==========================================

MONTHS_PER_YEAR = 12
OWNER_SUPERVISION_CAPACITY = 3  # Max LMSW owner can supervise
SUPERVISION_HOURS_PER_LMSW_WEEK = 1.0  # 1 hour per week per LMSW
LMSW_SHARE_SUPERVISION = 2  # 2 LMSWs share one supervision hour
EXTERNAL_SUPERVISION_HOURLY = 100.0  # $/hour for external LCSW
RAMP_UP_MONTHS = 5  # Therapist ramp-up period

# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class Therapist:
    id: int
    name: str
    credential: str
    hire_month: int
    sessions_per_week_target: int  # Actual completed sessions per week
    one_time_hiring_cost: float = 0.0
    start_full: bool = False  # Start with full caseload (for existing therapists)
    
    def is_active(self, current_month: int) -> bool:
        """Therapist is active immediately when hired (already credentialed)"""
        return current_month >= self.hire_month
    
    def get_capacity_percentage(self, current_month: int, ramp_speed: str = "Medium") -> float:
        """Ramp up based on speed setting"""
        if not self.is_active(current_month):
            return 0.0
        
        months_since_hire = current_month - self.hire_month
        
        # Different ramp schedules
        if ramp_speed == "Slow":
            schedule = [0.10, 0.30, 0.50, 0.70, 0.85, 1.0]
        elif ramp_speed == "Fast":
            schedule = [0.30, 0.60, 0.85, 1.0]
        else:  # Medium
            schedule = [0.20, 0.40, 0.60, 0.80, 1.0]
        
        if months_since_hire >= len(schedule):
            return 1.0
        else:
            return schedule[months_since_hire]
    
    def get_sessions_per_month(self, current_month: int, weeks_per_month: float, ramp_speed: str = "Medium") -> float:
        """Calculate actual sessions per month - sessions_per_week_target already represents completed sessions"""
        capacity_pct = self.get_capacity_percentage(current_month, ramp_speed)
        return self.sessions_per_week_target * weeks_per_month * capacity_pct

@dataclass
class ClientCohort:
    """Tracks a group of clients that started in the same month"""
    start_month: int
    initial_count: float
    current_count: float
    therapist_id: int

@dataclass
class AdminEmployee:
    """Admin employee - 1099 contractor"""
    id: int
    name: str
    hours_per_week: float
    hourly_rate: float
    start_month: int
    
    def is_active(self, current_month: int) -> bool:
        return current_month >= self.start_month and self.start_month > 0
    
    def get_monthly_cost(self, weeks_per_month: float) -> float:
        return self.hours_per_week * self.hourly_rate * weeks_per_month

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def apply_cohort_churn(cohort: ClientCohort, current_month: int, 
                       churn1: float, churn2: float, churn3: float, churn_ongoing: float) -> float:
    """Apply appropriate churn rate based on cohort age"""
    cohort_age = current_month - cohort.start_month + 1
    
    if cohort_age == 1:
        churned = cohort.current_count * churn1
    elif cohort_age == 2:
        churned = cohort.current_count * churn2
    elif cohort_age == 3:
        churned = cohort.current_count * churn3
    else:
        churned = cohort.current_count * churn_ongoing
    
    return churned

def calculate_runway(cash_balance: float, avg_cash_flow: float) -> Tuple[float, str, str]:
    """Calculate cash runway and status"""
    if avg_cash_flow >= 0:
        return float('inf'), 'green', 'Positive Cash Flow'
    
    runway = abs(cash_balance / avg_cash_flow) if avg_cash_flow < 0 else float('inf')
    
    if cash_balance < 0:
        color = 'red'
        status = 'Negative Balance (Operating on Credit)'
    elif runway < 3:
        color = 'red'
        status = f'{runway:.1f} months runway'
    elif runway < 6:
        color = 'orange'
        status = f'{runway:.1f} months runway'
    else:
        color = 'green'
        status = f'{runway:.1f} months runway'
    
    return runway, color, status

def get_ehr_cost(num_therapists: int, system: str, custom_cost: float = None) -> float:
    """Calculate EHR cost with tiered pricing"""
    if system == "Custom":
        return custom_cost if custom_cost else 75.0
    
    if system == "SimplePractice":
        if num_therapists <= 3:
            return 99.0
        elif num_therapists <= 9:
            return 89.0
        else:
            return 79.0
    elif system == "TherapyNotes":
        if num_therapists <= 3:
            return 59.0
        elif num_therapists <= 9:
            return 49.0
        else:
            return 39.0
    return 75.0

def calculate_supervision_costs(
    num_lmsw: int,
    weeks_per_month: float
) -> Tuple[float, int, int]:
    """Calculate supervision costs - only external cash costs
    Owner supervision (up to 3) = $0 cost
    External = $100/hour with 2 LMSWs sharing each hour"""
    
    if num_lmsw == 0:
        return 0.0, 0, 0
    
    hours_per_lmsw_month = SUPERVISION_HOURS_PER_LMSW_WEEK * weeks_per_month
    
    if num_lmsw <= OWNER_SUPERVISION_CAPACITY:
        # Owner provides supervision - no cost
        supervision_cost = 0.0
        owner_supervised = num_lmsw
        external_supervised = 0
    else:
        # Owner supervises 3, external supervises rest
        owner_supervised = OWNER_SUPERVISION_CAPACITY
        external_supervised = num_lmsw - OWNER_SUPERVISION_CAPACITY
        
        # External cost only: $100/hour split between 2 LMSWs = $50/LMSW/hour
        cost_per_lmsw_hour = EXTERNAL_SUPERVISION_HOURLY / LMSW_SHARE_SUPERVISION
        supervision_cost = external_supervised * hours_per_lmsw_month * cost_per_lmsw_hour
    
    return supervision_cost, owner_supervised, external_supervised

def calculate_breakeven_sessions(
    credential: str,
    lmsw_pay: float,
    lcsw_pay: float,
    weighted_rate: float,
    revenue_loss_pct: float,
    monthly_fixed: float,
    payroll_tax: float,
    weeks_per_month: float
) -> Dict[str, float]:
    """Calculate break-even sessions accounting for revenue loss"""
    pay = lmsw_pay if credential == "LMSW" else lcsw_pay
    effective_revenue = weighted_rate * (1 - revenue_loss_pct)
    variable_cost = pay * (1 + payroll_tax)
    contribution = effective_revenue - variable_cost
    
    if contribution <= 0:
        return {'monthly': float('inf'), 'weekly': float('inf'), 'contribution': contribution}
    
    breakeven_monthly = monthly_fixed / contribution
    return {
        'monthly': breakeven_monthly,
        'weekly': breakeven_monthly / weeks_per_month,
        'contribution': contribution
    }

def calculate_collections(
    revenue_history: Dict[str, List[Dict]],
    month: int,
    payer_mix: Dict[str, Dict],
    copay_rev: float,
    cc_fees: float
) -> float:
    """Calculate cash collections with payment delays"""
    collections = copay_rev - cc_fees
    
    for payer, info in payer_mix.items():
        delay_months = info["delay_days"] / 30.0
        
        if delay_months == 0:
            if len(revenue_history[payer]) >= month:
                collections += revenue_history[payer][month - 1]["revenue"]
            continue
        
        earned_month = month - delay_months
        if earned_month < 1:
            continue
        
        floor_m = int(np.floor(earned_month))
        ceil_m = int(np.ceil(earned_month))
        fraction = earned_month - floor_m
        
        floor_idx = floor_m - 1
        if 0 <= floor_idx < len(revenue_history[payer]):
            collections += revenue_history[payer][floor_idx]["revenue"] * (1 - fraction)
        
        if ceil_m != floor_m:
            ceil_idx = ceil_m - 1
            if 0 <= ceil_idx < len(revenue_history[payer]):
                collections += revenue_history[payer][ceil_idx]["revenue"] * fraction
    
    return collections

def get_seasonality_factor(month: int, apply_seasonality: bool) -> Tuple[float, float]:
    """Returns (demand_multiplier, attendance_multiplier) for seasonality"""
    if not apply_seasonality:
        return 1.0, 1.0
    
    # Month within year (1-12)
    month_in_year = ((month - 1) % 12) + 1
    
    # Demand affects new client interest
    # Attendance affects session completion
    seasonality = {
        1: (1.15, 1.05),   # January - New Year surge
        2: (1.05, 1.00),   # February  
        3: (1.00, 1.00),   # March
        4: (0.95, 0.95),   # April
        5: (0.90, 0.90),   # May
        6: (0.85, 0.85),   # June - Summer begins
        7: (0.80, 0.80),   # July - Summer low
        8: (0.85, 0.85),   # August
        9: (1.20, 1.10),   # September - Back to school
        10: (1.00, 1.00),  # October
        11: (0.85, 0.90),  # November - Holidays approach
        12: (0.70, 0.75),  # December - Holiday low
    }
    
    return seasonality.get(month_in_year, (1.0, 1.0))

def project_new_hire(
    credential: str,
    sessions_week: int,
    lmsw_pay: float,
    lcsw_pay: float,
    weighted_rate: float,
    revenue_loss_pct: float,
    payroll_tax: float,
    overhead_monthly: float,
    supervision_cost_monthly: float,
    hiring_cost: float,
    weeks_per_month: float,
    ramp_speed: str = "Medium",
    months: int = 12
) -> pd.DataFrame:
    """Project financial impact of hiring new therapist"""
    data = []
    pay_rate = lmsw_pay if credential == "LMSW" else lcsw_pay
    
    cumulative_profit = -hiring_cost  # Start with hiring cost
    
    # Create temporary therapist for capacity calculation
    temp_therapist = Therapist(999, "New Hire", credential, 1, sessions_week, hiring_cost, False)
    
    for m in range(1, months + 1):
        # Get capacity percentage
        capacity_pct = temp_therapist.get_capacity_percentage(m, ramp_speed)
        
        # Calculate sessions and revenue (sessions_week already represents completed sessions)
        sessions = sessions_week * weeks_per_month * capacity_pct
        revenue = sessions * weighted_rate * (1 - revenue_loss_pct)
        
        # Calculate costs
        pay_cost = sessions * pay_rate * (1 + payroll_tax)
        total_cost = pay_cost + overhead_monthly + supervision_cost_monthly
        
        # Profit
        monthly_profit = revenue - total_cost
        cumulative_profit += monthly_profit
        
        data.append({
            'month': m,
            'capacity_%': capacity_pct * 100,
            'sessions': sessions,
            'revenue': revenue,
            'costs': total_cost,
            'monthly_profit': monthly_profit,
            'cumulative_profit': cumulative_profit
        })
    
    return pd.DataFrame(data)

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("💵 Starting Capital")
starting_cash_balance = st.sidebar.number_input(
    "Starting Cash Balance ($)",
    value=10000.0,
    min_value=0.0,
    step=1000.0,
    help="Cash you're starting with"
)

st.sidebar.header("📅 Working Schedule")
working_weeks_per_year = st.sidebar.number_input(
    "Working Weeks per Year",
    min_value=40,
    max_value=52,
    value=45,
    help="Account for vacation, holidays, sick time. 45 weeks = ~7 weeks off"
)

# Calculate weeks per month based on working weeks
WEEKS_PER_MONTH = working_weeks_per_year / 12

st.sidebar.header("👤 Owner")
owner_sessions_per_week = st.sidebar.number_input("Your Sessions/Week (Actual Completed)", min_value=0, value=20)

st.sidebar.header("🧑‍⚕️ Therapist Hiring")
ramp_speed = st.sidebar.selectbox("Ramp-Up Speed", ["Slow", "Medium", "Fast"], index=1,
    help="Slow: 6 months to full, Medium: 5 months, Fast: 4 months")
st.sidebar.markdown(f"**Selected: {ramp_speed} ramp-up**")

therapists = []
therapists.append(Therapist(0, "Owner", "LCSW", 0, owner_sessions_per_week, 0, False))

default_hires = [
    (1, "Therapist 1", "LMSW", 3, 20, 3000.0),
    (2, "Therapist 2", "LMSW", 6, 20, 3000.0),
    (3, "Therapist 3", "LCSW", 9, 20, 3500.0)
]

for i in range(1, 13):
    with st.sidebar.expander(f"Therapist {i}", expanded=(i <= 3)):
        if i <= len(default_hires):
            def_month = default_hires[i-1][3]
            def_cred = default_hires[i-1][2]
            def_cost = default_hires[i-1][5]
        else:
            def_month = 0
            def_cred = "LMSW"
            def_cost = 3000.0
        
        col1, col2 = st.columns(2)
        hire_m = col1.number_input(f"Hire Month", 0, 36, def_month, key=f"hire_{i}")
        
        if hire_m == 0:
            # Existing therapist - show full caseload option
            start_full = st.checkbox("Start with Full Caseload", value=False, key=f"full_{i}")
            cred = col2.selectbox(f"Credential", ["LMSW", "LCSW"], index=0 if def_cred=="LMSW" else 1, key=f"cred_{i}")
            sess = st.slider(f"Sessions/Week (Actual)", 10, 30, 20, key=f"sess_{i}")
            cost = st.number_input(f"Hiring Cost", 0.0, value=def_cost, step=500.0, key=f"cost_{i}")
            therapists.append(Therapist(i, f"Therapist {i}", cred, hire_m, sess, cost, start_full))
        elif hire_m > 0:
            cred = col2.selectbox(f"Credential", ["LMSW", "LCSW"], index=0 if def_cred=="LMSW" else 1, key=f"cred_{i}")
            sess = st.slider(f"Sessions/Week (Actual)", 10, 30, 20, key=f"sess_{i}")
            cost = st.number_input(f"Hiring Cost", 0.0, value=def_cost, step=500.0, key=f"cost_{i}")
            therapists.append(Therapist(i, f"Therapist {i}", cred, hire_m, sess, cost, False))

st.sidebar.header("💰 Compensation")
lmsw_pay = st.sidebar.number_input("LMSW Pay/Session", 30.0, value=40.0, step=5.0)
lcsw_pay = st.sidebar.number_input("LCSW Pay/Session", 35.0, value=50.0, step=5.0)
payroll_tax = st.sidebar.slider("Payroll Tax %", 0.0, 20.0, 15.3, 0.1) / 100

st.sidebar.header("👥 Admin Employees")
st.sidebar.markdown("1099 contractors (no payroll tax)")

admin_employees = []

with st.sidebar.expander("Admin 1"):
    admin1_start = st.number_input("Start Month", 0, 36, 0, key="admin1_start")
    if admin1_start > 0:
        admin1_hours = st.number_input("Hours/Week", 0, 40, 20, key="admin1_hours")
        admin1_rate = st.number_input("Hourly Rate ($)", 0.0, 100.0, 25.0, step=5.0, key="admin1_rate")
        admin_employees.append(AdminEmployee(1, "Admin 1", admin1_hours, admin1_rate, admin1_start))

with st.sidebar.expander("Admin 2"):
    admin2_start = st.number_input("Start Month", 0, 36, 0, key="admin2_start")
    if admin2_start > 0:
        admin2_hours = st.number_input("Hours/Week", 0, 40, 15, key="admin2_hours")
        admin2_rate = st.number_input("Hourly Rate ($)", 0.0, 100.0, 30.0, step=5.0, key="admin2_rate")
        admin_employees.append(AdminEmployee(2, "Admin 2", admin2_hours, admin2_rate, admin2_start))

st.sidebar.header("📊 Payer Mix")
use_simple = st.sidebar.checkbox("Simple Model", value=True)

if use_simple:
    avg_ins = st.sidebar.number_input("Avg Insurance Rate", value=100.0)
    self_pct = st.sidebar.slider("Self-Pay %", 0, 50, 10) / 100
    self_rate = st.sidebar.number_input("Self-Pay Rate", value=150.0)
    payer_mix = {
        "Insurance": {"pct": 1 - self_pct, "rate": avg_ins, "delay_days": 45},
        "Self-Pay": {"pct": self_pct, "rate": self_rate, "delay_days": 0}
    }
else:
    st.sidebar.subheader("Detailed Mix")
    payer_mix = {}
    bcbs_p = st.sidebar.slider("BCBS %", 0, 100, 30)
    bcbs_r = st.sidebar.number_input("BCBS Rate", value=105.0)
    payer_mix["BCBS"] = {"pct": bcbs_p/100, "rate": bcbs_r, "delay_days": 30}
    
    aetna_p = st.sidebar.slider("Aetna %", 0, 100, 25)
    aetna_r = st.sidebar.number_input("Aetna Rate", value=110.0)
    payer_mix["Aetna"] = {"pct": aetna_p/100, "rate": aetna_r, "delay_days": 45}
    
    united_p = st.sidebar.slider("United %", 0, 100, 20)
    united_r = st.sidebar.number_input("United Rate", value=95.0)
    payer_mix["United"] = {"pct": united_p/100, "rate": united_r, "delay_days": 60}
    
    medicaid_p = st.sidebar.slider("Medicaid %", 0, 100, 15)
    medicaid_r = st.sidebar.number_input("Medicaid Rate", value=70.0)
    payer_mix["Medicaid"] = {"pct": medicaid_p/100, "rate": medicaid_r, "delay_days": 90}
    
    sp_p = st.sidebar.slider("Self-Pay %", 0, 100, 10)
    sp_r = st.sidebar.number_input("Self-Pay Rate", value=150.0)
    payer_mix["Self-Pay"] = {"pct": sp_p/100, "rate": sp_r, "delay_days": 0}

total_pct = sum(p["pct"] for p in payer_mix.values())
if abs(total_pct - 1.0) > 0.01:
    st.sidebar.error(f"⚠️ Payer mix = {total_pct*100:.1f}%, should be 100%")

weighted_rate = sum(p["pct"] * p["rate"] for p in payer_mix.values())

st.sidebar.header("💻 Tech & Overhead")
ehr_sys = st.sidebar.selectbox("EHR", ["SimplePractice", "TherapyNotes", "Custom"])
if ehr_sys == "Custom":
    ehr_cost_custom = st.sidebar.number_input("EHR Cost/Therapist", value=75.0)
else:
    st.sidebar.info(f"{ehr_sys}: Tiered pricing")
    ehr_cost_custom = None

telehealth = st.sidebar.number_input("Telehealth (monthly)", value=50.0)
other_tech = st.sidebar.number_input("Other Tech (monthly)", value=100.0)
other_overhead = st.sidebar.number_input("Other Overhead (monthly)", value=1500.0)

st.sidebar.header("📄 Billing")
bill_model = st.sidebar.selectbox("Billing", ["Owner Does It", "Billing Service (% of revenue)", "In-House Biller"])
if bill_model == "Billing Service (% of revenue)":
    bill_pct = st.sidebar.slider("Service Fee %", 4.0, 8.0, 6.0) / 100
elif bill_model == "In-House Biller":
    biller_cost = st.sidebar.number_input("Biller Salary", value=4500.0)

st.sidebar.header("🎯 Marketing")
mkt_model = st.sidebar.selectbox("Budget Model", ["Fixed Monthly", "Per Active Therapist", "Per Empty Slot"])
if mkt_model == "Fixed Monthly":
    base_mkt = st.sidebar.number_input("Monthly Budget", value=2000.0)
elif mkt_model == "Per Active Therapist":
    base_mkt = st.sidebar.number_input("Base Budget", value=1000.0)
    mkt_per_therapist = st.sidebar.number_input("Per Therapist", value=500.0)
else:
    mkt_per_slot = st.sidebar.number_input("Per Empty Slot", value=50.0)

cac_target = st.sidebar.number_input("Target CAC", value=150.0)
max_clients_per_month = st.sidebar.number_input("Max Clients Acquirable/Month", value=25, step=5,
    help="Realistic marketing channel capacity")
marketing_lag_weeks = st.sidebar.slider("Lead to Client (weeks)", 0.0, 4.0, 2.0, 0.5,
    help="Time from marketing spend to client starting")

st.sidebar.header("📉 Revenue Loss & Churn")
revenue_loss_pct = st.sidebar.slider("Revenue Loss % (Denials + Bad Debt)", 0, 20, 8,
    help="Combined insurance denials (~5-7%) and bad debt (~2-3%)") / 100

st.sidebar.markdown("**Client Churn** (applied to each cohort based on their tenure)")
churn1 = st.sidebar.slider("Month 1 Churn %", 0, 50, 25,
    help="% of new clients who leave after their first month") / 100
churn2 = st.sidebar.slider("Month 2 Churn %", 0, 50, 15,
    help="% of remaining clients who leave after their second month") / 100
churn3 = st.sidebar.slider("Month 3 Churn %", 0, 50, 10,
    help="% of remaining clients who leave after their third month") / 100
churn_ongoing = st.sidebar.slider("Ongoing Churn %", 0, 20, 5,
    help="% monthly churn after month 3") / 100

st.sidebar.header("🧑‍🎓 Client Behavior")
avg_sess_mo = st.sidebar.number_input("Sessions/Client/Month", value=3.2, step=0.1)
copay_pct = st.sidebar.slider("% with Copay", 0, 100, 20) / 100
avg_copay = st.sidebar.number_input("Avg Copay", value=25.0)
cc_fee = st.sidebar.slider("CC Fee %", 0.0, 5.0, 2.9, 0.1) / 100

st.sidebar.header("📅 Seasonality")
apply_seasonality = st.sidebar.checkbox("Apply Seasonality", value=False,
    help="December: -30%, September: +20%, Summer: -15%")

st.sidebar.header("⚙️ Simulation")
sim_months = st.sidebar.number_input("Months", 12, 60, value=24)

# Input validation
assert cac_target > 0, "CAC must be positive"
assert avg_sess_mo > 0, "Sessions/client must be positive"

# ==========================================
# SIMULATION
# ==========================================

# Calculate owner's starting caseload (KEEPING OWNER FULL FROM DAY 1)
owner = therapists[0]
owner_capacity_sessions_month1 = owner.sessions_per_week_target * WEEKS_PER_MONTH
owner_initial_clients = owner_capacity_sessions_month1 / avg_sess_mo if avg_sess_mo > 0 else 0

# Initialize tracking variables
monthly_data = []
revenue_history = {p: [] for p in payer_mix.keys()}
months_in_op = 0
cumulative_hiring = 0.0
hired_tracker = set()
marketing_pipeline = []  # Track leads in pipeline

# FIXED: Track client cohorts instead of just total numbers
therapist_cohorts = defaultdict(list)  # therapist_id -> list of ClientCohort objects

# Initialize owner's starting cohort
if owner_initial_clients > 0:
    therapist_cohorts[0].append(ClientCohort(
        start_month=1,  # FIX #1: Start at month 1, not 0
        initial_count=owner_initial_clients,
        current_count=owner_initial_clients,
        therapist_id=0
    ))

# FIX #21: Initialize existing employee therapists with full caseloads if requested
for therapist in therapists:
    if therapist.id != 0 and therapist.hire_month == 0 and therapist.start_full:
        initial_capacity = therapist.sessions_per_week_target * WEEKS_PER_MONTH
        initial_clients = initial_capacity / avg_sess_mo if avg_sess_mo > 0 else 0
        therapist_cohorts[therapist.id].append(ClientCohort(
            start_month=1,
            initial_count=initial_clients,
            current_count=initial_clients,
            therapist_id=therapist.id
        ))

for month in range(1, sim_months + 1):
    month_data = {"month": month}
    months_in_op += 1
    
    # Get seasonality factors
    demand_factor, attendance_factor = get_seasonality_factor(month, apply_seasonality)
    
    # Calculate active therapists and capacity
    active_ther = [t for t in therapists if t.is_active(month)]
    
    # Calculate capacity for each therapist
    therapist_capacity = {}
    for t in therapists:
        if t.is_active(month):
            sess = t.get_sessions_per_month(month, WEEKS_PER_MONTH, ramp_speed)
            therapist_capacity[t.id] = sess / avg_sess_mo if avg_sess_mo > 0 else 0
        else:
            therapist_capacity[t.id] = 0
    
    total_cap_sess = sum(t.get_sessions_per_month(month, WEEKS_PER_MONTH, ramp_speed) for t in therapists if t.is_active(month))
    
    # Apply attendance seasonality to capacity
    total_cap_sess *= attendance_factor
    total_cap_clients = total_cap_sess / avg_sess_mo if avg_sess_mo > 0 else 0
    
    lmsw_count = len([t for t in active_ther if t.credential == "LMSW" and t.id != 0])
    lcsw_count = len([t for t in active_ther if t.credential == "LCSW"])
    
    month_data["active_therapists"] = len(active_ther)
    month_data["active_lmsw"] = lmsw_count
    month_data["active_lcsw"] = lcsw_count
    month_data["total_capacity_sessions"] = total_cap_sess
    month_data["total_capacity_clients"] = total_cap_clients
    
    # FIXED: Apply churn to each cohort based on its age
    total_churned = 0
    for t_id, cohorts in therapist_cohorts.items():
        for cohort in cohorts:
            if cohort.current_count > 0:
                churned = apply_cohort_churn(cohort, month, churn1, churn2, churn3, churn_ongoing)
                cohort.current_count = max(0, cohort.current_count - churned)
                total_churned += churned
    
    # Calculate total active clients across all cohorts
    therapist_clients = defaultdict(float)
    for t_id, cohorts in therapist_cohorts.items():
        therapist_clients[t_id] = sum(c.current_count for c in cohorts)
    
    active_clients = sum(therapist_clients.values())
    
    # Marketing budget calculation
    if mkt_model == "Fixed Monthly":
        mkt_budget = base_mkt
    elif mkt_model == "Per Active Therapist":
        mkt_budget = base_mkt + (len(active_ther) * mkt_per_therapist)
    else:  # Per Empty Slot
        empty = max(0, total_cap_clients - active_clients)
        mkt_budget = empty * mkt_per_slot
    
    # Apply demand seasonality to marketing effectiveness
    effective_mkt_budget = mkt_budget * demand_factor
    
    # Process marketing pipeline (with lag)
    marketing_lag_months = marketing_lag_weeks / 4.33
    new_clients_total = 0
    
    if month > marketing_lag_months and len(marketing_pipeline) > int(marketing_lag_months):
        new_clients_total = marketing_pipeline.pop(0)
    
    # Allocate new clients to therapists with capacity and create new cohorts
    remaining_new = new_clients_total
    for t in therapists:
        if t.is_active(month) and remaining_new > 0:
            t_cap = therapist_capacity[t.id]
            t_current = therapist_clients[t.id]
            t_space = max(0, t_cap - t_current)
            
            if t_space > 0:
                allocated = min(t_space, remaining_new)
                # Create new cohort for these clients
                therapist_cohorts[t.id].append(ClientCohort(
                    start_month=month,
                    initial_count=allocated,
                    current_count=allocated,
                    therapist_id=t.id
                ))
                remaining_new -= allocated
    
    # Calculate new clients to add to pipeline
    cap_avail = max(0, total_cap_clients - sum(therapist_clients.values()))
    max_new_from_budget = effective_mkt_budget / cac_target if cac_target > 0 else 0
    
    # FIX #8: Apply monthly acquisition capacity constraint
    new_to_pipeline = min(max_new_from_budget, cap_avail, max_clients_per_month)
    
    # Track actual marketing spent (only what's needed)
    actual_mkt_spent = new_to_pipeline * cac_target
    marketing_savings = mkt_budget - actual_mkt_spent  # This becomes profit
    
    marketing_pipeline.append(new_to_pipeline)
    
    active_clients = sum(therapist_clients.values())
    
    month_data["churned_clients"] = total_churned
    month_data["new_clients"] = new_clients_total
    month_data["active_clients"] = active_clients
    month_data["capacity_utilization"] = (active_clients / total_cap_clients * 100) if total_cap_clients > 0 else 0
    month_data["marketing_budget"] = mkt_budget
    month_data["marketing_spent"] = actual_mkt_spent
    month_data["marketing_saved"] = marketing_savings
    
    # Sessions and revenue - based on each therapist's actual clients
    # FIX #4: Cap sessions at therapist capacity
    therapist_sessions = {}
    for t in therapists:
        if t.is_active(month):
            t_clients = therapist_clients[t.id]
            t_capacity_sessions = t.get_sessions_per_month(month, WEEKS_PER_MONTH, ramp_speed) * attendance_factor
            t_desired_sessions = t_clients * avg_sess_mo * attendance_factor
            # Cap at therapist capacity
            t_actual_sessions = min(t_desired_sessions, t_capacity_sessions)
            therapist_sessions[t.id] = t_actual_sessions
    
    total_sessions = sum(therapist_sessions.values())
    
    month_data["scheduled_sessions"] = total_sessions
    month_data["billable_sessions"] = total_sessions  # No cancel/no-show reductions
    
    # FIX #2: Revenue with copay correction
    # Insurance pays (rate - copay), patient pays copay separately
    rev_by_payer = {}
    copay_rev_total = 0
    
    for pname, pinfo in payer_mix.items():
        p_sess = total_sessions * pinfo["pct"]
        
        if pname != "Self-Pay":
            # Insurance: rate minus copay
            insurance_portion = p_sess * (pinfo["rate"] - avg_copay)
            copay_portion = p_sess * avg_copay * copay_pct  # Only copay_pct of patients have copays
            
            # Apply revenue loss to insurance portion only
            insurance_revenue = insurance_portion * (1 - revenue_loss_pct)
            
            rev_by_payer[pname] = insurance_revenue
            copay_rev_total += copay_portion
        else:
            # Self-pay: full rate, no copay
            p_rev = p_sess * pinfo["rate"]
            rev_by_payer[pname] = p_rev
        
        revenue_history[pname].append({"month": month, "revenue": rev_by_payer[pname]})
    
    cc_fees = copay_rev_total * cc_fee
    total_rev = sum(rev_by_payer.values()) + copay_rev_total - cc_fees
    
    month_data["copay_revenue"] = copay_rev_total
    month_data["cc_fees"] = cc_fees
    month_data["revenue_earned"] = total_rev
    
    # Track owner revenue separately
    owner_sessions = therapist_sessions.get(0, 0)
    owner_revenue_portion = (owner_sessions / total_sessions * total_rev) if total_sessions > 0 else 0
    month_data["owner_revenue"] = owner_revenue_portion
    
    # Cash collections
    collections = calculate_collections(revenue_history, month, payer_mix, copay_rev_total, cc_fees)
    month_data["collections"] = collections
    
    # FIX #9: Therapist costs - Owner has $0 cost
    ther_cost_total = 0.0
    
    for t in therapists:
        if t.is_active(month) and t.id != 0:  # Skip owner
            t_sessions = therapist_sessions.get(t.id, 0)
            pay_rate = lmsw_pay if t.credential == "LMSW" else lcsw_pay
            gross_pay = t_sessions * pay_rate
            # Total cost includes payroll tax
            ther_cost_total += gross_pay * (1 + payroll_tax)
    
    month_data["therapist_costs"] = ther_cost_total
    
    # FIX #10: Supervision costs - only external cash costs
    sup_cost, own_sup, ext_sup_count = calculate_supervision_costs(lmsw_count, WEEKS_PER_MONTH)
    
    month_data["supervision_cost"] = sup_cost
    month_data["owner_supervised_count"] = own_sup
    month_data["external_supervised_count"] = ext_sup_count
    
    # Tech costs
    cost_per_ther_ehr = get_ehr_cost(len(active_ther), ehr_sys, ehr_cost_custom)
    ehr_monthly = cost_per_ther_ehr * len(active_ther)
    tech_cost = ehr_monthly + telehealth + other_tech
    
    # Billing costs
    if bill_model == "Owner Does It":
        bill_cost = 0.0
    elif bill_model == "Billing Service (% of revenue)":
        bill_cost = total_rev * bill_pct
    else:
        bill_cost = biller_cost
    
    # FIX #12: Admin employee costs
    admin_cost = 0.0
    for admin in admin_employees:
        if admin.is_active(month):
            admin_cost += admin.get_monthly_cost(WEEKS_PER_MONTH)
    
    # One-time hiring costs
    hiring_this_mo = 0.0
    for t in active_ther:
        if t.id not in hired_tracker and t.id != 0:
            if t.hire_month == month:
                hiring_this_mo += t.one_time_hiring_cost
                hired_tracker.add(t.id)
                cumulative_hiring += t.one_time_hiring_cost
    
    month_data["one_time_hiring_costs"] = hiring_this_mo
    month_data["cumulative_hiring_costs"] = cumulative_hiring
    month_data["admin_cost"] = admin_cost
    
    # Total costs (owner has $0 cost now)
    total_costs = (ther_cost_total + sup_cost + tech_cost + bill_cost + 
                  actual_mkt_spent + other_overhead + admin_cost + hiring_this_mo)
    
    month_data["tech_cost"] = tech_cost
    month_data["billing_cost"] = bill_cost
    month_data["other_overhead"] = other_overhead
    month_data["total_costs"] = total_costs
    
    # Profit (includes marketing savings)
    profit_acc = total_rev - total_costs + marketing_savings
    cash_flow = collections - total_costs + marketing_savings
    
    month_data["profit_accrual"] = profit_acc
    month_data["cash_flow"] = cash_flow
    
    # Cash balance
    if month == 1:
        month_data["cash_balance"] = starting_cash_balance + cash_flow
    else:
        month_data["cash_balance"] = monthly_data[-1]["cash_balance"] + cash_flow
    
    # Store therapist client counts for display
    month_data["therapist_clients"] = dict(therapist_clients)
    
    # Store cohort details for debugging
    month_data["total_cohorts"] = sum(len(cohorts) for cohorts in therapist_cohorts.values())
    month_data["cohort_details"] = {
        t_id: [(c.start_month, c.current_count, month - c.start_month + 1) for c in cohorts]
        for t_id, cohorts in therapist_cohorts.items()
    }
    
    monthly_data.append(month_data)

df = pd.DataFrame(monthly_data)

# ==========================================
# DISPLAY
# ==========================================

st.header("📊 Practice Overview")

final = df.iloc[-1]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Active Clients", f"{final['active_clients']:.0f}")
c2.metric("Capacity", f"{final['total_capacity_clients']:.0f}")
c3.metric("Utilization", f"{final['capacity_utilization']:.1f}%")
c4.metric("Monthly Profit", f"${final['profit_accrual']:,.0f}")
c5.metric("Cash Balance", f"${final['cash_balance']:,.0f}")

# Add churn impact display
st.info(f"📊 **Churn Model Active**: Each client cohort experiences {churn1*100:.0f}% churn in month 1, "
        f"{churn2*100:.0f}% in month 2, {churn3*100:.0f}% in month 3, then {churn_ongoing*100:.0f}% ongoing. "
        f"Total cohorts tracked: {final['total_cohorts']}")

st.markdown("---")

# FIX #11: ANNUAL EXPENSE SUMMARY
st.header("💰 Annual Expense Summary")

for yr in range(1, (sim_months // 12) + 2):
    start = (yr - 1) * 12 + 1
    end = min(yr * 12, sim_months)
    
    if start <= sim_months:
        yr_data = df[(df['month'] >= start) & (df['month'] <= end)]
        
        if len(yr_data) > 0:
            therapist_salaries = yr_data['therapist_costs'].sum()
            marketing = yr_data['marketing_spent'].sum()
            tech = yr_data['tech_cost'].sum()
            overhead = yr_data['other_overhead'].sum() + yr_data['one_time_hiring_costs'].sum()
            supervision = yr_data['supervision_cost'].sum()
            billing = yr_data['billing_cost'].sum()
            admin = yr_data['admin_cost'].sum()
            total_expenses = therapist_salaries + marketing + tech + overhead + supervision + billing + admin
            
            mo_in_yr = len(yr_data)
            year_label = f"Year {yr}" + (f" ({mo_in_yr}mo)" if mo_in_yr < 12 else "")
            
            st.subheader(year_label)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Compensation & Staffing:**
                - Therapist Salaries: ${therapist_salaries:,.0f}
                - Admin Employees: ${admin:,.0f}
                - External Supervision: ${supervision:,.0f}
                """)
            
            with col2:
                st.markdown(f"""
                **Operating Costs:**
                - Marketing: ${marketing:,.0f}
                - Tech (EHR, telehealth, other): ${tech:,.0f}
                - Billing: ${billing:,.0f}
                - Overhead (insurance, legal, hiring): ${overhead:,.0f}
                """)
            
            st.metric(f"**{year_label} Total Expenses**", f"${total_expenses:,.0f}")
            st.markdown("---")

# OWNER COMPENSATION BY YEAR
st.header("💰 Owner Total Compensation by Year")

years_comp = []
for yr in range(1, (sim_months // 12) + 2):
    start = (yr - 1) * 12 + 1
    end = min(yr * 12, sim_months)
    
    if start <= sim_months:
        yr_data = df[(df['month'] >= start) & (df['month'] <= end)]
        mo_in_yr = len(yr_data)
        
        # Owner revenue (clinical contribution)
        owner_revenue = yr_data['owner_revenue'].sum()
        # All profit (owner has no salary)
        biz_profit = yr_data['profit_accrual'].sum()
        total_comp = biz_profit
        
        years_comp.append({
            'year': f"Year {yr}" + (f" ({mo_in_yr}mo)" if mo_in_yr < 12 else ""),
            'owner_revenue': owner_revenue,
            'profit': biz_profit,
            'total': total_comp
        })

for yc in years_comp:
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{yc['year']} - Owner Clinical Revenue", f"${yc['owner_revenue']:,.0f}")
    col2.metric(f"{yc['year']} - Business Profit", f"${yc['profit']:,.0f}")
    col3.metric(f"{yc['year']} - Total Compensation", f"${yc['total']:,.0f}")

st.markdown("---")

# CASH & RUNWAY
st.header("💵 Cash Position & Runway")

avg_cf = df['cash_flow'].mean()
runway, color, status = calculate_runway(final['cash_balance'], avg_cf)

col1, col2, col3 = st.columns(3)
col1.metric("Cash Balance", f"${final['cash_balance']:,.0f}")
col2.metric("Avg Monthly Cash Flow", f"${avg_cf:,.0f}")
col3.metric("Runway", status)

if color == 'red':
    st.error("⚠️ Cash position critical - consider line of credit")
elif color == 'orange':
    st.warning("⚠️ Limited cash runway - monitor closely")
else:
    st.success("✅ Healthy cash position")

st.markdown("---")

# HIRING DECISION CALCULATOR
st.header("👥 Hiring Decision Calculator")

util_pct = final['capacity_utilization']
empty_slots = final['total_capacity_clients'] - final['active_clients']

st.markdown(f"""
**Current State:**
- Capacity Utilization: {util_pct:.1f}%
- Empty Client Slots: {empty_slots:.0f}
""")

if util_pct > 85:
    st.success("✅ Recommendation: **HIRE NOW** - at capacity")
elif util_pct > 70:
    st.info("✓ Prepare to hire - approaching capacity")
else:
    st.warning("⚠️ Fix marketing first - low utilization")

st.markdown("### If You Hire an LMSW Next Month:")

# Calculate supervision cost for new LMSW
hours_per_month = SUPERVISION_HOURS_PER_LMSW_WEEK * WEEKS_PER_MONTH
if lmsw_count < OWNER_SUPERVISION_CAPACITY:
    supervision_cost_new = 0  # Owner supervision has no cost
else:
    supervision_cost_new = hours_per_month * (EXTERNAL_SUPERVISION_HOURLY / LMSW_SHARE_SUPERVISION)

proj = project_new_hire(
    "LMSW", 20, lmsw_pay, lcsw_pay, weighted_rate,
    revenue_loss_pct, payroll_tax,
    500, supervision_cost_new, 3000, WEEKS_PER_MONTH, ramp_speed, 12
)

breakeven_month = proj[proj['cumulative_profit'] >= 0]['month'].min() if any(proj['cumulative_profit'] >= 0) else None

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    - One-Time Hiring Cost: $3,000
    - Ramp Period: {5 if ramp_speed == 'Medium' else 6 if ramp_speed == 'Slow' else 4} months
    - Monthly Contribution (full): ${proj.iloc[-1]['monthly_profit']:,.0f}
    - Break-Even Month: {f"Month {breakeven_month}" if breakeven_month else "Not within 12 months"}
    """)

with col2:
    cash_needed = 3000 + abs(proj[proj['cumulative_profit'] < 0]['cumulative_profit'].min()) if any(proj['cumulative_profit'] < 0) else 3000
    st.markdown(f"""
    - Cash Required: ${cash_needed:,.0f}
    - Current Cash: ${final['cash_balance']:,.0f}
    - **Decision**: {"✅ Can afford" if final['cash_balance'] >= cash_needed else f"⚠️ Need ${cash_needed - final['cash_balance']:,.0f} more"}
    """)

st.dataframe(proj.style.format({
    'capacity_%': '{:.0f}%',
    'sessions': '{:.1f}',
    'revenue': '${:,.0f}',
    'costs': '${:,.0f}',
    'monthly_profit': '${:,.0f}',
    'cumulative_profit': '${:,.0f}'
}), use_container_width=True)

st.markdown("---")

# PER-THERAPIST PERFORMANCE CARDS
st.header("👥 Individual Therapist Performance")

st.markdown("**Detailed profit & loss for each therapist by year**")

# Build therapist performance data using actual client counts
therapist_performance = {}

for therapist in therapists:
    if therapist.hire_month == 0 and therapist.id != 0:
        continue  # Skip disabled therapist slots
    
    therapist_performance[therapist.name] = {}
    
    for year_num in range(1, (sim_months // 12) + 2):
        start_month = (year_num - 1) * 12 + 1
        end_month = min(year_num * 12, sim_months)
        
        if start_month <= sim_months:
            months_active = []
            for m in range(start_month, end_month + 1):
                if therapist.is_active(m):
                    months_active.append(m)
            
            if len(months_active) > 0:
                # Get actual sessions from therapist's client panel
                total_sessions = 0
                for m in months_active:
                    month_idx = m - 1
                    if month_idx < len(df):
                        t_clients = df.iloc[month_idx]['therapist_clients'].get(therapist.id, 0)
                        t_sessions = t_clients * avg_sess_mo
                        total_sessions += t_sessions
                
                revenue = total_sessions * weighted_rate * (1 - revenue_loss_pct)
                
                # Direct costs - Owner has $0 cost
                if therapist.id == 0:
                    direct_pay = 0
                    payroll_tax_amount = 0
                else:
                    pay_rate = lmsw_pay if therapist.credential == "LMSW" else lcsw_pay
                    direct_pay = total_sessions * pay_rate * (1 + payroll_tax)
                    payroll_tax_amount = 0  # Included in direct_pay
                
                gross_margin = revenue - direct_pay - payroll_tax_amount
                gross_margin_pct = (gross_margin / revenue * 100) if revenue > 0 else 0
                
                # Allocated costs
                year_data = df[(df['month'] >= start_month) & (df['month'] <= end_month)]
                avg_therapists = year_data['active_therapists'].mean()
                
                # Allocate marketing based on capacity
                total_marketing = year_data['marketing_spent'].sum()
                total_capacity = year_data['total_capacity_sessions'].sum()
                therapist_capacity = sum([therapist.get_sessions_per_month(m, WEEKS_PER_MONTH, ramp_speed) for m in months_active])
                therapist_capacity_pct = therapist_capacity / total_capacity if total_capacity > 0 else 0
                marketing_allocated = total_marketing * therapist_capacity_pct
                
                # Other allocations
                overhead_per_month = (other_overhead + telehealth + other_tech) / avg_therapists if avg_therapists > 0 else 0
                overhead_allocated = overhead_per_month * len(months_active)
                
                # Allocate actual supervision costs from year data
                total_supervision = year_data['supervision_cost'].sum()
                total_lmsw_months = year_data['active_lmsw'].sum()
                if therapist.credential == "LMSW" and total_lmsw_months > 0:
                    supervision_allocated = (len(months_active) / total_lmsw_months) * total_supervision
                else:
                    supervision_allocated = 0
                
                tech_allocated = cost_per_ther_ehr * len(months_active)
                
                total_allocated = overhead_allocated + supervision_allocated + marketing_allocated + tech_allocated
                
                net_margin = gross_margin - total_allocated
                net_margin_pct = (net_margin / revenue * 100) if revenue > 0 else 0
                
                therapist_performance[therapist.name][f"Year {year_num}"] = {
                    'credential': therapist.credential,
                    'months_active': len(months_active),
                    'sessions': billable_sessions,
                    'revenue': revenue,
                    'direct_pay': direct_pay,
                    'payroll_tax': payroll_tax_amount,
                    'gross_margin': gross_margin,
                    'gross_margin_pct': gross_margin_pct,
                    'overhead': overhead_allocated,
                    'supervision': supervision_allocated,
                    'marketing': marketing_allocated,
                    'tech': tech_allocated,
                    'total_allocated': total_allocated,
                    'net_margin': net_margin,
                    'net_margin_pct': net_margin_pct
                }

# Display therapist cards
for therapist_name, data in therapist_performance.items():
    if not data:
        continue
    
    st.markdown(f"### {therapist_name}")
    
    # Year columns
    year_cols = st.columns(len(data))
    
    for idx, (year_label, year_info) in enumerate(data.items()):
        with year_cols[idx]:
            st.markdown(f"**{year_label}** ({year_info['credential']}, {year_info['months_active']}mo)")
            
            st.metric("Revenue", f"${year_info['revenue']:,.0f}")
            st.metric("Gross Margin", f"${year_info['gross_margin']:,.0f}", 
                     delta=f"{year_info['gross_margin_pct']:.1f}%")
            
            with st.expander("Cost Detail"):
                st.markdown(f"""
                **Direct Costs:**
                - {'Salary' if therapist_name == 'Owner' else 'Pay'}: ${year_info['direct_pay']:,.0f}
                {f"- Payroll Tax: ${year_info['payroll_tax']:,.0f}" if year_info['payroll_tax'] > 0 else ""}
                
                **Allocated Costs:**
                - Overhead: ${year_info['overhead']:,.0f}
                - Supervision: ${year_info['supervision']:,.0f}
                - Marketing: ${year_info['marketing']:,.0f}
                - Technology: ${year_info['tech']:,.0f}
                - **Total**: ${year_info['total_allocated']:,.0f}
                """)
            
            # Net margin with color coding
            if year_info['net_margin'] > 0:
                st.success(f"**Net Margin:** ${year_info['net_margin']:,.0f} ({year_info['net_margin_pct']:.1f}%)")
            else:
                st.error(f"**Net Margin:** ${year_info['net_margin']:,.0f} ({year_info['net_margin_pct']:.1f}%)")
            
            st.markdown(f"_Sessions: {year_info['sessions']:.0f}_")
    
    st.markdown("---")

# BREAK-EVEN ANALYSIS
st.header("⚖️ Break-Even Sessions Per Therapist")

lmsw_fixed = 500
lcsw_fixed = 500

lmsw_be = calculate_breakeven_sessions("LMSW", lmsw_pay, lcsw_pay, weighted_rate, 
                                      revenue_loss_pct, lmsw_fixed, payroll_tax, WEEKS_PER_MONTH)
lcsw_be = calculate_breakeven_sessions("LCSW", lmsw_pay, lcsw_pay, weighted_rate,
                                      revenue_loss_pct, lcsw_fixed, payroll_tax, WEEKS_PER_MONTH)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### LMSW")
    if lmsw_be['monthly'] < float('inf'):
        st.metric("Monthly", f"{lmsw_be['monthly']:.1f}")
        st.metric("Weekly", f"{lmsw_be['weekly']:.1f}")
    else:
        st.metric("Monthly", "Never breaks even")
        st.metric("Weekly", "Never breaks even")
    st.markdown(f"Contribution: ${lmsw_be['contribution']:.2f}/session")

with col2:
    st.markdown("### LCSW")
    if lcsw_be['monthly'] < float('inf'):
        st.metric("Monthly", f"{lcsw_be['monthly']:.1f}")
        st.metric("Weekly", f"{lcsw_be['weekly']:.1f}")
    else:
        st.metric("Monthly", "Never breaks even")
        st.metric("Weekly", "Never breaks even")
    st.markdown(f"Contribution: ${lcsw_be['contribution']:.2f}/session")

st.markdown("---")

# SCENARIO ANALYSIS
st.header("📊 Scenario Analysis")

st.markdown("**Comparing outcomes under different assumptions:**")

scenarios = {
    'Pessimistic': {'util_adj': -0.15, 'churn_adj': 0.10},
    'Base Case': {'util_adj': 0.0, 'churn_adj': 0.0},
    'Optimistic': {'util_adj': 0.10, 'churn_adj': -0.05}
}

scenario_results = []
for name, params in scenarios.items():
    if name == 'Base Case':
        total_comp = sum(yc['total'] for yc in years_comp if 'Year 1' in yc['year'])
        scenario_results.append({
            'Scenario': name,
            'Year 1 Comp': total_comp,
            'Cash Low': df['cash_balance'].min(),
            'Final Cash': final['cash_balance']
        })
    else:
        # Simplified estimation
        util_factor = 1 + params['util_adj']
        churn_factor = 1 + params['churn_adj']
        total_comp_est = sum(yc['total'] for yc in years_comp if 'Year 1' in yc['year']) * util_factor
        cash_est = final['cash_balance'] * util_factor
        scenario_results.append({
            'Scenario': name,
            'Year 1 Comp': total_comp_est,
            'Cash Low': df['cash_balance'].min() * util_factor,
            'Final Cash': cash_est
        })

st.dataframe(pd.DataFrame(scenario_results).style.format({
    'Year 1 Comp': '${:,.0f}',
    'Cash Low': '${:,.0f}',
    'Final Cash': '${:,.0f}'
}), use_container_width=True)

st.markdown("---")

# CHARTS
st.header("📈 Visual Analytics")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Clients vs Capacity with churn visualization
ax1.plot(df["month"], df["active_clients"], 'o-', linewidth=2, label='Active Clients')
ax1.plot(df["month"], df["total_capacity_clients"], '--', linewidth=2, label='Capacity')
ax1.bar(df["month"], df["churned_clients"], alpha=0.3, color='red', label='Churned')
ax1.fill_between(df["month"], df["active_clients"], df["total_capacity_clients"], alpha=0.2)
ax1.set_title("Clients vs Capacity (with Churn)")
ax1.set_xlabel("Month")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Chart 2: Revenue vs Collections vs Costs
ax2.plot(df["month"], df["revenue_earned"], 'o-', linewidth=2, label='Revenue')
ax2.plot(df["month"], df["collections"], 's-', linewidth=2, label='Collections')
ax2.plot(df["month"], df["total_costs"], '^-', linewidth=2, label='Costs')
ax2.set_title("Revenue, Collections & Costs")
ax2.set_xlabel("Month")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Chart 3: Cash Balance
ax3.plot(df["month"], df["cash_balance"], 'o-', linewidth=2, color='green')
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax3.fill_between(df["month"], 0, df["cash_balance"], where=df["cash_balance"]<0, color='red', alpha=0.2)
ax3.fill_between(df["month"], 0, df["cash_balance"], where=df["cash_balance"]>=0, color='green', alpha=0.2)
ax3.set_title("Cash Balance")
ax3.set_xlabel("Month")
ax3.grid(True, alpha=0.3)

# Chart 4: Cost Breakdown
ax4.stackplot(df["month"], 
              df["therapist_costs"], 
              df["owner_total_cost"],
              df["supervision_cost"],
              df["tech_cost"], 
              df["marketing_spent"], 
              df["other_overhead"],
              labels=['Therapist', 'Owner', 'Supervision', 'Tech', 'Marketing', 'Other'], 
              alpha=0.8)
ax4.set_title("Cost Breakdown")
ax4.set_xlabel("Month")
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# DETAILED DATA
with st.expander("📋 Detailed Monthly Data"):
    display_df = df[["month", "active_clients", "total_capacity_clients", "churned_clients", "new_clients", 
                     "active_therapists", "revenue_earned", "collections", "total_costs", "profit_accrual", 
                     "cash_flow", "cash_balance", "marketing_budget", "marketing_spent", "marketing_saved"]]
    st.dataframe(display_df.style.format({
        "revenue_earned": "${:,.0f}", "collections": "${:,.0f}", "total_costs": "${:,.0f}",
        "profit_accrual": "${:,.0f}", "cash_flow": "${:,.0f}", "cash_balance": "${:,.0f}",
        "marketing_budget": "${:,.0f}", "marketing_spent": "${:,.0f}", "marketing_saved": "${:,.0f}",
        "active_clients": "{:.1f}", "total_capacity_clients": "{:.1f}", 
        "churned_clients": "{:.1f}", "new_clients": "{:.1f}"
    }), use_container_width=True)

with st.expander("💰 Cost Breakdown Detail"):
    cost_df = df[["month", "therapist_costs", "supervision_cost", "admin_cost",
                  "tech_cost", "billing_cost", "marketing_spent", "one_time_hiring_costs", "other_overhead", "total_costs"]]
    st.dataframe(cost_df.style.format({
        "therapist_costs": "${:,.0f}", "supervision_cost": "${:,.0f}", "admin_cost": "${:,.0f}",
        "tech_cost": "${:,.0f}", "billing_cost": "${:,.0f}", 
        "marketing_spent": "${:,.0f}", "one_time_hiring_costs": "${:,.0f}", "other_overhead": "${:,.0f}", 
        "total_costs": "${:,.0f}"
    }), use_container_width=True)

# SUPERVISION DETAIL
if df['active_lmsw'].max() > 0:
    with st.expander("🎓 Supervision Detail"):
        st.markdown(f"""
        **Final Month:**
        - LMSW Count: {final['active_lmsw']}
        - Owner Supervising: {final['owner_supervised_count']} (no cost)
        - External: {final['external_supervised_count']}
        - Monthly Cost: ${final['supervision_cost']:,.0f}
        
        **Cost Calculation:**
        - Owner supervision (up to 3): $0 (no cost)
        - External: ${EXTERNAL_SUPERVISION_HOURLY:.0f}/hr ÷ {LMSW_SHARE_SUPERVISION} LMSWs = ${EXTERNAL_SUPERVISION_HOURLY/LMSW_SHARE_SUPERVISION:.0f}/LMSW/hr
        """)

# COHORT DEBUGGING
with st.expander("🔍 Client Cohort Details (Debug)"):
    st.markdown("**Active cohorts by therapist in final month:**")
    for t_id, cohorts in final['cohort_details'].items():
        if cohorts:
            therapist_name = next((t.name for t in therapists if t.id == t_id), f"Therapist {t_id}")
            st.markdown(f"**{therapist_name}:**")
            for cohort in cohorts:
                st.markdown(f"- Started Month {cohort[0]}: {cohort[1]:.1f} clients (age: {cohort[2]} months)")

# EXPORT
st.markdown("---")
st.header("📥 Export")
csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, f"practice_model_v44_{sim_months}mo.csv", "text/csv")

st.success("✅ Model V4.4 Complete - Churn model fixed!")
