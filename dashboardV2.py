import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

st.set_page_config(page_title="Therapy Practice Financial Model V4", layout="wide")
st.title("🧠 Therapy Practice Financial Model V4")
st.markdown("Enhanced decision support for practice owners")

# ==========================================
# CONSTANTS
# ==========================================

MONTHS_PER_YEAR = 12
OWNER_SUPERVISION_CAPACITY = 3  # Max LMSW owner can supervise
EXTERNAL_SUPERVISION_COST_PER_LMSW = 200.0  # $/month
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
    sessions_per_week_target: int
    utilization_rate: float
    one_time_hiring_cost: float = 0.0
    
    def is_active(self, current_month: int) -> bool:
        """Therapist is active immediately when hired (already credentialed)"""
        return current_month >= self.hire_month
    
    def get_capacity_percentage(self, current_month: int) -> float:
        """Ramp up over 5 months at 20% increments"""
        if not self.is_active(current_month):
            return 0.0
        
        months_since_hire = current_month - self.hire_month
        
        if months_since_hire == 0:
            return 0.20
        elif months_since_hire == 1:
            return 0.40
        elif months_since_hire == 2:
            return 0.60
        elif months_since_hire == 3:
            return 0.80
        else:
            return 1.0
    
    def get_sessions_per_month(self, current_month: int) -> float:
        """Calculate actual sessions per month"""
        capacity_pct = self.get_capacity_percentage(current_month)
        return self.sessions_per_week_target * self.utilization_rate * WEEKS_PER_MONTH * capacity_pct

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_clv(
    avg_sessions_per_month: float,
    month1_churn: float,
    month2_churn: float,
    month3_churn: float,
    ongoing_churn: float,
    contribution_margin_per_session: float
) -> float:
    """Calculate Client Lifetime Value using survival curve"""
    survival_rate = 1.0
    total_sessions = 0.0
    churn_rates = [month1_churn, month2_churn, month3_churn] + [ongoing_churn] * 5
    
    for churn_rate in churn_rates:
        survival_rate *= (1 - churn_rate)
        total_sessions += survival_rate * avg_sessions_per_month
    
    return total_sessions * contribution_margin_per_session

def calculate_max_cac(clv: float, target_margin_pct: float, target_roi_pct: float) -> Tuple[float, float, float]:
    """Calculate maximum affordable CAC under different constraints"""
    max_by_margin = clv * (1 - target_margin_pct / 100)
    max_by_roi = clv / (1 + target_roi_pct / 100)
    conservative_max = min(max_by_margin, max_by_roi)
    return max_by_margin, max_by_roi, conservative_max

def calculate_runway(cash_balance: float, avg_cash_flow: float) -> Tuple[float, str, str]:
    """Calculate cash runway and status"""
    if avg_cash_flow >= 0:
        return float('inf'), 'green', 'Positive Cash Flow'
    
    runway = abs(cash_balance / avg_cash_flow)
    
    if cash_balance < 0:
        color = 'red'
        status = 'Negative Balance'
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

def calculate_supervision_costs(num_lmsw: int, owner_session_value: float) -> Tuple[float, float, int, int]:
    """Calculate supervision costs with leverage model"""
    if num_lmsw == 0:
        return 0.0, 0.0, 0, 0
    
    if num_lmsw <= OWNER_SUPERVISION_CAPACITY:
        supervision_cost = num_lmsw * owner_session_value
        external_cost = 0.0
        owner_supervised = num_lmsw
        external_supervised = 0
    else:
        owner_supervision_cost = OWNER_SUPERVISION_CAPACITY * owner_session_value
        external_supervisees = num_lmsw - OWNER_SUPERVISION_CAPACITY
        external_cost = external_supervisees * EXTERNAL_SUPERVISION_COST_PER_LMSW
        supervision_cost = owner_supervision_cost + external_cost
        owner_supervised = OWNER_SUPERVISION_CAPACITY
        external_supervised = external_supervisees
    
    return supervision_cost, external_cost, owner_supervised, external_supervised

def calculate_breakeven_sessions(
    credential: str,
    lmsw_pay: float,
    lcsw_pay: float,
    weighted_rate: float,
    no_show_rate: float,
    cancellation_rate: float,
    monthly_fixed: float,
    payroll_tax: float
) -> Dict[str, float]:
    """Calculate break-even sessions"""
    pay = lmsw_pay if credential == "LMSW" else lcsw_pay
    effective_revenue = weighted_rate * (1 - cancellation_rate) * (1 - no_show_rate)
    variable_cost = pay * (1 + payroll_tax)
    contribution = effective_revenue - variable_cost
    
    if contribution <= 0:
        return {'monthly': float('inf'), 'weekly': float('inf'), 'contribution': contribution}
    
    breakeven_monthly = monthly_fixed / contribution
    return {
        'monthly': breakeven_monthly,
        'weekly': breakeven_monthly / WEEKS_PER_MONTH,
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

def apply_monthly_churn(clients: float, month_num: int, m1: float, m2: float, m3: float, ongoing: float) -> float:
    """Apply tiered churn rates"""
    if month_num == 1:
        return clients * m1
    elif month_num == 2:
        return clients * m2
    elif month_num == 3:
        return clients * m3
    else:
        return clients * ongoing

def project_new_hire(
    credential: str,
    sessions_week: int,
    util_rate: float,
    lmsw_pay: float,
    lcsw_pay: float,
    weighted_rate: float,
    cancel_rate: float,
    noshow_rate: float,
    payroll_tax: float,
    overhead_monthly: float,
    supervision_cost: float,
    hiring_cost: float,
    months: int = 12
) -> pd.DataFrame:
    """Project financial impact of hiring new therapist"""
    data = []
    pay_rate = lmsw_pay if credential == "LMSW" else lcsw_pay
    
    cumulative_profit = -hiring_cost  # Start with hiring cost
    
    for m in range(1, months + 1):
        # Ramp-up capacity
        if m == 1:
            capacity = 0.20
        elif m == 2:
            capacity = 0.40
        elif m == 3:
            capacity = 0.60
        elif m == 4:
            capacity = 0.80
        else:
            capacity = 1.0
        
        # Calculate sessions and revenue
        sessions = sessions_week * util_rate * WEEKS_PER_MONTH * capacity
        billable = sessions * (1 - cancel_rate) * (1 - noshow_rate)
        revenue = billable * weighted_rate
        
        # Calculate costs
        pay_cost = sessions * (1 - cancel_rate) * pay_rate * (1 + payroll_tax)
        total_cost = pay_cost + overhead_monthly + supervision_cost
        
        # Profit
        monthly_profit = revenue - total_cost
        cumulative_profit += monthly_profit
        
        data.append({
            'month': m,
            'capacity_%': capacity * 100,
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
owner_sessions_per_week = st.sidebar.number_input("Your Sessions/Week", min_value=0, value=20)
owner_utilization = st.sidebar.slider("Your Utilization (%)", 0, 100, 85) / 100

st.sidebar.header("🧑‍⚕️ Therapist Hiring")
st.sidebar.markdown("**Ramp-up: 5 months (20%→40%→60%→80%→100%)**")

therapists = []
therapists.append(Therapist(0, "Owner", "LCSW", 0, owner_sessions_per_week, owner_utilization, 0))

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
        
        if hire_m > 0:
            cred = col2.selectbox(f"Credential", ["LMSW", "LCSW"], index=0 if def_cred=="LMSW" else 1, key=f"cred_{i}")
            sess = st.slider(f"Sessions/Week", 10, 30, 20, key=f"sess_{i}")
            util = st.slider(f"Utilization %", 50, 100, 85, key=f"util_{i}") / 100
            cost = st.number_input(f"Hiring Cost", 0.0, value=def_cost, step=500.0, key=f"cost_{i}")
            therapists.append(Therapist(i, f"Therapist {i}", cred, hire_m, sess, util, cost))

st.sidebar.header("💰 Compensation")
lmsw_pay = st.sidebar.number_input("LMSW Pay/Session", 30.0, value=40.0, step=5.0)
lcsw_pay = st.sidebar.number_input("LCSW Pay/Session", 35.0, value=50.0, step=5.0)
owner_takes_pay = st.sidebar.checkbox("Owner Takes Pay", value=True)
payroll_tax = st.sidebar.slider("Payroll Tax %", 0.0, 20.0, 15.3, 0.1) / 100

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
cpl = st.sidebar.number_input("Cost per Lead", value=35.0, step=5.0)

st.sidebar.header("🎯 Financial Targets")
target_margin = st.sidebar.slider("Target Margin %", 0, 50, 25)
target_roi = st.sidebar.slider("Required ROI %", 50, 500, 200)

st.sidebar.header("👥 Group Therapy")
do_groups = st.sidebar.checkbox("Offer Groups", value=False)
if do_groups:
    group_lcsw_pct = st.sidebar.slider("% LCSW Groups", 0, 100, 30) / 100
    group_size = st.sidebar.number_input("Clients/Group", value=8)
    group_rev_client = st.sidebar.number_input("Rev/Client", value=60.0)
    group_pay = st.sidebar.number_input("Pay/Session", value=120.0)
    group_sessions_mo = st.sidebar.number_input("Sessions/Month", value=4)

st.sidebar.header("🧑‍🎓 Client Behavior")
avg_sess_mo = st.sidebar.number_input("Sessions/Client/Month", value=3.2, step=0.1)
cancel_rate = st.sidebar.slider("Cancellation %", 0, 40, 20) / 100
noshow_rate = st.sidebar.slider("No-Show %", 0, 30, 5) / 100
copay_pct = st.sidebar.slider("% with Copay", 0, 100, 20) / 100
avg_copay = st.sidebar.number_input("Avg Copay", value=25.0)
cc_fee = st.sidebar.slider("CC Fee %", 0.0, 5.0, 2.9, 0.1) / 100

st.sidebar.header("📉 Churn")
churn1 = st.sidebar.slider("Month 1 %", 0, 50, 25) / 100
churn2 = st.sidebar.slider("Month 2 %", 0, 50, 15) / 100
churn3 = st.sidebar.slider("Month 3 %", 0, 50, 10) / 100
churn_ongoing = st.sidebar.slider("Ongoing %", 0, 20, 5) / 100

st.sidebar.header("⚙️ Simulation")
sim_months = st.sidebar.number_input("Months", 12, 60, value=24)

# Input validation
assert 0 <= cancel_rate <= 1, "Cancellation rate must be 0-100%"
assert 0 <= noshow_rate <= 1, "No-show rate must be 0-100%"
assert cac_target > 0, "CAC must be positive"
assert avg_sess_mo > 0, "Sessions/client must be positive"

# ==========================================
# SIMULATION
# ==========================================

# Calculate owner's starting caseload
owner = therapists[0]
owner_capacity_sessions_month1 = owner.sessions_per_week_target * owner.utilization_rate * WEEKS_PER_MONTH
owner_initial_clients = owner_capacity_sessions_month1 / avg_sess_mo if avg_sess_mo > 0 else 0

monthly_data = []
active_clients = owner_initial_clients  # Start with owner's existing caseload
revenue_history = {p: [] for p in payer_mix.keys()}
months_in_op = 0
cumulative_hiring = 0.0
hired_tracker = set()
owner_sess_value = weighted_rate * (1 - noshow_rate) * (1 - cancel_rate)

for month in range(1, sim_months + 1):
    month_data = {"month": month}
    months_in_op += 1
    
    active_ther = [t for t in therapists if t.is_active(month)]
    total_cap_sess = sum(t.get_sessions_per_month(month) for t in therapists)
    total_cap_clients = total_cap_sess / avg_sess_mo if avg_sess_mo > 0 else 0
    
    lmsw_count = len([t for t in active_ther if t.credential == "LMSW"])
    lcsw_count = len([t for t in active_ther if t.credential == "LCSW"])
    
    month_data["active_therapists"] = len(active_ther)
    month_data["active_lmsw"] = lmsw_count
    month_data["active_lcsw"] = lcsw_count
    month_data["total_capacity_sessions"] = total_cap_sess
    month_data["total_capacity_clients"] = total_cap_clients
    
    churned = apply_monthly_churn(active_clients, months_in_op, churn1, churn2, churn3, churn_ongoing)
    surviving = active_clients - churned
    
    if mkt_model == "Fixed Monthly":
        mkt_budget = base_mkt
    elif mkt_model == "Per Active Therapist":
        mkt_budget = base_mkt + (len(active_ther) * mkt_per_therapist)
    else:
        empty = max(0, total_cap_clients - surviving)
        mkt_budget = empty * mkt_per_slot
    
    cap_avail = max(0, total_cap_clients - surviving)
    max_new = mkt_budget / cac_target if cac_target > 0 else 0
    new_clients = min(max_new, cap_avail)
    active_clients = surviving + new_clients
    
    month_data["churned_clients"] = churned
    month_data["new_clients"] = new_clients
    month_data["active_clients"] = active_clients
    month_data["capacity_utilization"] = (active_clients / total_cap_clients * 100) if total_cap_clients > 0 else 0
    
    sched_sess = active_clients * avg_sess_mo
    actual_sess = sched_sess * (1 - cancel_rate)
    billable_sess = actual_sess * (1 - noshow_rate)
    
    month_data["scheduled_sessions"] = sched_sess
    month_data["actual_sessions"] = actual_sess
    month_data["billable_sessions"] = billable_sess
    
    rev_by_payer = {}
    for pname, pinfo in payer_mix.items():
        p_sess = billable_sess * pinfo["pct"]
        p_rev = p_sess * pinfo["rate"]
        rev_by_payer[pname] = p_rev
        revenue_history[pname].append({"month": month, "revenue": p_rev})
    
    copay_rev = billable_sess * copay_pct * avg_copay
    cc_fees = copay_rev * cc_fee
    total_rev = sum(rev_by_payer.values()) + copay_rev - cc_fees
    
    month_data["copay_revenue"] = copay_rev
    month_data["cc_fees"] = cc_fees
    month_data["revenue_earned"] = total_rev
    
    collections = calculate_collections(revenue_history, month, payer_mix, copay_rev, cc_fees)
    month_data["collections"] = collections
    
    ther_cost_pre = 0.0
    owner_pay_pre = 0.0
    
    for t in therapists:
        if t.is_active(month):
            t_sess = t.get_sessions_per_month(month)
            t_actual = (t_sess / total_cap_sess) * actual_sess if total_cap_sess > 0 else 0
            
            pay = t_actual * (lmsw_pay if t.credential == "LMSW" else lcsw_pay)
            
            if t.id == 0 and owner_takes_pay:
                owner_pay_pre = pay
            else:
                ther_cost_pre += pay
    
    ther_cost = ther_cost_pre * (1 + payroll_tax)
    owner_pay = owner_pay_pre * (1 + payroll_tax)
    
    sup_cost, ext_sup, own_sup, ext_sup_count = calculate_supervision_costs(lmsw_count, owner_sess_value)
    month_data["owner_supervised_count"] = own_sup
    month_data["external_supervised_count"] = ext_sup_count
    
    cost_per_ther_ehr = get_ehr_cost(len(active_ther) + 1, ehr_sys, ehr_cost_custom) if ehr_sys == "Custom" or ehr_cost_custom is None else get_ehr_cost(len(active_ther) + 1, ehr_sys)
    ehr_monthly = cost_per_ther_ehr * (len(active_ther) + 1)
    tech_cost = ehr_monthly + telehealth + other_tech
    
    if bill_model == "Owner Does It":
        bill_cost = 0.0
    elif bill_model == "Billing Service (% of revenue)":
        bill_cost = total_rev * bill_pct
    else:
        bill_cost = biller_cost
    
    mkt_actual = new_clients * cac_target
    
    grp_rev = 0.0
    grp_cost_pre = 0.0
    if do_groups and lcsw_count > 0:
        n_groups = int(lcsw_count * group_lcsw_pct)
        grp_rev = n_groups * group_size * group_rev_client * group_sessions_mo
        grp_cost_pre = n_groups * group_pay * group_sessions_mo
    grp_cost = grp_cost_pre * (1 + payroll_tax)
    
    hiring_this_mo = 0.0
    for t in active_ther:
        if t.id not in hired_tracker and t.id != 0:
            if t.hire_month == month:
                hiring_this_mo += t.one_time_hiring_cost
                hired_tracker.add(t.id)
                cumulative_hiring += t.one_time_hiring_cost
    
    month_data["one_time_hiring_costs"] = hiring_this_mo
    month_data["cumulative_hiring_costs"] = cumulative_hiring
    
    total_costs = ther_cost + owner_pay + sup_cost + tech_cost + bill_cost + mkt_actual + other_overhead + grp_cost + hiring_this_mo
    
    month_data["therapist_costs"] = ther_cost
    month_data["owner_therapist_pay"] = owner_pay
    month_data["supervision_cost"] = sup_cost
    month_data["external_supervision_cost"] = ext_sup
    month_data["tech_cost"] = tech_cost
    month_data["billing_cost"] = bill_cost
    month_data["marketing_budget"] = mkt_budget
    month_data["marketing_spent"] = mkt_actual
    month_data["other_overhead"] = other_overhead
    month_data["group_revenue"] = grp_rev
    month_data["group_cost"] = grp_cost
    month_data["total_costs"] = total_costs
    
    profit_acc = total_rev + grp_rev - total_costs
    cash_flow = collections + grp_rev - total_costs
    
    month_data["profit_accrual"] = profit_acc
    month_data["cash_flow"] = cash_flow
    
    if month == 1:
        month_data["cash_balance"] = starting_cash_balance + cash_flow
    else:
        month_data["cash_balance"] = monthly_data[-1]["cash_balance"] + cash_flow
    
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
        
        owner_salary = yr_data['owner_therapist_pay'].sum()
        biz_profit = yr_data['profit_accrual'].sum()
        total_comp = owner_salary + biz_profit
        
        years_comp.append({
            'year': f"Year {yr}" + (f" ({mo_in_yr}mo)" if mo_in_yr < 12 else ""),
            'salary': owner_salary,
            'profit': biz_profit,
            'total': total_comp
        })

for yc in years_comp:
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{yc['year']} - Clinical Salary", f"${yc['salary']:,.0f}")
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

# MARKETING DASHBOARD
st.header("🎯 Marketing Efficiency Dashboard")

avg_pay = lmsw_pay * 0.6 + lcsw_pay * 0.4
contrib_per_sess = weighted_rate * (1 - cancel_rate) * (1 - noshow_rate) - avg_pay * (1 + payroll_tax)
clv = calculate_clv(avg_sess_mo, churn1, churn2, churn3, churn_ongoing, contrib_per_sess)
max_by_margin, max_by_roi, conservative_max = calculate_max_cac(clv, target_margin, target_roi)

total_new = df["new_clients"].sum()
total_mkt_spent = df["marketing_spent"].sum()
actual_cac = total_mkt_spent / total_new if total_new > 0 else 0
required_conv = (cpl / conservative_max) * 100 if conservative_max > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("CLV", f"${clv:,.0f}")
col2.metric("Max Affordable CAC", f"${conservative_max:,.0f}")
col3.metric("Actual CAC", f"${actual_cac:,.0f}", delta=f"${actual_cac - conservative_max:,.0f}", delta_color="inverse")
col4.metric("Required Conversion", f"{required_conv:.1f}%")

st.markdown(f"""
**CAC Calculation Details:**
- By Margin ({target_margin}%): ${max_by_margin:,.0f}
- By ROI ({target_roi}%): ${max_by_roi:,.0f}
- **Conservative (used)**: ${conservative_max:,.0f}

**Conversion Analysis:**
- At ${cpl:.0f} per lead, you need {required_conv:.1f}% conversion
""")

if required_conv <= 20:
    st.success("✅ Conversion rate achievable (excellent)")
elif required_conv <= 30:
    st.info("✓ Conversion rate achievable (good)")
else:
    st.warning("⚠️ Conversion rate challenging - lower CPL or improve lead quality")

if actual_cac > conservative_max:
    st.error(f"❌ CAC is ${actual_cac - conservative_max:.0f} over budget - improve conversion or lower ad spend")
else:
    st.success(f"✅ CAC is under budget by ${conservative_max - actual_cac:.0f} - marketing is efficient")

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

proj = project_new_hire(
    "LMSW", 20, 0.85, lmsw_pay, lcsw_pay, weighted_rate,
    cancel_rate, noshow_rate, payroll_tax,
    500, 200, 3000, 12
)

breakeven_month = proj[proj['cumulative_profit'] >= 0]['month'].min() if any(proj['cumulative_profit'] >= 0) else None

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    - One-Time Hiring Cost: $3,000
    - Ramp Period: 5 months
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

# Build therapist performance data
therapist_performance = {}

for therapist in therapists:
    if therapist.hire_month == 0 and therapist.id != 0:
        continue  # Skip disabled therapist slots
    
    therapist_performance[therapist.name] = {}
    
    for year_num in range(1, (sim_months // 12) + 2):
        start_month = (year_num - 1) * 12 + 1
        end_month = min(year_num * 12, sim_months)
        
        if start_month <= sim_months:
            months_active = [m for m in range(start_month, end_month + 1) if therapist.is_active(m)]
            
            if len(months_active) > 0:
                # Calculate sessions and revenue
                total_sessions = sum([therapist.get_sessions_per_month(m) for m in months_active])
                billable_sessions = total_sessions * (1 - cancel_rate) * (1 - noshow_rate)
                revenue = billable_sessions * weighted_rate
                
                # Direct costs (therapist pay)
                if therapist.credential == "LMSW":
                    pay_rate = lmsw_pay
                else:
                    pay_rate = lcsw_pay
                
                if therapist.id == 0 and owner_takes_pay:
                    direct_pay = total_sessions * (1 - cancel_rate) * pay_rate * (1 + payroll_tax)
                else:
                    direct_pay = total_sessions * (1 - cancel_rate) * pay_rate * (1 + payroll_tax)
                
                gross_margin = revenue - direct_pay
                gross_margin_pct = (gross_margin / revenue * 100) if revenue > 0 else 0
                
                # Allocated costs
                year_data = df[(df['month'] >= start_month) & (df['month'] <= end_month)]
                avg_therapists = year_data['active_therapists'].mean()
                
                overhead_per_month = (other_overhead + telehealth + other_tech) / avg_therapists if avg_therapists > 0 else 0
                overhead_allocated = overhead_per_month * len(months_active)
                
                # Supervision costs
                if therapist.credential == "LMSW":
                    supervision_allocated = 200 * len(months_active)  # Simplified
                else:
                    supervision_allocated = 0
                
                # Marketing allocation (proportional to capacity)
                total_marketing = year_data['marketing_spent'].sum()
                total_capacity = year_data['total_capacity_sessions'].sum()
                therapist_capacity_pct = total_sessions / total_capacity if total_capacity > 0 else 0
                marketing_allocated = total_marketing * therapist_capacity_pct
                
                # Tech costs
                tech_allocated = 150 * len(months_active)  # Simplified EHR cost
                
                total_allocated = overhead_allocated + supervision_allocated + marketing_allocated + tech_allocated
                
                net_margin = gross_margin - total_allocated
                net_margin_pct = (net_margin / revenue * 100) if revenue > 0 else 0
                
                therapist_performance[therapist.name][f"Year {year_num}"] = {
                    'credential': therapist.credential,
                    'months_active': len(months_active),
                    'sessions': billable_sessions,
                    'revenue': revenue,
                    'direct_pay': direct_pay,
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
for therapist_name, years in therapist_performance.items():
    if not years:
        continue
    
    st.markdown(f"### {therapist_name}")
    
    # Create columns for each year this therapist was active
    year_cols = st.columns(len(years))
    
    for idx, (year_label, data) in enumerate(years.items()):
        with year_cols[idx]:
            st.markdown(f"**{year_label}** ({data['credential']}, {data['months_active']}mo)")
            
            st.metric("Revenue", f"${data['revenue']:,.0f}")
            st.metric("Gross Margin", f"${data['gross_margin']:,.0f}", 
                     delta=f"{data['gross_margin_pct']:.1f}%")
            
            with st.expander("Cost Detail"):
                st.markdown(f"""
                **Direct Costs:**
                - Therapist Pay: ${data['direct_pay']:,.0f}
                
                **Allocated Costs:**
                - Overhead: ${data['overhead']:,.0f}
                - Supervision: ${data['supervision']:,.0f}
                - Marketing: ${data['marketing']:,.0f}
                - Technology: ${data['tech']:,.0f}
                - **Total**: ${data['total_allocated']:,.0f}
                """)
            
            # Net margin with color coding
            if data['net_margin'] > 0:
                st.success(f"**Net Margin:** ${data['net_margin']:,.0f} ({data['net_margin_pct']:.1f}%)")
            else:
                st.error(f"**Net Margin:** ${data['net_margin']:,.0f} ({data['net_margin_pct']:.1f}%)")
            
            st.markdown(f"_Sessions: {data['sessions']:.0f}_")
    
    st.markdown("---")

st.markdown("---")

# BREAK-EVEN ANALYSIS
st.header("⚖️ Break-Even Sessions Per Therapist")

lmsw_fixed = 500
lcsw_fixed = 500

lmsw_be = calculate_breakeven_sessions("LMSW", lmsw_pay, lcsw_pay, weighted_rate, noshow_rate, cancel_rate, lmsw_fixed, payroll_tax)
lcsw_be = calculate_breakeven_sessions("LCSW", lmsw_pay, lcsw_pay, weighted_rate, noshow_rate, cancel_rate, lcsw_fixed, payroll_tax)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### LMSW")
    st.metric("Monthly", f"{lmsw_be['monthly']:.1f}")
    st.metric("Weekly", f"{lmsw_be['weekly']:.1f}")
    st.markdown(f"Contribution: ${lmsw_be['contribution']:.2f}/session")

with col2:
    st.markdown("### LCSW")
    st.metric("Monthly", f"{lcsw_be['monthly']:.1f}")
    st.metric("Weekly", f"{lcsw_be['weekly']:.1f}")
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

ax1.plot(df["month"], df["active_clients"], 'o-', linewidth=2, label='Clients')
ax1.plot(df["month"], df["total_capacity_clients"], '--', linewidth=2, label='Capacity')
ax1.fill_between(df["month"], df["active_clients"], df["total_capacity_clients"], alpha=0.2)
ax1.set_title("Clients vs Capacity")
ax1.set_xlabel("Month")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df["month"], df["revenue_earned"], 'o-', linewidth=2, label='Revenue')
ax2.plot(df["month"], df["collections"], 's-', linewidth=2, label='Collections')
ax2.plot(df["month"], df["total_costs"], '^-', linewidth=2, label='Costs')
ax2.set_title("Revenue, Collections & Costs")
ax2.set_xlabel("Month")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.plot(df["month"], df["cash_balance"], 'o-', linewidth=2, color='green')
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax3.fill_between(df["month"], 0, df["cash_balance"], where=df["cash_balance"]<0, color='red', alpha=0.2)
ax3.fill_between(df["month"], 0, df["cash_balance"], where=df["cash_balance"]>=0, color='green', alpha=0.2)
ax3.set_title("Cash Balance")
ax3.set_xlabel("Month")
ax3.grid(True, alpha=0.3)

ax4.stackplot(df["month"], df["therapist_costs"], df["owner_therapist_pay"], df["supervision_cost"],
              df["tech_cost"], df["marketing_spent"], df["other_overhead"],
              labels=['Therapist', 'Owner', 'Supervision', 'Tech', 'Marketing', 'Other'], alpha=0.8)
ax4.set_title("Cost Breakdown")
ax4.set_xlabel("Month")
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# DETAILED DATA (in expanders)
with st.expander("📋 Detailed Monthly Data"):
    display_df = df[["month", "active_clients", "total_capacity_clients", "new_clients", "active_therapists", 
                     "revenue_earned", "collections", "total_costs", "profit_accrual", "cash_flow", "cash_balance"]]
    st.dataframe(display_df.style.format({
        "revenue_earned": "${:,.0f}", "collections": "${:,.0f}", "total_costs": "${:,.0f}",
        "profit_accrual": "${:,.0f}", "cash_flow": "${:,.0f}", "cash_balance": "${:,.0f}",
        "active_clients": "{:.1f}", "total_capacity_clients": "{:.1f}", "new_clients": "{:.1f}"
    }), use_container_width=True)

with st.expander("💰 Cost Breakdown Detail"):
    cost_df = df[["month", "therapist_costs", "owner_therapist_pay", "supervision_cost", "tech_cost", 
                  "billing_cost", "marketing_spent", "one_time_hiring_costs", "other_overhead", "total_costs"]]
    st.dataframe(cost_df.style.format({
        "therapist_costs": "${:,.0f}", "owner_therapist_pay": "${:,.0f}", "supervision_cost": "${:,.0f}",
        "tech_cost": "${:,.0f}", "billing_cost": "${:,.0f}", "marketing_spent": "${:,.0f}",
        "one_time_hiring_costs": "${:,.0f}", "other_overhead": "${:,.0f}", "total_costs": "${:,.0f}"
    }), use_container_width=True)

# SUPERVISION DETAIL
if df['active_lmsw'].max() > 0:
    with st.expander("🎓 Supervision Detail"):
        st.markdown(f"""
        **Final Month:**
        - LMSW Count: {final['active_lmsw']}
        - Owner Supervising: {final['owner_supervised_count']}
        - External: {final['external_supervised_count']}
        - Monthly Cost: ${final['supervision_cost']:,.0f}
        """)

# EXPORT
st.markdown("---")
st.header("📥 Export")
csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, f"practice_model_{sim_months}mo.csv", "text/csv")

st.success("✅ Model V4 Complete")
