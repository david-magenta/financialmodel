import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple

st.set_page_config(page_title="Therapy Practice Financial Model V3", layout="wide")
st.title("🧠 Therapy Practice Financial Model - Corrected Edition")
st.markdown("Virtual LCSW practice in NYC - Comprehensive financial modeling with validated formulas")

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
        """
        Ramp up capacity over 4 months after hire due to schedule filling.
        This is NOT credentialing delay - therapist can see clients immediately,
        but it takes time to fill their schedule completely.
        """
        if not self.is_active(current_month):
            return 0.0
        
        months_since_hire = current_month - self.hire_month
        
        if months_since_hire == 0:
            return 0.25
        elif months_since_hire == 1:
            return 0.50
        elif months_since_hire == 2:
            return 0.75
        else:
            return 1.0
    
    def get_sessions_per_month(self, current_month: int) -> float:
        """Calculate actual sessions per month accounting for utilization and ramp-up"""
        weeks_per_month = 52 / 12
        capacity_pct = self.get_capacity_percentage(current_month)
        return self.sessions_per_week_target * self.utilization_rate * weeks_per_month * capacity_pct

# ==========================================
# HELPER FUNCTIONS
# ==========================================

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
    else:
        return 75.0

def calculate_supervision_costs(num_lmsw: int, owner_session_value: float) -> Tuple[float, float, int, int]:
    """Calculate supervision costs with leverage model"""
    if num_lmsw == 0:
        return 0.0, 0.0, 0, 0
    
    owner_capacity = 3
    
    if num_lmsw <= owner_capacity:
        supervision_cost = num_lmsw * owner_session_value
        external_cost = 0.0
        owner_supervised = num_lmsw
        external_supervised = 0
    else:
        owner_supervision_cost = owner_capacity * owner_session_value
        external_supervisees = num_lmsw - owner_capacity
        external_cost = external_supervisees * 200.0
        supervision_cost = owner_supervision_cost + external_cost
        owner_supervised = owner_capacity
        external_supervised = external_supervisees
    
    return supervision_cost, external_cost, owner_supervised, external_supervised

def calculate_breakeven_sessions(
    credential: str,
    lmsw_pay_per_session: float,
    lcsw_pay_per_session: float,
    weighted_rate: float,
    no_show_rate: float,
    cancellation_rate: float,
    monthly_fixed_costs: float,
    payroll_tax_rate: float
) -> Dict[str, float]:
    """Calculate break-even sessions needed to cover allocated costs"""
    if credential == "LMSW":
        pay_per_session = lmsw_pay_per_session
    else:
        pay_per_session = lcsw_pay_per_session
    
    effective_revenue = weighted_rate * (1 - cancellation_rate) * (1 - no_show_rate)
    variable_cost = pay_per_session * (1 + payroll_tax_rate)
    contribution_margin = effective_revenue - variable_cost
    
    if contribution_margin <= 0:
        return {
            'monthly': float('inf'),
            'weekly': float('inf'),
            'contribution_margin': contribution_margin,
            'effective_revenue': effective_revenue,
            'variable_cost': variable_cost
        }
    
    breakeven_monthly = monthly_fixed_costs / contribution_margin
    breakeven_weekly = breakeven_monthly / 4.33
    
    return {
        'monthly': breakeven_monthly,
        'weekly': breakeven_weekly,
        'contribution_margin': contribution_margin,
        'effective_revenue': effective_revenue,
        'variable_cost': variable_cost
    }

def calculate_collections(
    revenue_by_payer_history: Dict[str, List[Dict]],
    current_month: int,
    payer_mix: Dict[str, Dict],
    copay_revenue: float,
    cc_fees: float
) -> float:
    """Calculate cash collections with corrected payment delay logic"""
    collections = 0.0
    collections += copay_revenue - cc_fees
    
    for payer_name, payer_info in payer_mix.items():
        delay_months = payer_info["delay_days"] / 30.0
        
        if delay_months == 0:
            if len(revenue_by_payer_history[payer_name]) >= current_month:
                current_month_revenue = revenue_by_payer_history[payer_name][current_month - 1]["revenue"]
                collections += current_month_revenue
            continue
        
        revenue_earned_month = current_month - delay_months
        
        if revenue_earned_month < 1:
            continue
        
        month_floor = int(np.floor(revenue_earned_month))
        month_ceil = int(np.ceil(revenue_earned_month))
        fraction = revenue_earned_month - month_floor
        
        floor_idx = month_floor - 1
        if 0 <= floor_idx < len(revenue_by_payer_history[payer_name]):
            floor_revenue = revenue_by_payer_history[payer_name][floor_idx]["revenue"]
            collections += floor_revenue * (1 - fraction)
        
        if month_ceil != month_floor:
            ceil_idx = month_ceil - 1
            if 0 <= ceil_idx < len(revenue_by_payer_history[payer_name]):
                ceil_revenue = revenue_by_payer_history[payer_name][ceil_idx]["revenue"]
                collections += ceil_revenue * fraction
    
    return collections

def apply_monthly_churn(
    active_clients: float,
    months_in_practice: int,
    month1_churn: float,
    month2_churn: float,
    month3_churn: float,
    ongoing_churn: float
) -> float:
    """Apply appropriate churn rate based on practice maturity (CORRECTED)"""
    if months_in_practice == 1:
        churn_rate = month1_churn
    elif months_in_practice == 2:
        churn_rate = month2_churn
    elif months_in_practice == 3:
        churn_rate = month3_churn
    else:
        churn_rate = ongoing_churn
    
    return active_clients * churn_rate

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("👤 Owner (You)")
owner_sessions_per_week = st.sidebar.number_input("Your Sessions per Week", min_value=0, value=20)
owner_utilization = st.sidebar.slider("Your Utilization Rate (%)", 0, 100, 85) / 100

st.sidebar.header("🧑‍⚕️ Therapist Hiring Schedule")
st.sidebar.markdown("**Hire Month** = Month when therapist starts (already credentialed)")
st.sidebar.markdown("Set hire month to 0 to disable a therapist slot")

therapists = []
therapists.append(Therapist(0, "Owner", "LCSW", 0, owner_sessions_per_week, owner_utilization, 0))

default_hires = [
    (1, "Therapist 1", "LMSW", 3, 20, 3000),
    (2, "Therapist 2", "LMSW", 6, 20, 3000),
    (3, "Therapist 3", "LCSW", 9, 20, 3500)
]

for i in range(1, 13):
    with st.sidebar.expander(f"Therapist {i}", expanded=(i <= 3)):
        if i <= len(default_hires):
            default_month = default_hires[i-1][3]
            default_cred = default_hires[i-1][2]
            default_hire_cost = default_hires[i-1][5]
        else:
            default_month = 0
            default_cred = "LMSW"
            default_hire_cost = 3000.0
        
        col1, col2 = st.columns(2)
        hire_month = col1.number_input(f"Hire Month", min_value=0, max_value=36, value=default_month, key=f"hire_{i}")
        
        if hire_month > 0:
            credential = col2.selectbox(f"Credential", ["LMSW", "LCSW"], index=0 if default_cred=="LMSW" else 1, key=f"cred_{i}")
            sessions = st.slider(f"Target Sessions/Week", 10, 30, 20, key=f"sess_{i}")
            util = st.slider(f"Utilization %", 50, 100, 85, key=f"util_{i}") / 100
            hire_cost = st.number_input(f"One-Time Hiring Cost", min_value=0.0, value=default_hire_cost, step=500.0, key=f"hire_cost_{i}")
            therapists.append(Therapist(i, f"Therapist {i}", credential, hire_month, sessions, util, hire_cost))

st.sidebar.header("💰 Compensation & Taxes")
lmsw_pay_per_session = st.sidebar.number_input("LMSW Pay per Session ($)", min_value=30.0, value=40.0, step=5.0)
lcsw_pay_per_session = st.sidebar.number_input("LCSW Pay per Session ($)", min_value=35.0, value=50.0, step=5.0)
owner_takes_therapist_pay = st.sidebar.checkbox("Owner Takes Therapist Pay", value=True)
payroll_tax_rate = st.sidebar.slider("Payroll Tax Rate (%)", 0.0, 20.0, 15.3, 0.1) / 100

st.sidebar.header("📊 Revenue - Payer Mix")
use_simple_payer = st.sidebar.checkbox("Use Simple Model", value=True)

if use_simple_payer:
    avg_insurance_rate = st.sidebar.number_input("Avg Insurance Rate ($)", value=100.0)
    self_pay_pct = st.sidebar.slider("Self-Pay %", 0, 50, 10) / 100
    self_pay_rate = st.sidebar.number_input("Self-Pay Rate ($)", value=150.0)
    payer_mix = {
        "Insurance": {"pct": 1 - self_pay_pct, "rate": avg_insurance_rate, "delay_days": 45},
        "Self-Pay": {"pct": self_pay_pct, "rate": self_pay_rate, "delay_days": 0}
    }
else:
    st.sidebar.subheader("Detailed Payer Mix")
    payer_mix = {}
    bcbs_pct = st.sidebar.slider("BCBS %", 0, 100, 30)
    bcbs_rate = st.sidebar.number_input("BCBS Rate", value=105.0)
    payer_mix["BCBS"] = {"pct": bcbs_pct/100, "rate": bcbs_rate, "delay_days": 30}
    
    aetna_pct = st.sidebar.slider("Aetna %", 0, 100, 25)
    aetna_rate = st.sidebar.number_input("Aetna Rate", value=110.0)
    payer_mix["Aetna"] = {"pct": aetna_pct/100, "rate": aetna_rate, "delay_days": 45}
    
    united_pct = st.sidebar.slider("United %", 0, 100, 20)
    united_rate = st.sidebar.number_input("United Rate", value=95.0)
    payer_mix["United"] = {"pct": united_pct/100, "rate": united_rate, "delay_days": 60}
    
    medicaid_pct = st.sidebar.slider("Medicaid %", 0, 100, 15)
    medicaid_rate = st.sidebar.number_input("Medicaid Rate", value=70.0)
    payer_mix["Medicaid"] = {"pct": medicaid_pct/100, "rate": medicaid_rate, "delay_days": 90}
    
    selfpay_pct = st.sidebar.slider("Self-Pay %", 0, 100, 10)
    selfpay_rate = st.sidebar.number_input("Self-Pay Rate", value=150.0)
    payer_mix["Self-Pay"] = {"pct": selfpay_pct/100, "rate": selfpay_rate, "delay_days": 0}

total_payer_pct = sum(p["pct"] for p in payer_mix.values())
if abs(total_payer_pct - 1.0) > 0.01:
    st.sidebar.error(f"⚠️ Payer mix totals {total_payer_pct*100:.1f}% - should be 100%")

weighted_rate = sum(p["pct"] * p["rate"] for p in payer_mix.values())
weighted_delay = sum(p["pct"] * p["delay_days"] for p in payer_mix.values()) / 30

st.sidebar.header("💻 Technology & Overhead")
ehr_system = st.sidebar.selectbox("EHR System", ["SimplePractice", "TherapyNotes", "Custom"])

if ehr_system == "Custom":
    ehr_cost_per_therapist = st.sidebar.number_input("EHR Cost per Therapist", value=75.0)
else:
    st.sidebar.info(f"{ehr_system}: Tiered pricing applied automatically")
    ehr_cost_per_therapist = None

telehealth_cost = st.sidebar.number_input("Telehealth Platform (monthly)", value=50.0)
other_tech_cost = st.sidebar.number_input("Other Tech/Software (monthly)", value=100.0)
other_overhead = st.sidebar.number_input("Other Monthly Overhead", value=1500.0)

st.sidebar.header("📄 Billing")
billing_model = st.sidebar.selectbox("Billing Model", ["Owner Does It", "Billing Service (% of revenue)", "In-House Biller"])

if billing_model == "Billing Service (% of revenue)":
    billing_service_pct = st.sidebar.slider("Billing Service Fee (%)", 4.0, 8.0, 6.0) / 100
elif billing_model == "In-House Biller":
    biller_monthly_cost = st.sidebar.number_input("Biller Monthly Salary", value=4500.0)

st.sidebar.header("🎯 Marketing & Growth")
marketing_model = st.sidebar.selectbox("Marketing Budget Model", ["Fixed Monthly", "Per Active Therapist", "Per Empty Capacity Slot"])

if marketing_model == "Fixed Monthly":
    base_marketing_budget = st.sidebar.number_input("Monthly Marketing Budget", value=2000.0)
elif marketing_model == "Per Active Therapist":
    base_marketing = st.sidebar.number_input("Base Marketing Budget", value=1000.0)
    marketing_per_therapist = st.sidebar.number_input("Additional per Therapist", value=500.0)
elif marketing_model == "Per Empty Capacity Slot":
    marketing_per_empty_slot = st.sidebar.number_input("Marketing $ per Empty Client Slot", value=50.0)

client_acquisition_cost = st.sidebar.number_input("Target Client Acquisition Cost", value=150.0)
cost_per_lead = st.sidebar.number_input("Cost per Lead ($)", value=35.0, min_value=1.0, step=5.0)

st.sidebar.header("🎯 Target Financial Metrics")
target_profit_margin_pct = st.sidebar.slider("Target Profit Margin (%)", 0, 50, 25)
target_roi_pct = st.sidebar.slider("Required Marketing ROI (%)", 50, 500, 200)

st.sidebar.header("👥 Group Therapy")
offer_group_therapy = st.sidebar.checkbox("Offer Group Therapy", value=False)

if offer_group_therapy:
    pct_lcsw_doing_groups = st.sidebar.slider("% of LCSW Running Groups", 0, 100, 30) / 100
    clients_per_group = st.sidebar.number_input("Clients per Group", value=8)
    group_revenue_per_client = st.sidebar.number_input("Revenue per Client (Group)", value=60.0)
    group_therapist_pay = st.sidebar.number_input("Therapist Pay per Group Session", value=120.0)
    group_sessions_per_month = st.sidebar.number_input("Group Sessions per Month", value=4)

st.sidebar.header("🧑‍🎓 Client Behavior")
avg_sessions_per_client_per_month = st.sidebar.number_input("Avg Sessions per Client per Month", value=3.2, step=0.1)
cancellation_rate = st.sidebar.slider("Cancellation Rate %", 0, 40, 20) / 100
no_show_rate = st.sidebar.slider("No-Show Rate %", 0, 30, 5) / 100
copay_pct = st.sidebar.slider("% Sessions with Copay", 0, 100, 20) / 100
avg_copay = st.sidebar.number_input("Avg Copay Amount", value=25.0)
cc_fee_pct = st.sidebar.slider("CC Processing Fee %", 0.0, 5.0, 2.9, 0.1) / 100

st.sidebar.header("📉 Churn Rates")
month1_churn = st.sidebar.slider("Month 1 Churn %", 0, 50, 25) / 100
month2_churn = st.sidebar.slider("Month 2 Churn %", 0, 50, 15) / 100
month3_churn = st.sidebar.slider("Month 3 Churn %", 0, 50, 10) / 100
ongoing_churn = st.sidebar.slider("Ongoing Monthly Churn %", 0, 20, 5) / 100

st.sidebar.header("⚙️ Simulation")
months_to_simulate = st.sidebar.number_input("Months to Simulate", min_value=12, max_value=60, value=24)

# ==========================================
# MAIN SIMULATION
# ==========================================

monthly_data = []
active_clients_count = 0.0
weeks_per_month = 52 / 12
revenue_by_payer_history = {payer: [] for payer in payer_mix.keys()}
months_in_operation = 0
cumulative_hiring_costs = 0.0
therapists_hired_tracker = set()
owner_session_value = weighted_rate * (1 - no_show_rate) * (1 - cancellation_rate)

for month in range(1, months_to_simulate + 1):
    month_data = {"month": month}
    months_in_operation += 1
    
    active_therapists = [t for t in therapists if t.is_active(month)]
    total_capacity_sessions = sum(t.get_sessions_per_month(month) for t in therapists)
    total_capacity_clients = total_capacity_sessions / avg_sessions_per_client_per_month if avg_sessions_per_client_per_month > 0 else 0
    
    active_lmsw_count = len([t for t in active_therapists if t.credential == "LMSW"])
    active_lcsw_count = len([t for t in active_therapists if t.credential == "LCSW"])
    
    month_data["active_therapists"] = len(active_therapists)
    month_data["active_lmsw"] = active_lmsw_count
    month_data["active_lcsw"] = active_lcsw_count
    month_data["total_capacity_sessions"] = total_capacity_sessions
    month_data["total_capacity_clients"] = total_capacity_clients
    
    churned_clients = apply_monthly_churn(active_clients_count, months_in_operation, month1_churn, month2_churn, month3_churn, ongoing_churn)
    surviving_clients = active_clients_count - churned_clients
    
    if marketing_model == "Fixed Monthly":
        marketing_budget = base_marketing_budget
    elif marketing_model == "Per Active Therapist":
        marketing_budget = base_marketing + (len(active_therapists) * marketing_per_therapist)
    else:
        empty_slots = max(0, total_capacity_clients - surviving_clients)
        marketing_budget = empty_slots * marketing_per_empty_slot
    
    capacity_available = max(0, total_capacity_clients - surviving_clients)
    max_new_clients_from_budget = marketing_budget / client_acquisition_cost if client_acquisition_cost > 0 else 0
    new_clients = min(max_new_clients_from_budget, capacity_available)
    active_clients_count = surviving_clients + new_clients
    
    month_data["churned_clients"] = churned_clients
    month_data["new_clients"] = new_clients
    month_data["active_clients"] = active_clients_count
    month_data["capacity_utilization"] = (active_clients_count / total_capacity_clients * 100) if total_capacity_clients > 0 else 0
    
    scheduled_sessions = active_clients_count * avg_sessions_per_client_per_month
    actual_sessions = scheduled_sessions * (1 - cancellation_rate)
    billable_sessions = actual_sessions * (1 - no_show_rate)
    
    month_data["scheduled_sessions"] = scheduled_sessions
    month_data["actual_sessions"] = actual_sessions
    month_data["billable_sessions"] = billable_sessions
    
    revenue_by_payer = {}
    for payer_name, payer_info in payer_mix.items():
        payer_sessions = billable_sessions * payer_info["pct"]
        payer_revenue = payer_sessions * payer_info["rate"]
        revenue_by_payer[payer_name] = payer_revenue
        revenue_by_payer_history[payer_name].append({"month": month, "revenue": payer_revenue, "delay_months": payer_info["delay_days"] / 30})
    
    copay_revenue = billable_sessions * copay_pct * avg_copay
    cc_fees = copay_revenue * cc_fee_pct
    total_revenue = sum(revenue_by_payer.values()) + copay_revenue - cc_fees
    
    month_data["copay_revenue"] = copay_revenue
    month_data["cc_fees"] = cc_fees
    month_data["revenue_earned"] = total_revenue
    
    collections = calculate_collections(revenue_by_payer_history, month, payer_mix, copay_revenue, cc_fees)
    month_data["collections"] = collections
    
    therapist_costs_pre_tax = 0.0
    owner_therapist_pay_pre_tax = 0.0
    
    for therapist in therapists:
        if therapist.is_active(month):
            therapist_sessions = therapist.get_sessions_per_month(month)
            therapist_actual_sessions = (therapist_sessions / total_capacity_sessions) * actual_sessions if total_capacity_sessions > 0 else 0
            
            if therapist.credential == "LMSW":
                pay = therapist_actual_sessions * lmsw_pay_per_session
            else:
                pay = therapist_actual_sessions * lcsw_pay_per_session
            
            if therapist.id == 0 and owner_takes_therapist_pay:
                owner_therapist_pay_pre_tax = pay
            else:
                therapist_costs_pre_tax += pay
    
    therapist_costs = therapist_costs_pre_tax * (1 + payroll_tax_rate)
    owner_therapist_pay = owner_therapist_pay_pre_tax * (1 + payroll_tax_rate)
    
    supervision_cost, external_supervision, owner_supervised, external_supervised = calculate_supervision_costs(active_lmsw_count, owner_session_value)
    month_data["owner_supervised_count"] = owner_supervised
    month_data["external_supervised_count"] = external_supervised
    
    if ehr_system == "Custom":
        cost_per_therapist_ehr = ehr_cost_per_therapist
    else:
        cost_per_therapist_ehr = get_ehr_cost(len(active_therapists) + 1, ehr_system)
    
    ehr_monthly_cost = cost_per_therapist_ehr * (len(active_therapists) + 1) if len(active_therapists) > 0 else cost_per_therapist_ehr
    total_tech_cost = ehr_monthly_cost + telehealth_cost + other_tech_cost
    
    if billing_model == "Owner Does It":
        billing_cost = 0.0
    elif billing_model == "Billing Service (% of revenue)":
        billing_cost = total_revenue * billing_service_pct
    else:
        billing_cost = biller_monthly_cost
    
    marketing_cost_actual = new_clients * client_acquisition_cost
    
    group_revenue = 0.0
    group_cost_pre_tax = 0.0
    if offer_group_therapy and active_lcsw_count > 0:
        num_groups = int(active_lcsw_count * pct_lcsw_doing_groups)
        group_revenue = num_groups * clients_per_group * group_revenue_per_client * group_sessions_per_month
        group_cost_pre_tax = num_groups * group_therapist_pay * group_sessions_per_month
    
    group_cost = group_cost_pre_tax * (1 + payroll_tax_rate)
    
    one_time_hiring_costs_this_month = 0.0
    for therapist in active_therapists:
        if therapist.id not in therapists_hired_tracker and therapist.id != 0:
            if therapist.hire_month == month:
                one_time_hiring_costs_this_month += therapist.one_time_hiring_cost
                therapists_hired_tracker.add(therapist.id)
                cumulative_hiring_costs += therapist.one_time_hiring_cost
    
    month_data["one_time_hiring_costs"] = one_time_hiring_costs_this_month
    month_data["cumulative_hiring_costs"] = cumulative_hiring_costs
    
    total_costs = therapist_costs + owner_therapist_pay + supervision_cost + total_tech_cost + billing_cost + marketing_cost_actual + other_overhead + group_cost + one_time_hiring_costs_this_month
    
    month_data["therapist_costs"] = therapist_costs
    month_data["owner_therapist_pay"] = owner_therapist_pay
    month_data["supervision_cost"] = supervision_cost
    month_data["external_supervision_cost"] = external_supervision
    month_data["tech_cost"] = total_tech_cost
    month_data["billing_cost"] = billing_cost
    month_data["marketing_budget"] = marketing_budget
    month_data["marketing_spent"] = marketing_cost_actual
    month_data["other_overhead"] = other_overhead
    month_data["group_revenue"] = group_revenue
    month_data["group_cost"] = group_cost
    month_data["total_costs"] = total_costs
    
    profit_accrual = total_revenue + group_revenue - total_costs
    cash_flow = collections + group_revenue - total_costs
    
    month_data["profit_accrual"] = profit_accrual
    month_data["cash_flow"] = cash_flow
    
    if month == 1:
        month_data["cash_balance"] = cash_flow
    else:
        month_data["cash_balance"] = monthly_data[-1]["cash_balance"] + cash_flow
    
    monthly_data.append(month_data)

df = pd.DataFrame(monthly_data)

# ==========================================
# DISPLAY RESULTS
# ==========================================

st.header("📊 Practice Overview")

final_month = df.iloc[-1]
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Active Clients", f"{final_month['active_clients']:.0f}")
col2.metric("Client Capacity", f"{final_month['total_capacity_clients']:.0f}")
col3.metric("Capacity Utilization", f"{final_month['capacity_utilization']:.1f}%")
col4.metric("Monthly Profit", f"${final_month['profit_accrual']:,.0f}")
col5.metric("Cash Balance", f"${final_month['cash_balance']:,.0f}")

avg_monthly_burn = df['cash_flow'].mean()
if avg_monthly_burn < 0:
    months_runway = abs(final_month['cash_balance'] / avg_monthly_burn) if avg_monthly_burn != 0 else float('inf')
    st.warning(f"⚠️ Negative cash flow: {months_runway:.1f} months runway remaining")
elif final_month['cash_balance'] < 0:
    st.error(f"❌ Negative cash balance: ${abs(final_month['cash_balance']):,.0f} in debt")
else:
    st.success(f"✅ Positive cash flow and cash balance")

st.markdown("---")

st.header("📋 Monthly Financial Summary")
display_df = df[["month", "active_clients", "total_capacity_clients", "new_clients", "active_therapists", "revenue_earned", "collections", "total_costs", "profit_accrual", "cash_flow", "cash_balance"]]
display_df.columns = ["Month", "Clients", "Capacity", "New", "Therapists", "Revenue", "Collections", "Costs", "Profit", "Cash Flow", "Cash Balance"]
st.dataframe(display_df.style.format({
    "Revenue": "${:,.0f}", "Collections": "${:,.0f}", "Costs": "${:,.0f}",
    "Profit": "${:,.0f}", "Cash Flow": "${:,.0f}", "Cash Balance": "${:,.0f}",
    "Clients": "{:.1f}", "Capacity": "{:.1f}", "New": "{:.1f}"
}), use_container_width=True)

st.header("💰 Monthly Cost Breakdown")
cost_breakdown_df = df[["month", "therapist_costs", "owner_therapist_pay", "supervision_cost", "tech_cost", "billing_cost", "marketing_spent", "group_cost", "one_time_hiring_costs", "other_overhead", "total_costs"]]
cost_breakdown_df.columns = ["Month", "Therapist Pay", "Owner Pay", "Supervision", "Technology", "Billing", "Marketing", "Group Cost", "Hiring Costs", "Other Overhead", "Total"]
st.dataframe(cost_breakdown_df.style.format({
    "Therapist Pay": "${:,.0f}", "Owner Pay": "${:,.0f}", "Supervision": "${:,.0f}",
    "Technology": "${:,.0f}", "Billing": "${:,.0f}", "Marketing": "${:,.0f}",
    "Group Cost": "${:,.0f}", "Hiring Costs": "${:,.0f}", "Other Overhead": "${:,.0f}", "Total": "${:,.0f}"
}), use_container_width=True)

st.info(f"💼 Total one-time hiring costs: ${cumulative_hiring_costs:,.0f}")

# SUPERVISION DASHBOARD
st.header("🎓 Supervision Analysis")

if df['active_lmsw'].max() > 0:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current LMSW", f"{final_month['active_lmsw']:.0f}")
    col2.metric("Owner Supervising", f"{final_month['owner_supervised_count']}")
    col3.metric("External Supervision", f"{final_month['external_supervised_count']}")
    col4.metric("Monthly Cost", f"${final_month['supervision_cost']:,.0f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Owner Supervision**")
        st.markdown(f"- Capacity: 3 LMSW max")
        st.markdown(f"- Cost: ${owner_session_value:.0f}/LMSW")
        st.markdown(f"- Current: {final_month['owner_supervised_count']}")
    
    with col2:
        st.markdown("**External Supervision**")
        st.markdown(f"- When: >3 LMSW")
        st.markdown(f"- Cost: $200/LMSW")
        st.markdown(f"- Current: {final_month['external_supervised_count']}")
    
    fig_sup, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    owner_costs = df['owner_supervised_count'] * owner_session_value
    external_costs = df['external_supervision_cost']
    ax1.stackplot(df['month'], owner_costs, external_costs, labels=['Owner', 'External'], alpha=0.8, colors=['steelblue', 'coral'])
    ax1.set_title("Supervision Costs")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Cost ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['month'], df['active_lmsw'], marker='o', linewidth=2, label='LMSW Count')
    ax2.axhline(y=3, color='red', linestyle='--', linewidth=2, label='Owner Capacity')
    ax2.fill_between(df['month'], 0, 3, alpha=0.2, color='green')
    ax2.fill_between(df['month'], 3, df['active_lmsw'], where=df['active_lmsw']>3, alpha=0.2, color='orange')
    ax2.set_title("LMSW vs Owner Capacity")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("LMSW Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_sup)

st.markdown("---")

# BREAK-EVEN ANALYSIS
st.header("⚖️ Break-Even Analysis")

lmsw_fixed = (ehr_monthly_cost / max(len([t for t in therapists if t.is_active(months_to_simulate)]), 1)) + (other_overhead / max(len([t for t in therapists if t.is_active(months_to_simulate)]), 1)) + 200
lcsw_fixed = (ehr_monthly_cost / max(len([t for t in therapists if t.is_active(months_to_simulate)]), 1)) + (other_overhead / max(len([t for t in therapists if t.is_active(months_to_simulate)]), 1))

lmsw_be = calculate_breakeven_sessions("LMSW", lmsw_pay_per_session, lcsw_pay_per_session, weighted_rate, no_show_rate, cancellation_rate, lmsw_fixed, payroll_tax_rate)
lcsw_be = calculate_breakeven_sessions("LCSW", lmsw_pay_per_session, lcsw_pay_per_session, weighted_rate, no_show_rate, cancellation_rate, lcsw_fixed, payroll_tax_rate)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### LMSW Break-Even")
    st.metric("Monthly Sessions", f"{lmsw_be['monthly']:.1f}")
    st.metric("Weekly Sessions", f"{lmsw_be['weekly']:.1f}")
    st.markdown(f"**Contribution:** ${lmsw_be['contribution_margin']:.2f}/session")

with col2:
    st.markdown("### LCSW Break-Even")
    st.metric("Monthly Sessions", f"{lcsw_be['monthly']:.1f}")
    st.metric("Weekly Sessions", f"{lcsw_be['weekly']:.1f}")
    st.markdown(f"**Contribution:** ${lcsw_be['contribution_margin']:.2f}/session")

st.markdown("---")

# CHARTS
st.header("📈 Key Metrics Over Time")

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

ax1.plot(df["month"], df["active_clients"], marker='o', linewidth=2, color='steelblue', label='Active Clients')
ax1.plot(df["month"], df["total_capacity_clients"], linestyle='--', linewidth=2, color='red', label='Capacity')
ax1.fill_between(df["month"], df["active_clients"], df["total_capacity_clients"], alpha=0.2, color='orange')
ax1.set_title("Client Growth vs Capacity")
ax1.set_xlabel("Month")
ax1.set_ylabel("Clients")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df["month"], df["revenue_earned"], marker='o', linewidth=2, label='Revenue', linestyle='--', alpha=0.7, color='green')
ax2.plot(df["month"], df["collections"], marker='s', linewidth=2, label='Collections', color='darkgreen')
ax2.plot(df["month"], df["total_costs"], marker='^', linewidth=2, label='Costs', color='red')
ax2.set_title("Revenue, Collections & Costs")
ax2.set_xlabel("Month")
ax2.set_ylabel("Dollars ($)")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.plot(df["month"], df["cash_balance"], marker='o', linewidth=2, color='green')
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax3.fill_between(df["month"], 0, df["cash_balance"], where=df["cash_balance"]<0, color='red', alpha=0.2)
ax3.fill_between(df["month"], 0, df["cash_balance"], where=df["cash_balance"]>=0, color='green', alpha=0.2)
ax3.set_title("Cash Balance")
ax3.set_xlabel("Month")
ax3.set_ylabel("Cash ($)")
ax3.grid(True, alpha=0.3)

ax4.stackplot(df["month"], df["therapist_costs"], df["owner_therapist_pay"], df["supervision_cost"], df["tech_cost"], df["billing_cost"], df["marketing_spent"], df["one_time_hiring_costs"], df["other_overhead"],
              labels=['Therapist', 'Owner', 'Supervision', 'Tech', 'Billing', 'Marketing', 'Hiring', 'Other'], alpha=0.8)
ax4.set_title("Cost Breakdown")
ax4.set_xlabel("Month")
ax4.set_ylabel("Costs ($)")
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig1)

# ANNUAL P&L
st.header("📊 Annual P&L Statements")

years = []
for year_num in range(1, (months_to_simulate // 12) + 2):
    start_month = (year_num - 1) * 12 + 1
    end_month = min(year_num * 12, months_to_simulate)
    
    if start_month <= months_to_simulate:
        year_data = df[(df['month'] >= start_month) & (df['month'] <= end_month)]
        months_in_year = len(year_data)
        total_revenue_year = year_data['revenue_earned'].sum() + year_data['group_revenue'].sum()
        
        years.append({
            'Year': f"Year {year_num}" + (f" ({months_in_year}mo)" if months_in_year < 12 else ""),
            'Revenue': total_revenue_year,
            'Costs': year_data['total_costs'].sum(),
            'Profit': year_data['profit_accrual'].sum(),
            'Margin %': (year_data['profit_accrual'].sum() / total_revenue_year * 100) if total_revenue_year > 0 else 0,
            'Cash Flow': year_data['cash_flow'].sum()
        })

st.dataframe(pd.DataFrame(years).style.format({
    'Revenue': '${:,.0f}', 'Costs': '${:,.0f}', 'Profit': '${:,.0f}',
    'Margin %': '{:.1f}%', 'Cash Flow': '${:,.0f}'
}), use_container_width=True)

# PER-THERAPIST P&L
st.header("👥 Per-Therapist P&L (Gross vs Net)")

therapist_data = []
for therapist in therapists:
    if therapist.id == 0 or therapist.hire_month == 0:
        continue
    
    for year_num in range(1, (months_to_simulate // 12) + 2):
        start = (year_num - 1) * 12 + 1
        end = min(year_num * 12, months_to_simulate)
        
        if start <= months_to_simulate:
            months_active = [m for m in range(start, end + 1) if therapist.is_active(m)]
            
            if len(months_active) > 0:
                sessions = sum([therapist.get_sessions_per_month(m) for m in months_active])
                billable = sessions * (1 - cancellation_rate) * (1 - no_show_rate)
                revenue = billable * weighted_rate
                
                pay_rate = lmsw_pay_per_session if therapist.credential == "LMSW" else lcsw_pay_per_session
                direct_pay = sessions * (1 - cancellation_rate) * pay_rate * (1 + payroll_tax_rate)
                gross_margin = revenue - direct_pay
                
                overhead = 500 * len(months_active)
                supervision = (200 if therapist.credential == "LMSW" else 0) * len(months_active)
                net_margin = gross_margin - overhead - supervision
                
                therapist_data.append({
                    'Therapist': therapist.name,
                    'Credential': therapist.credential,
                    'Year': f"Y{year_num}",
                    'Revenue': revenue,
                    'Direct Pay': direct_pay,
                    'Gross Margin': gross_margin,
                    'Gross %': (gross_margin/revenue*100) if revenue>0 else 0,
                    'Overhead': overhead,
                    'Supervision': supervision,
                    'Net Margin': net_margin,
                    'Net %': (net_margin/revenue*100) if revenue>0 else 0
                })

if therapist_data:
    st.dataframe(pd.DataFrame(therapist_data).style.format({
        'Revenue': '${:,.0f}', 'Direct Pay': '${:,.0f}', 'Gross Margin': '${:,.0f}',
        'Gross %': '{:.1f}%', 'Overhead': '${:,.0f}', 'Supervision': '${:,.0f}',
        'Net Margin': '${:,.0f}', 'Net %': '{:.1f}%'
    }), use_container_width=True)

st.markdown("---")

# STRATEGIC RECOMMENDATIONS
st.header("🎯 Strategic Recommendations")

survival = 1.0
sessions_total = 0.0
for i, rate in enumerate([month1_churn, month2_churn, month3_churn] + [ongoing_churn]*5):
    survival *= (1 - rate)
    sessions_total += survival * avg_sessions_per_client_per_month

avg_pay = lmsw_pay_per_session * 0.6 + lcsw_pay_per_session * 0.4
contrib = weighted_rate * (1 - cancellation_rate) * (1 - no_show_rate) - avg_pay * (1 + payroll_tax_rate)
clv = sessions_total * contrib

total_new = df["new_clients"].sum()
total_mkt = df["marketing_spent"].sum()
actual_cac = total_mkt / total_new if total_new > 0 else 0

max_cac_margin = clv * (1 - target_profit_margin_pct / 100)
max_cac_roi = clv / (1 + target_roi_pct / 100)
max_cac = min(max_cac_margin, max_cac_roi)
required_conv = (cost_per_lead / max_cac) * 100 if max_cac > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("CLV", f"${clv:,.0f}")
col2.metric("Max CAC", f"${max_cac:,.0f}")
col3.metric("Actual CAC", f"${actual_cac:,.0f}")
col4.metric("Required Conv.", f"{required_conv:.1f}%")

if actual_cac > max_cac:
    with st.expander("⚠️ CAC Too High", expanded=True):
        st.markdown(f"**Overspending:** ${(actual_cac - max_cac) * total_new:,.0f}")
        st.markdown("**Actions:** Improve conversion, lower CPL, boost referrals")

if cancellation_rate + no_show_rate > 0.20:
    lost = df["scheduled_sessions"].sum() * (cancellation_rate + no_show_rate) * weighted_rate
    with st.expander("📉 High Cancellation Rate", expanded=True):
        st.markdown(f"**Lost Revenue:** ${lost:,.0f}")
        st.markdown("**Actions:** 24hr policy, reminders, CC on file")

if final_month['cash_balance'] < -5000:
    with st.expander("💸 Cash Flow Gap", expanded=True):
        st.markdown(f"**Working Capital Needed:** ${abs(final_month['cash_balance'])*1.2:,.0f}")

if final_month['capacity_utilization'] < 70:
    with st.expander("📊 Low Utilization", expanded=True):
        st.markdown(f"**Empty Slots:** {final_month['total_capacity_clients'] - final_month['active_clients']:.0f}")
        st.markdown("**Actions:** Increase marketing or improve conversion")

st.markdown("---")

# EXPORT
st.header("📥 Export Data")
csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, f"practice_model_{months_to_simulate}mo.csv", "text/csv")

st.success("✅ Model Complete - All Corrections Implemented")
