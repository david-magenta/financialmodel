import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict

st.set_page_config(page_title="Therapy Practice Dashboard V2", layout="wide")
st.title("🧠 Therapy Practice Financial Dashboard - Professional Edition")
st.markdown("Virtual practice financial modeling with therapist scaling, credential mix optimization, and strategic recommendations")

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
    
    def is_active(self, current_month):
        """Therapist is active 3 months after hire (credentialing complete)"""
        return current_month >= self.hire_month + 3
    
    def get_capacity_percentage(self, current_month):
        """Ramp up capacity over 4 months after credentialing"""
        if not self.is_active(current_month):
            return 0.0
        
        months_since_credentialed = current_month - (self.hire_month + 3)
        
        if months_since_credentialed == 0:
            return 0.25
        elif months_since_credentialed == 1:
            return 0.50
        elif months_since_credentialed == 2:
            return 0.75
        else:
            return 1.0
    
    def get_sessions_per_month(self, current_month):
        weeks_per_month = 52 / 12
        capacity_pct = self.get_capacity_percentage(current_month)
        return self.sessions_per_week_target * self.utilization_rate * weeks_per_month * capacity_pct

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("👤 Owner (You)")
owner_sessions_per_week = st.sidebar.number_input("Your Sessions per Week", min_value=0, value=20)
owner_utilization = st.sidebar.slider("Your Utilization Rate (%)", 0, 100, 85) / 100

st.sidebar.header("🧑‍⚕️ Therapist Hiring Schedule")
st.sidebar.markdown("Configure up to 12 therapists. Set hire month to 0 to disable.")

therapists = []
therapists.append(Therapist(0, "Owner", "LCSW", 0, owner_sessions_per_week, owner_utilization))

default_hires = [
    (1, "Therapist 1", "LMSW", 3, 20),
    (2, "Therapist 2", "LMSW", 6, 20),
    (3, "Therapist 3", "LCSW", 9, 20)
]

for i in range(1, 13):
    with st.sidebar.expander(f"Therapist {i}", expanded=(i <= 3)):
        if i <= len(default_hires):
            default_month = default_hires[i-1][3]
            default_cred = default_hires[i-1][2]
        else:
            default_month = 0
            default_cred = "LMSW"
        
        col1, col2 = st.columns(2)
        hire_month = col1.number_input(f"Hire Month (0=disabled)", 
                                       min_value=0, max_value=36, 
                                       value=default_month, key=f"hire_{i}")
        
        if hire_month > 0:
            credential = col2.selectbox(f"Credential", 
                                       ["LMSW", "LCSW"], 
                                       index=0 if default_cred=="LMSW" else 1,
                                       key=f"cred_{i}")
            sessions = st.slider(f"Target Sessions/Week", 10, 30, 20, key=f"sess_{i}")
            util = st.slider(f"Utilization %", 50, 100, 85, key=f"util_{i}") / 100
            
            therapists.append(Therapist(i, f"Therapist {i}", credential, 
                                       hire_month, sessions, util))

st.sidebar.header("💰 Compensation")
lmsw_pay_per_session = st.sidebar.number_input("LMSW Pay per Session ($)", 
                                                min_value=30.0, value=40.0, step=5.0)
lcsw_pay_per_session = st.sidebar.number_input("LCSW Pay per Session ($)", 
                                                min_value=35.0, value=50.0, step=5.0)
owner_takes_therapist_pay = st.sidebar.checkbox("Owner Takes Therapist Pay", value=True)

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

weighted_rate = sum(p["pct"] * p["rate"] for p in payer_mix.values())
weighted_delay = sum(p["pct"] * p["delay_days"] for p in payer_mix.values()) / 30

st.sidebar.header("💻 Technology & Overhead")
ehr_system = st.sidebar.selectbox("EHR System", 
                                   ["SimplePractice", "TherapyNotes", "Custom"])

if ehr_system == "Custom":
    ehr_cost_per_therapist = st.sidebar.number_input("EHR Cost per Therapist", value=75.0)
else:
    st.sidebar.info(f"{ehr_system}: Tiered pricing (calculated automatically)")
    ehr_cost_per_therapist = None

telehealth_cost = st.sidebar.number_input("Telehealth Platform (monthly)", value=50.0)
other_tech_cost = st.sidebar.number_input("Other Tech/Software (monthly)", value=100.0)
other_overhead = st.sidebar.number_input("Other Monthly Overhead", value=1500.0)

st.sidebar.header("📄 Billing")
billing_model = st.sidebar.selectbox("Billing Model", 
                                     ["Owner Does It", "Billing Service (% of revenue)", "In-House Biller"])

if billing_model == "Billing Service (% of revenue)":
    billing_service_pct = st.sidebar.slider("Billing Service Fee (%)", 4.0, 8.0, 6.0) / 100
elif billing_model == "In-House Biller":
    biller_monthly_cost = st.sidebar.number_input("Biller Monthly Salary", value=4500.0)

st.sidebar.header("🎯 Marketing")
marketing_model = st.sidebar.selectbox("Marketing Budget Model", 
                                       ["Fixed Monthly", "Per Active Therapist", "Per Empty Capacity Slot"])

if marketing_model == "Fixed Monthly":
    base_marketing_budget = st.sidebar.number_input("Monthly Marketing Budget", value=2000.0)
elif marketing_model == "Per Active Therapist":
    base_marketing = st.sidebar.number_input("Base Marketing Budget", value=1000.0)
    marketing_per_therapist = st.sidebar.number_input("Additional per Therapist", value=500.0)
elif marketing_model == "Per Empty Capacity Slot":
    marketing_per_empty_slot = st.sidebar.number_input("Marketing $ per Empty Client Slot", value=50.0)

client_acquisition_cost = st.sidebar.number_input("Client Acquisition Cost", value=150.0)

st.sidebar.header("👥 Group Therapy")
offer_group_therapy = st.sidebar.checkbox("Offer Group Therapy", value=False)

if offer_group_therapy:
    pct_lcsw_doing_groups = st.sidebar.slider("% of LCSW Running Groups", 0, 100, 30) / 100
    clients_per_group = st.sidebar.number_input("Clients per Group", value=8)
    group_revenue_per_client = st.sidebar.number_input("Revenue per Client (Group)", value=60.0)
    group_therapist_pay = st.sidebar.number_input("Therapist Pay per Group Session", value=120.0)
    group_sessions_per_month = st.sidebar.number_input("Group Sessions per Month", value=4)

st.sidebar.header("🧑‍🎓 Client Behavior")
avg_sessions_per_client_per_month = st.sidebar.number_input("Avg Sessions per Client per Month", value=2.5, step=0.1)
copay_pct = st.sidebar.slider("% Sessions with Copay", 0, 100, 20) / 100
avg_copay = st.sidebar.number_input("Avg Copay Amount", value=25.0)
cc_fee_pct = st.sidebar.slider("CC Processing Fee %", 0.0, 5.0, 2.9, 0.1) / 100
no_show_rate = st.sidebar.slider("No-Show Rate %", 0, 30, 5) / 100

st.sidebar.header("📉 Churn Rates")
month1_churn = st.sidebar.slider("Month 1 Churn %", 0, 50, 25) / 100
month2_churn = st.sidebar.slider("Month 2 Churn %", 0, 50, 15) / 100
month3_churn = st.sidebar.slider("Month 3 Churn %", 0, 50, 10) / 100
ongoing_churn = st.sidebar.slider("Ongoing Monthly Churn %", 0, 20, 5) / 100

st.sidebar.header("⚙️ Simulation")
months_to_simulate = st.sidebar.number_input("Months to Simulate", min_value=12, max_value=60, value=24)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_ehr_cost(num_therapists, system):
    """Calculate EHR cost with tiered pricing"""
    if system == "SimplePractice":
        if num_therapists <= 3:
            return 99
        elif num_therapists <= 9:
            return 89
        else:
            return 79
    elif system == "TherapyNotes":
        if num_therapists <= 3:
            return 59
        elif num_therapists <= 9:
            return 49
        else:
            return 39
    else:
        return ehr_cost_per_therapist

def calculate_supervision_costs(num_lmsw):
    """Calculate supervision costs"""
    if num_lmsw == 0:
        return 0, 0
    
    if num_lmsw <= 3:
        owner_time_cost = 4
        external_cost = 0
    else:
        owner_time_cost = 4
        external_supervisees = num_lmsw - 3
        external_cost = external_supervisees * 150
    
    return owner_time_cost, external_cost

# ==========================================
# MAIN SIMULATION
# ==========================================

monthly_data = []
active_clients_count = 0
weeks_per_month = 52 / 12

revenue_by_payer_history = {payer: [] for payer in payer_mix.keys()}

for month in range(1, months_to_simulate + 1):
    month_data = {"month": month}
    
    active_therapists = [t for t in therapists if t.is_active(month)]
    total_therapists_hired = len([t for t in therapists if t.hire_month > 0 and t.hire_month <= month])
    
    total_capacity_sessions = sum(t.get_sessions_per_month(month) for t in therapists)
    total_capacity_clients = total_capacity_sessions / avg_sessions_per_client_per_month if avg_sessions_per_client_per_month > 0 else 0
    
    active_lmsw_count = len([t for t in active_therapists if t.credential == "LMSW"])
    active_lcsw_count = len([t for t in active_therapists if t.credential == "LCSW"])
    
    month_data["total_therapists_hired"] = total_therapists_hired
    month_data["active_therapists"] = len(active_therapists)
    month_data["active_lmsw"] = active_lmsw_count
    month_data["active_lcsw"] = active_lcsw_count
    month_data["total_capacity_sessions"] = total_capacity_sessions
    month_data["total_capacity_clients"] = total_capacity_clients
    
    churned_clients = active_clients_count * ongoing_churn
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
    
    expected_sessions = active_clients_count * avg_sessions_per_client_per_month
    
    revenue_by_payer = {}
    for payer_name, payer_info in payer_mix.items():
        payer_sessions = expected_sessions * payer_info["pct"]
        payer_revenue = payer_sessions * payer_info["rate"] * (1 - no_show_rate)
        revenue_by_payer[payer_name] = payer_revenue
        revenue_by_payer_history[payer_name].append({
            "month": month,
            "revenue": payer_revenue,
            "delay_months": payer_info["delay_days"] / 30
        })
    
    copay_revenue = expected_sessions * copay_pct * avg_copay * (1 - no_show_rate)
    cc_fees = copay_revenue * cc_fee_pct
    
    total_revenue = sum(revenue_by_payer.values()) + copay_revenue - cc_fees
    
    month_data["expected_sessions"] = expected_sessions
    month_data["revenue_earned"] = total_revenue
    month_data["copay_revenue"] = copay_revenue
    month_data["cc_fees"] = cc_fees
    
    collections = 0
    for payer_name, payer_info in payer_mix.items():
        delay_months = payer_info["delay_days"] / 30
        delay_floor = int(delay_months)
        delay_fraction = delay_months - delay_floor
        
        if delay_floor == 0:
            collections += revenue_by_payer[payer_name]
        else:
            delay_idx_floor = month - 1 - delay_floor
            if 0 <= delay_idx_floor < len(revenue_by_payer_history[payer_name]):
                collections += revenue_by_payer_history[payer_name][delay_idx_floor]["revenue"] * (1 - delay_fraction)
            
            if delay_fraction > 0:
                delay_idx_ceil = month - 1 - (delay_floor + 1)
                if 0 <= delay_idx_ceil < len(revenue_by_payer_history[payer_name]):
                    collections += revenue_by_payer_history[payer_name][delay_idx_ceil]["revenue"] * delay_fraction
    
    collections += copay_revenue - cc_fees
    
    month_data["collections"] = collections
    
    therapist_costs = 0
    owner_therapist_pay_amount = 0
    
    for therapist in therapists:
        if therapist.is_active(month):
            therapist_sessions = therapist.get_sessions_per_month(month)
            therapist_actual_sessions = (therapist_sessions / total_capacity_sessions) * expected_sessions if total_capacity_sessions > 0 else 0
            
            if therapist.credential == "LMSW":
                pay = therapist_actual_sessions * lmsw_pay_per_session
            else:
                pay = therapist_actual_sessions * lcsw_pay_per_session
            
            if therapist.id == 0 and owner_takes_therapist_pay:
                owner_therapist_pay_amount = pay
            else:
                therapist_costs += pay
    
    owner_supervision_hours, external_supervision_cost = calculate_supervision_costs(active_lmsw_count)
    
    if ehr_system == "Custom":
        ehr_monthly_cost = ehr_cost_per_therapist * (len(active_therapists) + 1)
    else:
        cost_per = get_ehr_cost(len(active_therapists) + 1, ehr_system)
        ehr_monthly_cost = cost_per * (len(active_therapists) + 1)
    
    total_tech_cost = ehr_monthly_cost + telehealth_cost + other_tech_cost
    
    if billing_model == "Owner Does It":
        billing_cost = 0
        owner_billing_hours = 15
    elif billing_model == "Billing Service (% of revenue)":
        billing_cost = total_revenue * billing_service_pct
        owner_billing_hours = 0
    else:
        billing_cost = biller_monthly_cost
        owner_billing_hours = 0
    
    marketing_cost_actual = new_clients * client_acquisition_cost
    
    group_revenue = 0
    group_cost = 0
    if offer_group_therapy and active_lcsw_count > 0:
        num_groups = int(active_lcsw_count * pct_lcsw_doing_groups)
        group_revenue = num_groups * clients_per_group * group_revenue_per_client * group_sessions_per_month
        group_cost = num_groups * group_therapist_pay * group_sessions_per_month
    
    total_costs = (therapist_costs + owner_therapist_pay_amount + external_supervision_cost + 
                   total_tech_cost + billing_cost + marketing_cost_actual + other_overhead + group_cost)
    
    month_data["therapist_costs"] = therapist_costs
    month_data["owner_therapist_pay"] = owner_therapist_pay_amount
    month_data["supervision_cost"] = external_supervision_cost
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
col2.metric("Active Therapists", f"{final_month['active_therapists']}")
col3.metric("Capacity Utilization", f"{final_month['capacity_utilization']:.1f}%")
col4.metric("Monthly Profit", f"${final_month['profit_accrual']:,.0f}")
col5.metric("Cash Balance", f"${final_month['cash_balance']:,.0f}")

st.header("📋 Monthly Financial Summary")
display_df = df[["month", "active_clients", "new_clients", "active_therapists", 
                 "revenue_earned", "collections", "total_costs", "profit_accrual", "cash_balance"]]
display_df.columns = ["Month", "Clients", "New", "Therapists", "Revenue", "Collections", "Costs", "Profit", "Cash"]
st.dataframe(display_df.style.format({
    "Revenue": "${:,.0f}",
    "Collections": "${:,.0f}",
    "Costs": "${:,.0f}",
    "Profit": "${:,.0f}",
    "Cash": "${:,.0f}",
    "Clients": "{:.1f}",
    "New": "{:.1f}"
}), use_container_width=True)

st.header("📈 Key Metrics Over Time")

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

ax1.plot(df["month"], df["active_clients"], marker='o', linewidth=2, color='steelblue')
ax1.plot(df["month"], df["total_capacity_clients"], linestyle='--', color='red', label='Capacity')
ax1.set_title("Client Growth vs Capacity")
ax1.set_xlabel("Month")
ax1.set_ylabel("Clients")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df["month"], df["revenue_earned"], marker='o', linewidth=2, label='Revenue (Accrual)', linestyle='--', alpha=0.7)
ax2.plot(df["month"], df["collections"], marker='s', linewidth=2, label='Collections (Cash)')
ax2.plot(df["month"], df["total_costs"], marker='^', linewidth=2, label='Costs', color='red')
ax2.set_title("Revenue, Collections & Costs")
ax2.set_xlabel("Month")
ax2.set_ylabel("Dollars ($)")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.plot(df["month"], df["cash_balance"], marker='o', linewidth=2, color='green')
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax3.fill_between(df["month"], 0, df["cash_balance"], 
                  where=df["cash_balance"]<0, color='red', alpha=0.2)
ax3.fill_between(df["month"], 0, df["cash_balance"], 
                  where=df["cash_balance"]>=0, color='green', alpha=0.2)
ax3.set_title("Cash Balance Over Time")
ax3.set_xlabel("Month")
ax3.set_ylabel("Cash Balance ($)")
ax3.grid(True, alpha=0.3)

ax4.plot(df["month"], df["active_therapists"], marker='o', linewidth=2, label='Active', color='blue')
ax4.plot(df["month"], df["active_lmsw"], marker='s', linewidth=2, label='LMSW', color='orange')
ax4.plot(df["month"], df["active_lcsw"], marker='^', linewidth=2, label='LCSW', color='purple')
ax4.set_title("Therapist Team Composition")
ax4.set_xlabel("Month")
ax4.set_ylabel("Count")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig1)

st.success("✅ Dashboard V2 Running Successfully!")

# ==========================================
# STRATEGIC RECOMMENDATIONS ENGINE
# ==========================================

st.header("🎯 Strategic Levers & Optimization Opportunities")

recommendations = []

# Get current state metrics
current_month = df.iloc[-1]
avg_monthly_profit = df["profit_accrual"].mean()
avg_clients = df["active_clients"].mean()
total_annual_revenue = df["revenue_earned"].sum()

# Calculate CLV
survival_rate = 1.0
total_expected_sessions = 0
for i in range(1, 9):
    if i == 1:
        survival_rate *= (1 - month1_churn)
    elif i == 2:
        survival_rate *= (1 - month2_churn)
    elif i == 3:
        survival_rate *= (1 - month3_churn)
    else:
        survival_rate *= (1 - ongoing_churn)
    total_expected_sessions += survival_rate * avg_sessions_per_client_per_month

contribution_margin = weighted_rate * (1 - no_show_rate) - (lmsw_pay_per_session * 0.7 + lcsw_pay_per_session * 0.3)
current_clv = total_expected_sessions * contribution_margin

# --- RECOMMENDATION 1: Credential Mix ---
if current_month["active_lmsw"] > 0:
    lmsw_pct = current_month["active_lmsw"] / max(current_month["active_therapists"], 1) * 100
    monthly_lmsw_savings = current_month["active_lmsw"] * (lcsw_pay_per_session - lmsw_pay_per_session) * avg_sessions_per_client_per_month * (avg_clients / max(current_month["active_therapists"], 1))
    monthly_supervision_cost = current_month["supervision_cost"]
    net_monthly_savings = monthly_lmsw_savings - monthly_supervision_cost
    
    with st.expander("💰 Supervision Leverage Strategy", expanded=True):
        st.markdown(f"**Current State:** {lmsw_pct:.0f}% LMSW mix ({current_month['active_lmsw']:.0f} LMSW, {current_month['active_lcsw']:.0f} LCSW)")
        st.markdown(f"**Savings:** ${monthly_lmsw_savings:.0f}/month in lower therapist costs")
        st.markdown(f"**Cost:** ${monthly_supervision_cost:.0f}/month in supervision" + (" (you handle 3, external for rest)" if current_month['active_lmsw'] > 3 else " (you handle all)"))
        st.markdown(f"**Net Impact:** ${net_monthly_savings:.0f}/month (${net_monthly_savings*12:.0f}/year)")
        
        if net_monthly_savings > 0:
            st.success("✅ Continue LMSW strategy - positive ROI")
        else:
            st.warning("⚠️ Consider hiring more LCSWs - supervision costs exceed savings")

# --- RECOMMENDATION 2: Marketing Efficiency ---
total_new_clients = df["new_clients"].sum()
total_marketing_spent = df["marketing_spent"].sum()
actual_cac = total_marketing_spent / total_new_clients if total_new_clients > 0 else 0
max_profitable_cac = current_clv * 0.25

if actual_cac > max_profitable_cac:
    with st.expander("⚠️ Reduce Client Acquisition Cost", expanded=True):
        st.markdown(f"**Current State:** ${actual_cac:.0f} CAC vs ${max_profitable_cac:.0f} max recommended")
        st.markdown(f"**Impact:** Overspending ${(actual_cac - max_profitable_cac) * total_new_clients:.0f} over {months_to_simulate} months")
        st.markdown("**Action Items:**")
        st.markdown(f"- Target CAC: ${max_profitable_cac:.0f} (25% of CLV)")
        st.markdown("- Improve conversion rates")
        st.markdown("- Focus on referral programs")
        st.markdown("- Test lower-cost channels")

# --- RECOMMENDATION 3: Retention ---
if ongoing_churn > 0.03:
    improved_churn = ongoing_churn * 0.6
    new_lifetime_sessions = 0
    survival = 1.0
    for i in range(1, 15):
        if i <= 3:
            if i == 1: survival *= (1 - month1_churn)
            elif i == 2: survival *= (1 - month2_churn)
            else: survival *= (1 - month3_churn)
        else:
            survival *= (1 - improved_churn)
        new_lifetime_sessions += survival * avg_sessions_per_client_per_month
    
    improved_clv = new_lifetime_sessions * contribution_margin
    clv_increase = improved_clv - current_clv
    
    with st.expander("💎 Reduce Ongoing Churn Rate", expanded=True):
        st.markdown(f"**Current State:** {ongoing_churn*100:.0f}% monthly churn → {total_expected_sessions:.1f} lifetime sessions")
        st.markdown(f"**Impact:** Reducing to {improved_churn*100:.1f}% would increase CLV by ${clv_increase:.0f} ({(clv_increase/current_clv*100):.0f}% improvement)")
        st.markdown("**Action Items:**")
        st.markdown("- Check-ins at sessions 3, 6, 10")
        st.markdown("- Survey at-risk clients")
        st.markdown("- Package pricing (pre-pay discounts)")
        st.markdown(f"- Annual impact: ${clv_increase * total_new_clients:.0f}")

# --- RECOMMENDATION 4: Self-Pay Mix ---
if use_simple_payer and self_pay_pct < 0.15:
    target_self_pay = 0.15
    current_weighted_rate_calc = (1 - self_pay_pct) * avg_insurance_rate + self_pay_pct * self_pay_rate
    target_weighted_rate = (1 - target_self_pay) * avg_insurance_rate + target_self_pay * self_pay_rate
    rate_increase = target_weighted_rate - current_weighted_rate_calc
    
    with st.expander("💵 Increase Self-Pay Mix"):
        st.markdown(f"**Current State:** {self_pay_pct*100:.0f}% self-pay at ${self_pay_rate:.0f}")
        st.markdown(f"**Impact:** Increasing to {target_self_pay*100:.0f}% would increase avg rate by ${rate_increase:.0f}/session")
        st.markdown("**Action Items:**")
        st.markdown("- Market to HSA/FSA clients")
        st.markdown("- Offer superbills for OON benefits")
        st.markdown("- Specialize in niches (trauma, couples)")
        st.markdown("- No insurance delays = better cash flow!")

# --- RECOMMENDATION 5: Group Therapy ---
if not offer_group_therapy and current_month["active_lcsw"] >= 2:
    potential_groups = int(current_month["active_lcsw"] * 0.3)
    group_revenue_potential = potential_groups * 8 * 60 * 4
    group_cost_potential = potential_groups * 120 * 4
    group_profit_potential = group_revenue_potential - group_cost_potential
    
    with st.expander("🚀 Launch Group Therapy", expanded=True):
        st.markdown(f"**Current State:** Not offering groups. Have {current_month['active_lcsw']:.0f} LCSW available")
        st.markdown(f"**Impact:** Potential +${group_profit_potential:.0f}/month at 75% margins")
        st.markdown("**Action Items:**")
        st.markdown(f"- Start with {potential_groups} group(s)")
        st.markdown("- Groups: DBT skills, anxiety, grief")
        st.markdown("- 8 clients × $60 = $480 revenue vs $120 cost")
        st.markdown(f"- Annual impact: ${group_profit_potential * 12:.0f}")

# --- RECOMMENDATION 6: No-Shows ---
if no_show_rate > 0.03:
    annual_sessions_lost = df["expected_sessions"].sum() * no_show_rate
    annual_revenue_lost = annual_sessions_lost * weighted_rate
    
    with st.expander("📉 Reduce No-Show Rate"):
        st.markdown(f"**Current State:** {no_show_rate*100:.0f}% no-show rate")
        st.markdown(f"**Impact:** Losing ${annual_revenue_lost:,.0f}/year in missed sessions")
        st.markdown("**Action Items:**")
        st.markdown("- Automated reminders (24hr + 2hr)")
        st.markdown("- Credit card on file")
        st.markdown("- 24hr cancellation policy")
        st.markdown(f"- Reducing to 3% saves ${(no_show_rate - 0.03) / no_show_rate * annual_revenue_lost:,.0f}/year")

# --- RECOMMENDATION 7: Cash Flow ---
min_cash_balance = df["cash_balance"].min()
if min_cash_balance < -5000:
    working_capital_needed = abs(min_cash_balance) * 1.2
    
    with st.expander("⚠️ Address Working Capital Gap", expanded=True):
        st.markdown(f"**Current State:** Minimum cash: ${min_cash_balance:,.0f}")
        st.markdown(f"**Impact:** Need ${working_capital_needed:,.0f} working capital")
        st.markdown("**Action Items:**")
        st.markdown(f"- Secure ${working_capital_needed:,.0f} line of credit")
        st.markdown("- Consider invoice factoring")
        st.markdown("- Increase self-pay mix")
        st.markdown("- Build cash reserves")

# ==========================================
# SCENARIO ANALYSIS
# ==========================================

st.header("📊 Scenario Analysis")

scenarios = {
    "Current": {
        "churn": ongoing_churn,
        "self_pay": self_pay_pct if use_simple_payer else 0.10,
        "no_show": no_show_rate
    },
    "Conservative": {
        "churn": min(ongoing_churn + 0.03, 0.20),
        "self_pay": max((self_pay_pct if use_simple_payer else 0.10) - 0.10, 0),
        "no_show": min(no_show_rate * 1.3, 0.20)
    },
    "Optimistic": {
        "churn": max(ongoing_churn - 0.02, 0.02),
        "self_pay": min((self_pay_pct if use_simple_payer else 0.10) + 0.15, 0.50),
        "no_show": max(no_show_rate * 0.7, 0.02)
    }
}

scenario_results = []

for scenario_name, params in scenarios.items():
    scen_survival = 1.0
    scen_sessions = 0
    for i in range(1, 12):
        if i == 1:
            scen_survival *= (1 - month1_churn)
        elif i == 2:
            scen_survival *= (1 - month2_churn)
        elif i == 3:
            scen_survival *= (1 - month3_churn)
        else:
            scen_survival *= (1 - params["churn"])
        scen_sessions += scen_survival * avg_sessions_per_client_per_month
    
    scen_weighted_rate = (1 - params["self_pay"]) * avg_insurance_rate + params["self_pay"] * self_pay_rate
    scen_revenue_per_session = scen_weighted_rate * (1 - params["no_show"])
    scen_cost_per_session = lmsw_pay_per_session * 0.6 + lcsw_pay_per_session * 0.4
    scen_margin = scen_revenue_per_session - scen_cost_per_session
    
    scen_clv = scen_sessions * scen_margin
    scen_max_cac = scen_clv * 0.25
    
    scen_monthly_profit = avg_monthly_profit
    if scenario_name == "Conservative":
        scen_monthly_profit *= 0.7
    elif scenario_name == "Optimistic":
        scen_monthly_profit *= 1.4
    
    scenario_results.append({
        "Scenario": scenario_name,
        "CLV": scen_clv,
        "Max CAC": scen_max_cac,
        "Monthly Profit": scen_monthly_profit,
        "Lifetime Sessions": scen_sessions
    })

scenario_df = pd.DataFrame(scenario_results)

st.dataframe(scenario_df.style.format({
    "CLV": "${:,.0f}",
    "Max CAC": "${:,.0f}",
    "Monthly Profit": "${:,.0f}",
    "Lifetime Sessions": "{:.1f}"
}), use_container_width=True)

st.markdown("**Scenario Assumptions:**")
st.markdown(f"""
- **Conservative:** Churn +3%, Self-pay -10%, No-shows +30%
- **Optimistic:** Churn -2%, Self-pay +15%, No-shows -30%
""")

# Scenario visualization
fig_scenarios, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

scenario_df.plot(x="Scenario", y=["CLV", "Max CAC"], kind="bar", ax=ax_left, color=['steelblue', 'orange'])
ax_left.set_title("CLV & Max CAC by Scenario")
ax_left.set_ylabel("Dollars ($)")
ax_left.set_xlabel("")
ax_left.legend()
ax_left.grid(True, alpha=0.3, axis='y')

scenario_df.plot(x="Scenario", y="Monthly Profit", kind="bar", ax=ax_right, color='green')
ax_right.set_title("Monthly Profit by Scenario")
ax_right.set_ylabel("Profit ($)")
ax_right.set_xlabel("")
ax_right.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
st.pyplot(fig_scenarios)

# ==========================================
# SUMMARY
# ==========================================

st.header("📥 Key Takeaways")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Summary")
    st.markdown(f"""
    **Month {months_to_simulate}:**
    - Active Clients: {final_month['active_clients']:.0f}
    - Monthly Revenue: ${final_month['revenue_earned']:,.0f}
    - Monthly Profit: ${final_month['profit_accrual']:,.0f}
    - Cash Balance: ${final_month['cash_balance']:,.0f}
    
    **Team:**
    - {final_month['active_therapists']:.0f} therapists ({final_month['active_lmsw']:.0f} LMSW, {final_month['active_lcsw']:.0f} LCSW)
    - {final_month['capacity_utilization']:.1f}% capacity utilization
    
    **Economics:**
    - CLV: ${current_clv:,.0f}
    - Max CAC: ${current_clv * 0.25:,.0f}
    - Actual CAC: ${actual_cac:,.0f}
    """)

with col2:
    st.subheader("Top Priorities")
    priorities = []
    
    if actual_cac > max_profitable_cac:
        priorities.append(f"1. **Reduce CAC** - Currently ${actual_cac:.0f}, target ${max_profitable_cac:.0f}")
    
    if ongoing_churn > 0.03:
        priorities.append(f"2. **Improve Retention** - {ongoing_churn*100:.0f}% churn is high")
    
    if not offer_group_therapy and current_month["active_lcsw"] >= 2:
        priorities.append("3. **Launch Groups** - 75% margins available")
    
    if min_cash_balance < -5000:
        priorities.append(f"4. **Fix Cash Flow** - Need ${abs(min_cash_balance)*1.2:,.0f} working capital")
    
    if self_pay_pct < 0.15:
        priorities.append("5. **Increase Self-Pay** - Target 15%+")
    
    for p in priorities[:5]:
        st.markdown(p)

csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download Data (CSV)",
    data=csv,
    file_name=f"therapy_practice_{months_to_simulate}months.csv",
    mime="text/csv"
)
