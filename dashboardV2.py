import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

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
        """Therapist is active immediately when hired (already credentialed)"""
        return current_month >= self.hire_month
    
    def get_capacity_percentage(self, current_month):
        """Ramp up capacity over 4 months after hire"""
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
    
    def get_sessions_per_month(self, current_month):
        weeks_per_month = 52 / 12
        capacity_pct = self.get_capacity_percentage(current_month)
        # Utilization reduces target sessions
        return self.sessions_per_week_target * self.utilization_rate * weeks_per_month * capacity_pct

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("👤 Owner (You)")
owner_sessions_per_week = st.sidebar.number_input("Your Sessions per Week", min_value=0, value=20)
owner_utilization = st.sidebar.slider("Your Utilization Rate (%)", 0, 100, 85) / 100

st.sidebar.header("🧑‍⚕️ Therapist Hiring Schedule")
st.sidebar.markdown("**Hire Month** = Month when therapist starts seeing clients (already credentialed)")
st.sidebar.markdown("Set hire month to 0 to disable a therapist slot")

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
        hire_month = col1.number_input(f"Hire Month", 
                                       min_value=0, max_value=36, 
                                       value=default_month, key=f"hire_{i}",
                                       help="Month when therapist starts (0 = disabled)")
        
        if hire_month > 0:
            credential = col2.selectbox(f"Credential", 
                                       ["LMSW", "LCSW"], 
                                       index=0 if default_cred=="LMSW" else 1,
                                       key=f"cred_{i}")
            sessions = st.slider(f"Target Sessions/Week", 10, 30, 20, key=f"sess_{i}")
            util = st.slider(f"Utilization %", 50, 100, 85, key=f"util_{i}") / 100
            
            therapists.append(Therapist(i, f"Therapist {i}", credential, 
                                       hire_month, sessions, util))

st.sidebar.header("💰 Compensation & Taxes")
lmsw_pay_per_session = st.sidebar.number_input("LMSW Pay per Session ($)", 
                                                min_value=30.0, value=40.0, step=5.0)
lcsw_pay_per_session = st.sidebar.number_input("LCSW Pay per Session ($)", 
                                                min_value=35.0, value=50.0, step=5.0)
owner_takes_therapist_pay = st.sidebar.checkbox("Owner Takes Therapist Pay", value=True,
                                                 help="If checked, owner pays self like employee. Revenue nets out in P&L.")
payroll_tax_rate = st.sidebar.slider("Payroll Tax Rate (%)", 0.0, 20.0, 15.3, 0.1,
                                     help="FICA self-employment tax: 15.3%") / 100

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
    st.sidebar.info(f"{ehr_system}: Tiered pricing applied automatically")
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
avg_sessions_per_client_per_month = st.sidebar.number_input("Avg Sessions per Client per Month", value=2.5, step=0.1,
                                                             help="Planned frequency, before cancellations")
cancellation_rate = st.sidebar.slider("Cancellation Rate %", 0, 40, 20,
                                      help="% of scheduled sessions that get canceled (may reschedule)") / 100
no_show_rate = st.sidebar.slider("No-Show Rate %", 0, 30, 5,
                                 help="% of scheduled sessions where client doesn't show (not billed)") / 100
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

def calculate_supervision_costs(num_lmsw, owner_session_value):
    """
    Calculate supervision costs
    - Owner can supervise 3 LMSW: cost = foregone session per LMSW (~$100 each)
    - External LCSW supervises 2 LMSW at a time: $200/month per LMSW
    """
    if num_lmsw == 0:
        return 0, 0
    
    if num_lmsw <= 3:
        # Owner does supervision - opportunity cost
        supervision_cost = num_lmsw * owner_session_value
        external_cost = 0
    else:
        # Owner handles 3, rest need external LCSW
        owner_supervision_cost = 3 * owner_session_value
        external_supervisees = num_lmsw - 3
        external_cost = external_supervisees * 200  # $200/month per LMSW
        supervision_cost = owner_supervision_cost + external_cost
    
    return supervision_cost, external_cost

# ==========================================
# MAIN SIMULATION
# ==========================================

monthly_data = []
active_clients_count = 0
weeks_per_month = 52 / 12

revenue_by_payer_history = {payer: [] for payer in payer_mix.keys()}

# Calculate owner session value for supervision opportunity cost
owner_session_value = weighted_rate * (1 - no_show_rate)

for month in range(1, months_to_simulate + 1):
    month_data = {"month": month}
    
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
    
    # Scheduled sessions (before cancellations)
    scheduled_sessions = active_clients_count * avg_sessions_per_client_per_month
    
    # Apply cancellation rate (these don't happen, reduce session count)
    actual_sessions = scheduled_sessions * (1 - cancellation_rate)
    
    # Revenue only on sessions that happen (after no-shows)
    billable_sessions = actual_sessions * (1 - no_show_rate)
    
    month_data["scheduled_sessions"] = scheduled_sessions
    month_data["actual_sessions"] = actual_sessions
    month_data["billable_sessions"] = billable_sessions
    
    revenue_by_payer = {}
    for payer_name, payer_info in payer_mix.items():
        payer_sessions = billable_sessions * payer_info["pct"]
        payer_revenue = payer_sessions * payer_info["rate"]
        revenue_by_payer[payer_name] = payer_revenue
        revenue_by_payer_history[payer_name].append({
            "month": month,
            "revenue": payer_revenue,
            "delay_months": payer_info["delay_days"] / 30
        })
    
    copay_revenue = billable_sessions * copay_pct * avg_copay
    cc_fees = copay_revenue * cc_fee_pct
    
    total_revenue = sum(revenue_by_payer.values()) + copay_revenue - cc_fees
    
    month_data["copay_revenue"] = copay_revenue
    month_data["cc_fees"] = cc_fees
    month_data["revenue_earned"] = total_revenue
    
    # Collections (cash basis)
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
    
    # COSTS
    therapist_costs_pre_tax = 0
    owner_therapist_pay_pre_tax = 0
    
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
    
    # Apply payroll taxes
    therapist_costs = therapist_costs_pre_tax * (1 + payroll_tax_rate)
    owner_therapist_pay = owner_therapist_pay_pre_tax * (1 + payroll_tax_rate)
    
    # Supervision costs
    supervision_cost, external_supervision = calculate_supervision_costs(active_lmsw_count, owner_session_value)
    
    # EHR costs
    if ehr_system == "Custom":
        ehr_monthly_cost = ehr_cost_per_therapist * (len(active_therapists) + 1)
    else:
        cost_per = get_ehr_cost(len(active_therapists) + 1, ehr_system)
        ehr_monthly_cost = cost_per * (len(active_therapists) + 1)
    
    total_tech_cost = ehr_monthly_cost + telehealth_cost + other_tech_cost
    
    # Billing costs
    if billing_model == "Owner Does It":
        billing_cost = 0
    elif billing_model == "Billing Service (% of revenue)":
        billing_cost = total_revenue * billing_service_pct
    else:
        billing_cost = biller_monthly_cost
    
    # Marketing
    marketing_cost_actual = new_clients * client_acquisition_cost
    
    # Group therapy
    group_revenue = 0
    group_cost_pre_tax = 0
    if offer_group_therapy and active_lcsw_count > 0:
        num_groups = int(active_lcsw_count * pct_lcsw_doing_groups)
        group_revenue = num_groups * clients_per_group * group_revenue_per_client * group_sessions_per_month
        group_cost_pre_tax = num_groups * group_therapist_pay * group_sessions_per_month
    
    group_cost = group_cost_pre_tax * (1 + payroll_tax_rate)
    
    # Total costs
    total_costs = (therapist_costs + owner_therapist_pay + supervision_cost + 
                   total_tech_cost + billing_cost + marketing_cost_actual + other_overhead + group_cost)
    
    month_data["therapist_costs"] = therapist_costs
    month_data["owner_therapist_pay"] = owner_therapist_pay
    month_data["supervision_cost"] = supervision_cost
    month_data["tech_cost"] = total_tech_cost
    month_data["billing_cost"] = billing_cost
    month_data["marketing_budget"] = marketing_budget
    month_data["marketing_spent"] = marketing_cost_actual
    month_data["other_overhead"] = other_overhead
    month_data["group_revenue"] = group_revenue
    month_data["group_cost"] = group_cost
    month_data["total_costs"] = total_costs
    
    # Profit & Cash
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

# Main data table
st.header("📋 Monthly Financial Summary")
display_df = df[["month", "active_clients", "total_capacity_clients", "new_clients", "active_therapists", 
                 "revenue_earned", "collections", "total_costs", "profit_accrual", "cash_balance"]]
display_df.columns = ["Month", "Clients", "Capacity", "New", "Therapists", "Revenue", "Collections", "Costs", "Profit", "Cash"]
st.dataframe(display_df.style.format({
    "Revenue": "${:,.0f}",
    "Collections": "${:,.0f}",
    "Costs": "${:,.0f}",
    "Profit": "${:,.0f}",
    "Cash": "${:,.0f}",
    "Clients": "{:.1f}",
    "Capacity": "{:.1f}",
    "New": "{:.1f}"
}), use_container_width=True)

# Cost Breakdown Table
st.header("💰 Monthly Cost Breakdown")
cost_breakdown_df = df[["month", "therapist_costs", "owner_therapist_pay", "supervision_cost",
                        "tech_cost", "billing_cost", "marketing_spent", "group_cost", "other_overhead", "total_costs"]]
cost_breakdown_df.columns = ["Month", "Therapist Pay", "Owner Pay", "Supervision", 
                              "Technology", "Billing", "Marketing", "Group Cost", "Other Overhead", "Total"]
st.dataframe(cost_breakdown_df.style.format({
    "Therapist Pay": "${:,.0f}",
    "Owner Pay": "${:,.0f}",
    "Supervision": "${:,.0f}",
    "Technology": "${:,.0f}",
    "Billing": "${:,.0f}",
    "Marketing": "${:,.0f}",
    "Group Cost": "${:,.0f}",
    "Other Overhead": "${:,.0f}",
    "Total": "${:,.0f}"
}), use_container_width=True)

# Charts
st.header("📈 Key Metrics Over Time")

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Client growth
ax1.plot(df["month"], df["active_clients"], marker='o', linewidth=2, color='steelblue', label='Active Clients')
ax1.plot(df["month"], df["total_capacity_clients"], linestyle='--', color='red', label='Capacity')
ax1.fill_between(df["month"], df["active_clients"], df["total_capacity_clients"], alpha=0.2, color='orange')
ax1.set_title("Client Growth vs Capacity")
ax1.set_xlabel("Month")
ax1.set_ylabel("Clients")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Revenue & Collections
ax2.plot(df["month"], df["revenue_earned"], marker='o', linewidth=2, label='Revenue (Accrual)', linestyle='--', alpha=0.7)
ax2.plot(df["month"], df["collections"], marker='s', linewidth=2, label='Collections (Cash)')
ax2.plot(df["month"], df["total_costs"], marker='^', linewidth=2, label='Costs', color='red')
ax2.set_title("Revenue, Collections & Costs")
ax2.set_xlabel("Month")
ax2.set_ylabel("Dollars ($)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Cash balance
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

# Cost Breakdown Stacked Area
ax4.stackplot(df["month"], 
              df["therapist_costs"], df["owner_therapist_pay"], df["supervision_cost"],
              df["tech_cost"], df["billing_cost"], df["marketing_spent"], df["other_overhead"],
              labels=['Therapist Pay', 'Owner Pay', 'Supervision', 'Tech', 'Billing', 'Marketing', 'Other'],
              alpha=0.8)
ax4.set_title("Cost Breakdown Over Time (Stacked)")
ax4.set_xlabel("Month")
ax4.set_ylabel("Costs ($)")
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig1)

# ==========================================
# ANNUAL P&L STATEMENTS
# ==========================================

st.header("📊 Annual Profit & Loss Statements")

# Calculate years
years = []
for year_num in range(1, (months_to_simulate // 12) + 2):
    start_month = (year_num - 1) * 12 + 1
    end_month = min(year_num * 12, months_to_simulate)
    
    if start_month <= months_to_simulate:
        year_data = df[(df['month'] >= start_month) & (df['month'] <= end_month)]
        
        months_in_year = len(year_data)
        annualized_factor = 12 / months_in_year  # Annualize partial years
        
        years.append({
            'Year': f"Year {year_num}" + (f" ({months_in_year} months)" if months_in_year < 12 else ""),
            'Revenue': year_data['revenue_earned'].sum(),
            'Group Revenue': year_data['group_revenue'].sum(),
            'Total Revenue': year_data['revenue_earned'].sum() + year_data['group_revenue'].sum(),
            'Therapist Costs': year_data['therapist_costs'].sum(),
            'Owner Pay': year_data['owner_therapist_pay'].sum(),
            'Supervision': year_data['supervision_cost'].sum(),
            'Technology': year_data['tech_cost'].sum(),
            'Billing': year_data['billing_cost'].sum(),
            'Marketing': year_data['marketing_spent'].sum(),
            'Other Overhead': year_data['other_overhead'].sum(),
            'Group Costs': year_data['group_cost'].sum(),
            'Total Costs': year_data['total_costs'].sum(),
            'Net Profit': year_data['profit_accrual'].sum(),
            'Profit Margin %': (year_data['profit_accrual'].sum() / (year_data['revenue_earned'].sum() + year_data['group_revenue'].sum()) * 100) if (year_data['revenue_earned'].sum() + year_data['group_revenue'].sum()) > 0 else 0,
            'Avg Monthly Profit': year_data['profit_accrual'].mean(),
            'Annualized Profit': year_data['profit_accrual'].sum() * annualized_factor
        })

annual_pl_df = pd.DataFrame(years)
st.dataframe(annual_pl_df.style.format({
    'Revenue': '${:,.0f}',
    'Group Revenue': '${:,.0f}',
    'Total Revenue': '${:,.0f}',
    'Therapist Costs': '${:,.0f}',
    'Owner Pay': '${:,.0f}',
    'Supervision': '${:,.0f}',
    'Technology': '${:,.0f}',
    'Billing': '${:,.0f}',
    'Marketing': '${:,.0f}',
    'Other Overhead': '${:,.0f}',
    'Group Costs': '${:,.0f}',
    'Total Costs': '${:,.0f}',
    'Net Profit': '${:,.0f}',
    'Profit Margin %': '{:.1f}%',
    'Avg Monthly Profit': '${:,.0f}',
    'Annualized Profit': '${:,.0f}'
}), use_container_width=True)

# ==========================================
# PER-THERAPIST ANNUAL P&L
# ==========================================

st.header("👥 Per-Therapist Annual P&L Analysis")

st.markdown("""
This shows the average annual P&L contribution per therapist for each year, 
based on their target sessions and the practice's financial model.
""")

therapist_annual_data = []

for therapist in therapists:
    if therapist.id == 0 or therapist.hire_month == 0:
        continue  # Skip owner and disabled therapists
    
    for year_num in range(1, (months_to_simulate // 12) + 2):
        start_month = (year_num - 1) * 12 + 1
        end_month = min(year_num * 12, months_to_simulate)
        
        if start_month <= months_to_simulate:
            # Only include months where therapist is active
            therapist_months = [m for m in range(start_month, end_month + 1) if therapist.is_active(m)]
            
            if len(therapist_months) > 0:
                # Calculate annual metrics for this therapist
                annual_sessions = sum([therapist.get_sessions_per_month(m) for m in therapist_months])
                annualized_sessions = annual_sessions * (12 / len(therapist_months))
                
                # Revenue
                annual_revenue = annual_sessions * weighted_rate * (1 - cancellation_rate) * (1 - no_show_rate)
                annualized_revenue = annual_revenue * (12 / len(therapist_months))
                
                # Direct costs (therapist pay)
                if therapist.credential == "LMSW":
                    pay_rate = lmsw_pay_per_session
                else:
                    pay_rate = lcsw_pay_per_session
                
                annual_pay = annual_sessions * pay_rate * (1 + payroll_tax_rate)
                annualized_pay = annual_pay * (12 / len(therapist_months))
                
                # Allocated overhead (tech + portion of other overhead)
                avg_therapists_this_year = df[(df['month'] >= start_month) & (df['month'] <= end_month)]['active_therapists'].mean()
                allocated_tech = ehr_monthly_cost / avg_therapists_this_year if avg_therapists_this_year > 0 else 0
                allocated_overhead = (other_overhead + telehealth_cost + other_tech_cost) / avg_therapists_this_year if avg_therapists_this_year > 0 else 0
                
                annual_allocated_costs = (allocated_tech + allocated_overhead) * len(therapist_months)
                annualized_allocated_costs = annual_allocated_costs * (12 / len(therapist_months))
                
                # Supervision cost (if LMSW)
                if therapist.credential == "LMSW":
                    # Check if this therapist is in first 3 (owner supervises) or external
                    lmsw_rank = len([t for t in therapists if t.credential == "LMSW" and t.hire_month <= therapist.hire_month and t.hire_month > 0])
                    if lmsw_rank <= 3:
                        supervision_cost_annual = owner_session_value * len(therapist_months)
                    else:
                        supervision_cost_annual = 200 * len(therapist_months)
                    annualized_supervision = supervision_cost_annual * (12 / len(therapist_months))
                else:
                    supervision_cost_annual = 0
                    annualized_supervision = 0
                
                # Net contribution
                annual_contribution = annual_revenue - annual_pay - annual_allocated_costs - supervision_cost_annual
                annualized_contribution = annual_contribution * (12 / len(therapist_months))
                
                therapist_annual_data.append({
                    'Therapist': therapist.name,
                    'Credential': therapist.credential,
                    'Year': f"Year {year_num}",
                    'Months Active': len(therapist_months),
                    'Sessions (Actual)': int(annual_sessions),
                    'Sessions (Annualized)': int(annualized_sessions),
                    'Revenue (Annualized)': annualized_revenue,
                    'Therapist Pay': annualized_pay,
                    'Supervision': annualized_supervision,
                    'Allocated Overhead': annualized_allocated_costs,
                    'Net Contribution': annualized_contribution,
                    'Contribution Margin %': (annualized_contribution / annualized_revenue * 100) if annualized_revenue > 0 else 0
                })

if therapist_annual_data:
    therapist_pl_df = pd.DataFrame(therapist_annual_data)
    st.dataframe(therapist_pl_df.style.format({
        'Sessions (Actual)': '{:,.0f}',
        'Sessions (Annualized)': '{:,.0f}',
        'Revenue (Annualized)': '${:,.0f}',
        'Therapist Pay': '${:,.0f}',
        'Supervision': '${:,.0f}',
        'Allocated Overhead': '${:,.0f}',
        'Net Contribution': '${:,.0f}',
        'Contribution Margin %': '{:.1f}%'
    }), use_container_width=True)
else:
    st.info("No employed therapists configured. Add therapists in the sidebar.")

# ==========================================
# STRATEGIC RECOMMENDATIONS
# ==========================================

st.header("🎯 Strategic Recommendations")

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

contribution_margin = weighted_rate * (1 - cancellation_rate) * (1 - no_show_rate) - (lmsw_pay_per_session * 0.7 + lcsw_pay_per_session * 0.3) * (1 + payroll_tax_rate)
current_clv = total_expected_sessions * contribution_margin

avg_monthly_profit = df["profit_accrual"].mean()
total_new_clients = df["new_clients"].sum()
total_marketing_spent = df["marketing_spent"].sum()
actual_cac = total_marketing_spent / total_new_clients if total_new_clients > 0 else 0
max_profitable_cac = current_clv * 0.25

recommendations = []

# Key metrics summary
col1, col2, col3 = st.columns(3)
col1.metric("Client Lifetime Value", f"${current_clv:,.0f}")
col2.metric("Max Affordable CAC", f"${max_profitable_cac:,.0f}")
col3.metric("Actual CAC", f"${actual_cac:,.0f}", 
           delta=f"${actual_cac - max_profitable_cac:,.0f}" if actual_cac > max_profitable_cac else f"${max_profitable_cac - actual_cac:,.0f}",
           delta_color="inverse" if actual_cac > max_profitable_cac else "normal")

# Generate recommendations
if final_month["active_lmsw"] > 0:
    lmsw_pct = final_month["active_lmsw"] / max(final_month["active_therapists"], 1) * 100
    supervision_cost_monthly = df["supervision_cost"].mean()
    
    with st.expander("💰 Supervision Leverage Strategy", expanded=(supervision_cost_monthly > 300)):
        st.markdown(f"**Current Mix:** {lmsw_pct:.0f}% LMSW ({final_month['active_lmsw']:.0f} LMSW, {final_month['active_lcsw']:.0f} LCSW)")
        st.markdown(f"**Monthly Supervision Cost:** ${supervision_cost_monthly:.0f}")
        
        if final_month['active_lmsw'] <= 3:
            st.info(f"✅ Optimal range: You're supervising {final_month['active_lmsw']:.0f} LMSW (within your 3-person capacity)")
        else:
            st.warning(f"⚠️ Above optimal: {final_month['active_lmsw'] - 3:.0f} LMSW require external supervision at $200/month each")

if actual_cac > max_profitable_cac:
    with st.expander("⚠️ Marketing Efficiency - CAC Too High", expanded=True):
        st.markdown(f"**Current CAC:** ${actual_cac:.0f}")
        st.markdown(f"**Target CAC:** ${max_profitable_cac:.0f} (25% of CLV)")
        st.markdown(f"**Overspending:** ${(actual_cac - max_profitable_cac) * total_new_clients:,.0f} over {months_to_simulate} months")
        st.markdown("**Actions:**")
        st.markdown("- Improve conversion rate on leads")
        st.markdown("- Focus on referrals (lower CAC)")
        st.markdown("- Increase retention to boost CLV")

if cancellation_rate + no_show_rate > 0.20:
    total_lost_sessions = df["scheduled_sessions"].sum() * (cancellation_rate + no_show_rate)
    lost_revenue = total_lost_sessions * weighted_rate
    
    with st.expander("📉 High Cancellation/No-Show Rate", expanded=True):
        st.markdown(f"**Combined Rate:** {(cancellation_rate + no_show_rate)*100:.0f}%")
        st.markdown(f"**Lost Revenue:** ${lost_revenue:,.0f} over {months_to_simulate} months")
        st.markdown("**Actions:**")
        st.markdown("- 24hr cancellation policy")
        st.markdown("- Automated reminders")
        st.markdown("- Credit card on file")
        st.markdown("- Address barriers to attendance")

min_cash = df["cash_balance"].min()
if min_cash < -5000:
    with st.expander("💸 Working Capital Gap", expanded=True):
        st.markdown(f"**Minimum Cash Balance:** ${min_cash:,.0f}")
        st.markdown(f"**Working Capital Needed:** ${abs(min_cash)*1.2:,.0f}")
        st.markdown("**Actions:**")
        st.markdown("- Secure line of credit")
        st.markdown("- Increase self-pay mix (immediate payment)")
        st.markdown("- Build cash reserves during positive months")

# Download
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download Monthly Data (CSV)",
    data=csv,
    file_name=f"therapy_practice_{months_to_simulate}months.csv",
    mime="text/csv"
)

st.success("✅ Dashboard Complete - All Features Active")
