# Insights
    st.markdown("---")
    st.markdown("**Key Insights:**")
    
    if test_ongoing_churn < ongoing_churn:
        churn_impact = ((ongoing_churn - test_ongoing_churn) / ongoing_churn) * 100
        st.success(f"✅ Reducing churn by {churn_impact:.0f}% adds ${clv_delta:,.0f} per client")
    elif test_ongoing_churn > ongoing_churn:
        st.warning(f"⚠️ Higher churn reduces CLV by ${abs(clv_delta):,.0f} per client")
    
    if test_self_pay_pct > (self_pay_pct if use_simple_payer else 0.10):
        rate_improvement = test_weighted_rate - weighted_rate
        st.success(f"✅ More self-pay increases avg rate by ${rate_improvement:.0f}/session")
    
    if test_no_show < no_show_rate:
        sessions_saved = (no_show_rate - test_no_show) * test_monthly_sessions
        revenue_saved = sessions_saved * test_weighted_rate
        st.success(f"✅ Reducing no-shows saves {sessions_saved:.0f} sessions/month = ${revenue_saved:,.0f}")
    
    if test_monthly_profit > avg_monthly_profit * 1.2:
        st.success(f"🚀 This configuration increases profit by {(test_monthly_profit/avg_monthly_profit - 1)*100:.0f}%!")
    elif test_monthly_profit < avg_monthly_profit * 0.8:
        st.error(f"📉 This configuration reduces profit by {(1 - test_monthly_profit/avg_monthly_profit)*100:.0f}%")

# ==========================================
# BREAK-EVEN & TARGETS CALCULATOR
# ==========================================

st.header("🎯 Break-Even & Target Calculator")

target_col1, target_col2 = st.columns(2)

with target_col1:
    st.subheader("Break-Even Analysis")
    
    # Fixed costs
    fixed_monthly_costs = (df["tech_cost"].mean() + df["other_overhead"].mean() + 
                          df["billing_cost"].mean())
    
    # Variable costs per session
    avg_pay_per_session = (lmsw_pay_per_session * 0.6 + lcsw_pay_per_session * 0.4)
    variable_cost_per_session = avg_pay_per_session
    
    # Contribution margin per session
    cm_per_session = weighted_rate * (1 - no_show_rate) - variable_cost_per_session
    
    # Break-even sessions
    if cm_per_session > 0:
        breakeven_sessions = fixed_monthly_costs / cm_per_session
        breakeven_clients = breakeven_sessions / avg_sessions_per_client_per_month
        
        st.metric("Break-Even Sessions/Month", f"{breakeven_sessions:.0f}")
        st.metric("Break-Even Clients", f"{breakeven_clients:.0f}")
        st.metric("Current vs Break-Even", f"{avg_clients - breakeven_clients:+.0f} clients")
        
        if avg_clients > breakeven_clients:
            st.success(f"✅ You're {((avg_clients - breakeven_clients) / breakeven_clients * 100):.0f}% above break-even")
        else:
            st.error(f"⚠️ Need {breakeven_clients - avg_clients:.0f} more clients to break even")
    else:
        st.error("⚠️ Negative contribution margin - costs exceed revenue per session!")

with target_col2:
    st.subheader("Target Profit Calculator")
    
    target_monthly_profit = st.number_input("Target Monthly Profit ($)", 
                                           min_value=0, value=10000, step=1000)
    
    if cm_per_session > 0:
        sessions_needed = (fixed_monthly_costs + target_monthly_profit) / cm_per_session
        clients_needed = sessions_needed / avg_sessions_per_client_per_month
        therapists_needed = clients_needed / (20 * (52/12) * 0.85 / avg_sessions_per_client_per_month)
        
        st.metric("Sessions Needed/Month", f"{sessions_needed:.0f}")
        st.metric("Clients Needed", f"{clients_needed:.0f}")
        st.metric("Therapists Needed", f"{therapists_needed:.1f}")
        
        gap = clients_needed - avg_clients
        if gap > 0:
            months_to_target = gap / df["new_clients"].mean() if df["new_clients"].mean() > 0 else 999
            st.info(f"Need {gap:.0f} more clients. At current growth rate: {months_to_target:.0f} months to reach target.")
        else:
            st.success(f"✅ Already exceeding target by ${(avg_monthly_profit - target_monthly_profit):,.0f}/month")

# ==========================================
# THERAPIST HIRING TIMELINE VISUALIZATION
# ==========================================

st.header("👥 Therapist Hiring Timeline")

# Create timeline visualization
timeline_data = []
for therapist in therapists:
    if therapist.id > 0:  # Skip owner
        timeline_data.append({
            "Therapist": therapist.name,
            "Credential": therapist.credential,
            "Hire Month": therapist.hire_month,
            "Credentialed Month": therapist.hire_month + 3,
            "Full Capacity Month": therapist.hire_month + 7,
            "Status": "Active" if therapist.is_active(months_to_simulate) else "Ramping Up" if therapist.hire_month <= months_to_simulate else "Future"
        })

if timeline_data:
    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True)
    
    # Gantt-style chart
    fig_timeline, ax_timeline = plt.subplots(figsize=(12, max(4, len(timeline_data) * 0.5)))
    
    colors = {"LMSW": "orange", "LCSW": "purple"}
    
    for idx, therapist_row in enumerate(timeline_data):
        # Credentialing period
        ax_timeline.barh(idx, 3, left=therapist_row["Hire Month"], 
                        color='gray', alpha=0.3, height=0.6, label='Credentialing' if idx==0 else '')
        # Ramp-up period
        ax_timeline.barh(idx, 4, left=therapist_row["Credentialed Month"], 
                        color=colors[therapist_row["Credential"]], alpha=0.5, height=0.6, 
                        label=f'{therapist_row["Credential"]} Ramp-Up' if idx==0 else '')
        # Full capacity
        ax_timeline.barh(idx, max(1, months_to_simulate - therapist_row["Full Capacity Month"]), 
                        left=therapist_row["Full Capacity Month"], 
                        color=colors[therapist_row["Credential"]], alpha=0.9, height=0.6,
                        label=f'{therapist_row["Credential"]} Full' if idx==0 else '')
    
    ax_timeline.set_yticks(range(len(timeline_data)))
    ax_timeline.set_yticklabels([t["Therapist"] for t in timeline_data])
    ax_timeline.set_xlabel("Month")
    ax_timeline.set_title("Therapist Hiring & Ramp-Up Timeline")
    ax_timeline.grid(True, alpha=0.3, axis='x')
    ax_timeline.legend(loc='upper right')
    
    st.pyplot(fig_timeline)
else:
    st.info("No additional therapists configured. Add therapists in the sidebar to see timeline.")

# ==========================================
# CAPACITY PLANNING
# ==========================================

st.header("📈 Capacity Planning & Utilization")

capacity_metrics = df[["month", "active_clients", "total_capacity_clients", "capacity_utilization", "new_clients"]].copy()

fig_capacity, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Capacity fill rate
ax1.plot(capacity_metrics["month"], capacity_metrics["active_clients"], 
         marker='o', linewidth=2, label='Active Clients', color='steelblue')
ax1.plot(capacity_metrics["month"], capacity_metrics["total_capacity_clients"], 
         linestyle='--', linewidth=2, label='Total Capacity', color='red')
ax1.fill_between(capacity_metrics["month"], capacity_metrics["active_clients"], 
                  capacity_metrics["total_capacity_clients"], alpha=0.2, color='orange', 
                  label='Unused Capacity')
ax1.set_xlabel("Month")
ax1.set_ylabel("Clients")
ax1.set_title("Capacity Utilization Over Time")
ax1.legend()
ax1.grid(True, alpha=0.3)

# New client acquisition rate
ax2.bar(capacity_metrics["month"], capacity_metrics["new_clients"], color='green', alpha=0.7)
ax2.axhline(y=capacity_metrics["new_clients"].mean(), color='blue', linestyle='--', 
            label=f'Average: {capacity_metrics["new_clients"].mean():.1f} clients/month')
ax2.set_xlabel("Month")
ax2.set_ylabel("New Clients")
ax2.set_title("Client Acquisition Rate")
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

st.pyplot(fig_capacity)

# Capacity insights
avg_utilization = capacity_metrics["capacity_utilization"].mean()
final_utilization = capacity_metrics["capacity_utilization"].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Average Utilization", f"{avg_utilization:.1f}%")
col2.metric("Current Utilization", f"{final_utilization:.1f}%")
col3.metric("Unused Capacity", f"{capacity_metrics['total_capacity_clients'].iloc[-1] - capacity_metrics['active_clients'].iloc[-1]:.0f} clients")

if avg_utilization < 60:
    st.warning(f"""
    ⚠️ **Low Capacity Utilization ({avg_utilization:.0f}%)**
    
    You have significant unused capacity. Consider:
    - Increasing marketing budget to fill faster
    - Slowing therapist hiring until utilization improves
    - Reducing CAC to make growth more profitable
    """)
elif avg_utilization > 90:
    st.success(f"""
    ✅ **High Capacity Utilization ({avg_utilization:.0f}%)**
    
    You're efficiently using capacity. Consider:
    - Hiring additional therapists to expand
    - Building waitlist for future capacity
    - Raising rates due to high demand
    """)

# ==========================================
# EXPORT & SUMMARY
# ==========================================

st.header("📥 Export & Summary")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Key Takeaways")
    st.markdown(f"""
    **Financial Summary (Month {months_to_simulate}):**
    - Active Clients: {final_month['active_clients']:.0f}
    - Monthly Revenue: ${final_month['revenue_earned']:,.0f}
    - Monthly Profit: ${final_month['profit_accrual']:,.0f}
    - Profit Margin: {(final_month['profit_accrual']/final_month['revenue_earned']*100):.1f}%
    - Cash Balance: ${final_month['cash_balance']:,.0f}
    
    **Team Composition:**
    - Active Therapists: {final_month['active_therapists']:.0f} ({final_month['active_lmsw']:.0f} LMSW, {final_month['active_lcsw']:.0f} LCSW)
    - Capacity Utilization: {final_month['capacity_utilization']:.1f}%
    
    **Client Economics:**
    - Lifetime Value: ${current_clv:,.0f}
    - Max Affordable CAC: ${current_clv * 0.25:,.0f}
    - Actual CAC: ${actual_cac:,.0f}
    - Lifetime Sessions: {total_expected_sessions:.1f}
    """)

with col2:
    st.subheader("Top 3 Priorities")
    
    top_3 = [r for r in recommendations if r["priority"] == 1][:3]
    if not top_3:
        top_3 = recommendations[:3]
    
    for i, rec in enumerate(top_3, 1):
        st.markdown(f"""
        **{i}. {rec['title']}**
        - {rec['financial_impact']}
        - {rec['recommendation']}
        """)

# Download button for data
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download Full Data (CSV)",
    data=csv,
    file_name=f"therapy_practice_analysis_{months_to_simulate}months.csv",
    mime="text/csv"
)

st.success("🎉 Complete Financial Analysis Generated!")

# ==========================================
# INTERACTIVE WHAT-IF ANALYZER
# ==========================================

st.header("🔬 What-If Analysis - Test Your Assumptions")

st.markdown("""
Adjust key variables below to see immediate impact on your practice economics. 
Changes are compared against your current model.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎛️ Adjust Variables")
    
    test_ongoing_churn = st.slider("Test: Ongoing Churn %", 0, 20, int(ongoing_churn*100), key="test_churn") / 100
    test_self_pay_pct = st.slider("Test: Self-Pay %", 0, 50, int((self_pay_pct if use_simple_payer else 10)*100), key="test_selfpay") / 100
    test_no_show = st.slider("Test: No-Show %", 0, 20, int(no_show_rate*100), key="test_noshow") / 100
    test_therapist_count = st.slider("Test: Number of Therapists", 1, 15, len([t for t in therapists if t.id > 0]), key="test_therapists")
    test_marketing_budget = st.slider("Test: Monthly Marketing Budget", 0, 10000, int(df["marketing_budget"].mean()), step=100, key="test_marketing")

with col2:
    st.subheader("📊 Impact Analysis")
    
    # Recalculate CLV with test parameters
    test_survival = 1.0
    test_lifetime_sessions = 0
    for i in range(1, 12):
        if i == 1:
            test_survival *= (1 - month1_churn)
        elif i == 2:
            test_survival *= (1 - month2_churn)
        elif i == 3:
            test_survival *= (1 - month3_churn)
        else:
            test_survival *= (1 - test_ongoing_churn)
        test_lifetime_sessions += test_survival * avg_sessions_per_client_per_month
    
    # Revenue calculation
    test_weighted_rate = (1 - test_self_pay_pct) * avg_insurance_rate + test_self_pay_pct * self_pay_rate
    test_revenue_per_session = test_weighted_rate * (1 - test_no_show)
    
    test_clv = test_lifetime_sessions * (test_revenue_per_session - lcsw_pay_per_session * 0.6 - lmsw_pay_per_session * 0.4)
    test_max_cac = test_clv * 0.25
    
    # Profit estimation
    test_monthly_sessions = test_therapist_count * 20 * (52/12) * 0.85  # Rough estimate
    test_monthly_revenue = test_monthly_sessions * test_revenue_per_session
    test_monthly_costs = (test_therapist_count * lcsw_pay_per_session * 0.5 * test_monthly_sessions / test_therapist_count +
                         test_therapist_count * 150 + test_marketing_budget + 2000)
    test_monthly_profit = test_monthly_revenue - test_monthly_costs
    
    # Show deltas
    clv_delta = test_clv - current_clv
    cac_delta = test_max_cac - (current_clv * 0.25)
    profit_delta = test_monthly_profit - avg_monthly_profit
    
    st.metric("Client Lifetime Value", f"${test_clv:,.0f}", 
             delta=f"${clv_delta:,.0f} ({clv_delta/current_clv*100:+.1f}%)" if current_clv > 0 else "")
    
    st.metric("Max Affordable CAC", f"${test_max_cac:,.0f}",
             delta=f"${cac_delta:,.0f}" if current_clv > 0 else "")
    
    st.metric("Est. Monthly Profit", f"${test_monthly_profit:,.0f}",
             delta=f"${profit_delta:,.0f} ({profit_delta/avg_monthly_profit*100:+.1f}%)" if avg_monthly_profit > 0 else "")
    
    st.metric("Lifetime Sessions", f"{test_lifetime_sessions:.1f}",
             delta=f"{test_lifetime_sessions - total_expected_sessions:+.1f}")
    
    # Insights
    st.markdown("---")
    st.markdown("**Key Insights:**")
    
    if test_ongoing_churn < ongoing_churn:
        churn_impact = ((ongoing_churn - test_ongoing_churn) / ongoing_churn) * 100
        st.success(f"✅ Reducing churn by {churn_impact:.import streamlit as st
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
    credential: str  # LMSW or LCSW
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
owner_credential = "LCSW"  # Owner is always LCSW

st.sidebar.header("🧑‍⚕️ Therapist Hiring Schedule")
st.sidebar.markdown("Configure up to 12 therapists. Set hire month to 0 to disable.")

therapists = []
# Owner is always therapist 0
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
    
    # Simple payer mix
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

# Calculate weighted average
weighted_rate = sum(p["pct"] * p["rate"] for p in payer_mix.values())
weighted_delay = sum(p["pct"] * p["delay_days"] for p in payer_mix.values()) / 30  # Convert to months

st.sidebar.header("💻 Technology & Overhead")
ehr_system = st.sidebar.selectbox("EHR System", 
                                   ["SimplePractice", "TherapyNotes", "Custom"])

if ehr_system == "Custom":
    ehr_cost_per_therapist = st.sidebar.number_input("EHR Cost per Therapist", value=75.0)
else:
    st.sidebar.info(f"{ehr_system}: Tiered pricing (calculated automatically)")
    ehr_cost_per_therapist = None  # Will calculate based on tier

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
    """Calculate supervision costs - owner handles up to 3, then external needed"""
    if num_lmsw == 0:
        return 0, 0
    
    if num_lmsw <= 3:
        # Owner handles supervision
        owner_time_cost = 4  # 4 hours per month
        external_cost = 0
    else:
        # Owner handles 3, rest need external
        owner_time_cost = 4
        external_supervisees = num_lmsw - 3
        external_cost = external_supervisees * 150  # $150/month per LMSW
    
    return owner_time_cost, external_cost

def get_churn_rate(client_months_active):
    """Get churn rate based on how long client has been active"""
    if client_months_active == 1:
        return month1_churn
    elif client_months_active == 2:
        return month2_churn
    elif client_months_active == 3:
        return month3_churn
    else:
        return ongoing_churn

# ==========================================
# MAIN SIMULATION
# ==========================================

# Initialize tracking
monthly_data = []
active_clients_count = 0
weeks_per_month = 52 / 12

# Collections tracking (for cash flow with delays)
revenue_by_payer_history = {payer: [] for payer in payer_mix.keys()}

for month in range(1, months_to_simulate + 1):
    month_data = {"month": month}
    
    # ----- CAPACITY CALCULATION -----
    active_therapists = [t for t in therapists if t.is_active(month)]
    total_therapists_hired = len([t for t in therapists if t.hire_month > 0 and t.hire_month <= month])
    
    # Calculate total capacity
    total_capacity_sessions = sum(t.get_sessions_per_month(month) for t in therapists)
    total_capacity_clients = total_capacity_sessions / avg_sessions_per_client_per_month if avg_sessions_per_client_per_month > 0 else 0
    
    # Count credentials
    active_lmsw_count = len([t for t in active_therapists if t.credential == "LMSW"])
    active_lcsw_count = len([t for t in active_therapists if t.credential == "LCSW"])
    
    month_data["total_therapists_hired"] = total_therapists_hired
    month_data["active_therapists"] = len(active_therapists)
    month_data["active_lmsw"] = active_lmsw_count
    month_data["active_lcsw"] = active_lcsw_count
    month_data["total_capacity_sessions"] = total_capacity_sessions
    month_data["total_capacity_clients"] = total_capacity_clients
    
    # ----- CLIENT FLOW -----
    # Churn
    churned_clients = active_clients_count * ongoing_churn
    surviving_clients = active_clients_count - churned_clients
    
    # Marketing budget (dynamic based on model)
    if marketing_model == "Fixed Monthly":
        marketing_budget = base_marketing_budget
    elif marketing_model == "Per Active Therapist":
        marketing_budget = base_marketing + (len(active_therapists) * marketing_per_therapist)
    else:  # Per Empty Capacity Slot
        empty_slots = max(0, total_capacity_clients - surviving_clients)
        marketing_budget = empty_slots * marketing_per_empty_slot
    
    # New clients
    capacity_available = max(0, total_capacity_clients - surviving_clients)
    max_new_clients_from_budget = marketing_budget / client_acquisition_cost if client_acquisition_cost > 0 else 0
    new_clients = min(max_new_clients_from_budget, capacity_available)
    
    active_clients_count = surviving_clients + new_clients
    
    month_data["churned_clients"] = churned_clients
    month_data["new_clients"] = new_clients
    month_data["active_clients"] = active_clients_count
    month_data["capacity_utilization"] = (active_clients_count / total_capacity_clients * 100) if total_capacity_clients > 0 else 0
    
    # ----- REVENUE CALCULATION -----
    expected_sessions = active_clients_count * avg_sessions_per_client_per_month
    
    # Calculate revenue by payer
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
    
    # Copay revenue and CC fees
    copay_revenue = expected_sessions * copay_pct * avg_copay * (1 - no_show_rate)
    cc_fees = copay_revenue * cc_fee_pct
    
    total_revenue = sum(revenue_by_payer.values()) + copay_revenue - cc_fees
    
    month_data["expected_sessions"] = expected_sessions
    month_data["revenue_earned"] = total_revenue
    month_data["copay_revenue"] = copay_revenue
    month_data["cc_fees"] = cc_fees
    
    # ----- COLLECTIONS (CASH FLOW) -----
    collections = 0
    for payer_name, payer_info in payer_mix.items():
        delay_months = payer_info["delay_days"] / 30
        delay_floor = int(delay_months)
        delay_fraction = delay_months - delay_floor
        
        # Get revenue from delayed months
        if delay_floor == 0:
            # Immediate payment (self-pay)
            collections += revenue_by_payer[payer_name]
        else:
            # Delayed payment
            delay_idx_floor = month - 1 - delay_floor
            if 0 <= delay_idx_floor < len(revenue_by_payer_history[payer_name]):
                collections += revenue_by_payer_history[payer_name][delay_idx_floor]["revenue"] * (1 - delay_fraction)
            
            if delay_fraction > 0:
                delay_idx_ceil = month - 1 - (delay_floor + 1)
                if 0 <= delay_idx_ceil < len(revenue_by_payer_history[payer_name]):
                    collections += revenue_by_payer_history[payer_name][delay_idx_ceil]["revenue"] * delay_fraction
    
    # Add copay (immediate)
    collections += copay_revenue - cc_fees
    
    month_data["collections"] = collections
    
    # ----- COSTS -----
    # Therapist pay
    therapist_costs = 0
    owner_therapist_pay_amount = 0
    
    for therapist in therapists:
        if therapist.is_active(month):
            therapist_sessions = therapist.get_sessions_per_month(month)
            # Allocate sessions proportionally to this therapist
            therapist_actual_sessions = (therapist_sessions / total_capacity_sessions) * expected_sessions if total_capacity_sessions > 0 else 0
            
            if therapist.credential == "LMSW":
                pay = therapist_actual_sessions * lmsw_pay_per_session
            else:
                pay = therapist_actual_sessions * lcsw_pay_per_session
            
            if therapist.id == 0 and owner_takes_therapist_pay:
                owner_therapist_pay_amount = pay
            else:
                therapist_costs += pay
    
    # Supervision costs
    owner_supervision_hours, external_supervision_cost = calculate_supervision_costs(active_lmsw_count)
    
    # Tech costs (with tiered pricing)
    if ehr_system == "Custom":
        ehr_monthly_cost = ehr_cost_per_therapist * (len(active_therapists) + 1)  # +1 for owner
    else:
        cost_per = get_ehr_cost(len(active_therapists) + 1, ehr_system)
        ehr_monthly_cost = cost_per * (len(active_therapists) + 1)
    
    total_tech_cost = ehr_monthly_cost + telehealth_cost + other_tech_cost
    
    # Billing costs
    if billing_model == "Owner Does It":
        billing_cost = 0
        owner_billing_hours = 15
    elif billing_model == "Billing Service (% of revenue)":
        billing_cost = total_revenue * billing_service_pct
        owner_billing_hours = 0
    else:  # In-house biller
        billing_cost = biller_monthly_cost
        owner_billing_hours = 0
    
    # Marketing costs
    marketing_cost_actual = new_clients * client_acquisition_cost
    
    # Group therapy
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
    
    # ----- PROFIT -----
    profit_accrual = total_revenue + group_revenue - total_costs
    cash_flow = collections + group_revenue - total_costs
    
    month_data["profit_accrual"] = profit_accrual
    month_data["cash_flow"] = cash_flow
    
    if month == 1:
        month_data["cash_balance"] = cash_flow
    else:
        month_data["cash_balance"] = monthly_data[-1]["cash_balance"] + cash_flow
    
    monthly_data.append(month_data)

# Convert to DataFrame
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

# Main data table
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

# Charts
st.header("📈 Key Metrics Over Time")

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Client growth
ax1.plot(df["month"], df["active_clients"], marker='o', linewidth=2, color='steelblue')
ax1.plot(df["month"], df["total_capacity_clients"], linestyle='--', color='red', label='Capacity')
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

# Therapist scaling
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

st.success("✅ Dashboard V2 Complete with Strategic Analysis")

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
for i in range(1, 9):  # 8 month average lifespan
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
    
    recommendations.append({
        "priority": 1 if net_monthly_savings > 500 else 3,
        "category": "💰 Cost Optimization",
        "title": "Supervision Leverage Strategy",
        "current_state": f"{lmsw_pct:.0f}% LMSW mix ({current_month['active_lmsw']:.0f} LMSW, {current_month['active_lcsw']:.0f} LCSW)",
        "financial_impact": f"Saves ${monthly_lmsw_savings:.0f}/month in therapist costs",
        "cost": f"Costs ${monthly_supervision_cost:.0f}/month in supervision" + (" (you handle 3, external for rest)" if current_month['active_lmsw'] > 3 else " (you handle all)"),
        "net_impact": f"Net savings: ${net_monthly_savings:.0f}/month (${net_monthly_savings*12:.0f}/year)",
        "recommendation": "Continue LMSW strategy - positive ROI" if net_monthly_savings > 0 else "Consider hiring more LCSWs - supervision costs exceed savings",
        "action_items": [
            f"Current model is {'optimal' if net_monthly_savings > 0 else 'suboptimal'}",
            f"You can supervise up to 3 LMSW (currently {min(current_month['active_lmsw'], 3):.0f})",
            "Group supervision = 1 hour covers 3 LMSW" if current_month['active_lmsw'] <= 3 else f"Need external supervisor for {current_month['active_lmsw']-3:.0f} LMSW"
        ]
    })

# --- RECOMMENDATION 2: Marketing Efficiency ---
total_new_clients = df["new_clients"].sum()
total_marketing_spent = df["marketing_spent"].sum()
actual_cac = total_marketing_spent / total_new_clients if total_new_clients > 0 else 0
max_profitable_cac = current_clv * 0.25  # 25% of CLV for 4x return

if actual_cac > max_profitable_cac:
    recommendations.append({
        "priority": 1,
        "category": "⚠️ Marketing Efficiency",
        "title": "Reduce Client Acquisition Cost",
        "current_state": f"${actual_cac:.0f} CAC vs ${max_profitable_cac:.0f} max recommended",
        "financial_impact": f"Overspending ${(actual_cac - max_profitable_cac) * total_new_clients:.0f} over {months_to_simulate} months",
        "recommendation": "Your CAC is too high relative to CLV. Need to improve marketing efficiency or increase CLV.",
        "action_items": [
            f"Target CAC: ${max_profitable_cac:.0f} (25% of CLV for 4x ROI)",
            "Improve website conversion rate",
            "Focus on referral programs (typically $0-50 CAC)",
            "Test lower-cost channels (SEO, community partnerships)",
            f"Alternative: Increase CLV by improving retention"
        ]
    })
elif actual_cac < max_profitable_cac * 0.5:
    recommendations.append({
        "priority": 2,
        "category": "📈 Growth Opportunity",
        "title": "Accelerate Growth - Marketing Underutilized",
        "current_state": f"${actual_cac:.0f} CAC vs ${max_profitable_cac:.0f} max affordable",
        "financial_impact": f"Could spend ${(max_profitable_cac - actual_cac) * total_new_clients:.0f} more profitably",
        "recommendation": "Your marketing is highly efficient. You can afford to spend more to accelerate growth.",
        "action_items": [
            f"You have ${max_profitable_cac - actual_cac:.0f} headroom per client",
            "Consider 2x-ing marketing budget to fill capacity faster",
            "Invest in brand building and SEO",
            "Test premium channels (Psychology Today, Therapy Den)"
        ]
    })

# --- RECOMMENDATION 3: Improve Retention (Reduce Churn) ---
if ongoing_churn > 0.03:  # More than 3%
    improved_churn = ongoing_churn * 0.6  # 40% reduction
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
    
    recommendations.append({
        "priority": 1,
        "category": "💎 Retention Optimization",
        "title": "Reduce Ongoing Churn Rate",
        "current_state": f"{ongoing_churn*100:.0f}% monthly churn → {total_expected_sessions:.1f} lifetime sessions",
        "financial_impact": f"Reducing to {improved_churn*100:.1f}% would increase CLV by ${clv_increase:.0f} ({(clv_increase/current_clv*100):.0f}% improvement)",
        "recommendation": f"Each 1% churn reduction = ${clv_increase/((ongoing_churn-improved_churn)*100):.0f} additional CLV per client",
        "action_items": [
            "Implement regular check-ins at session 3, 6, 10",
            "Survey clients at risk of dropping",
            "Offer package pricing (pre-pay 10 sessions)",
            "Match clients better with therapist specialties",
            f"Annual impact: ${clv_increase * total_new_clients:.0f} on {total_new_clients:.0f} new clients"
        ]
    })

# --- RECOMMENDATION 4: Self-Pay Mix ---
if use_simple_payer and self_pay_pct < 0.15:
    target_self_pay = 0.15
    current_weighted_rate = (1 - self_pay_pct) * avg_insurance_rate + self_pay_pct * self_pay_rate
    target_weighted_rate = (1 - target_self_pay) * avg_insurance_rate + target_self_pay * self_pay_rate
    rate_increase = target_weighted_rate - current_weighted_rate
    
    improved_clv_selfpay = total_expected_sessions * (rate_increase * (1 - no_show_rate) - 0)  # No cost change
    
    recommendations.append({
        "priority": 2,
        "category": "💵 Revenue Optimization",
        "title": "Increase Self-Pay Mix",
        "current_state": f"{self_pay_pct*100:.0f}% self-pay at ${self_pay_rate:.0f} vs {(1-self_pay_pct)*100:.0f}% insurance at ${avg_insurance_rate:.0f}",
        "financial_impact": f"Increasing to {target_self_pay*100:.0f}% self-pay would increase avg rate by ${rate_increase:.0f}/session",
        "recommendation": f"Each client at {target_self_pay*100:.0f}% self-pay worth ${improved_clv_selfpay:.0f} more",
        "action_items": [
            "Market to clients with HSA/FSA accounts",
            "Offer superbills for out-of-network benefits",
            "Specialize in niches (trauma, EMDR, couples) that attract self-pay",
            "Add 'cash pay' option with slight discount vs OON",
            "No insurance delays = better cash flow too!"
        ]
    })

# --- RECOMMENDATION 5: Group Therapy ---
if not offer_group_therapy and current_month["active_lcsw"] >= 2:
    potential_groups = int(current_month["active_lcsw"] * 0.3)  # 30% of LCSW
    group_revenue_potential = potential_groups * clients_per_group * 60 * 4  # Default values
    group_cost_potential = potential_groups * 120 * 4
    group_profit_potential = group_revenue_potential - group_cost_potential
    
    recommendations.append({
        "priority": 1,
        "category": "🚀 High-Leverage Opportunity",
        "title": "Launch Group Therapy",
        "current_state": f"Not offering groups. Have {current_month['active_lcsw']:.0f} licensed therapists available",
        "financial_impact": f"Potential +${group_profit_potential:.0f}/month at 75% margins",
        "recommendation": f"Group therapy is 2-3x more profitable than individual (75% vs 40% margins)",
        "action_items": [
            f"Start with {potential_groups} group(s) (30% of LCSW capacity)",
            "Common groups: DBT skills, anxiety management, grief support",
            "8 clients × $60/session = $480 revenue vs $120 therapist cost",
            "Requires: LCSW + specialized training",
            f"Annual impact: ${group_profit_potential * 12:.0f}"
        ]
    })
elif offer_group_therapy:
    avg_group_profit = df["group_revenue"].mean() - df["group_cost"].mean()
    recommendations.append({
        "priority": 3,
        "category": "✅ Active Strategy",
        "title": "Group Therapy Performance",
        "current_state": f"Currently running groups - ${avg_group_profit:.0f}/month profit",
        "financial_impact": f"Groups contributing ${avg_group_profit/avg_monthly_profit*100:.1f}% of monthly profit",
        "recommendation": "Continue and consider expanding" if avg_group_profit > 1000 else "Monitor closely - may need more groups",
        "action_items": [
            f"Current profit: ${avg_group_profit:.0f}/month",
            "Consider adding more groups if waitlist exists",
            "Train additional LCSWs in group facilitation"
        ]
    })

# --- RECOMMENDATION 6: Billing Model ---
if billing_model == "Billing Service (% of revenue)" and total_annual_revenue > 50000 * 12:
    annual_billing_service_cost = total_annual_revenue * billing_service_pct
    annual_inhouse_cost = 4500 * 12
    annual_savings = annual_billing_service_cost - annual_inhouse_cost
    
    if annual_savings > 10000:
        recommendations.append({
            "priority": 2,
            "category": "💰 Cost Reduction",
            "title": "Switch to In-House Billing",
            "current_state": f"Paying {billing_service_pct*100:.1f}% = ${annual_billing_service_cost:,.0f}/year",
            "financial_impact": f"Save ${annual_savings:,.0f}/year by hiring in-house biller at $4,500/month",
            "recommendation": f"Break-even at ${4500/billing_service_pct:,.0f}/month revenue. You're at ${total_annual_revenue/12:,.0f}/month",
            "action_items": [
                f"Annual savings: ${annual_savings:,.0f}",
                "Hire experienced medical biller (W-2 or contractor)",
                "Invest in billing software/training",
                "ROI payback in < 2 months"
            ]
        })

# --- RECOMMENDATION 7: No-Show Rate ---
if no_show_rate > 0.03:
    annual_sessions_lost = df["expected_sessions"].sum() * no_show_rate
    annual_revenue_lost = annual_sessions_lost * weighted_rate
    
    recommendations.append({
        "priority": 2,
        "category": "📉 Revenue Recovery",
        "title": "Reduce No-Show Rate",
        "current_state": f"{no_show_rate*100:.0f}% no-show rate",
        "financial_impact": f"Losing ${annual_revenue_lost:,.0f}/year in missed sessions",
        "recommendation": f"{annual_sessions_lost:.0f} sessions lost per year - implement prevention strategies",
        "action_items": [
            "Automated appointment reminders (24hr + 2hr before)",
            "Require credit card on file",
            "Late cancellation policy (24hr notice required)",
            "Consider deposits for first session",
            f"Reducing to 3% saves ${(no_show_rate - 0.03) / no_show_rate * annual_revenue_lost:,.0f}/year"
        ]
    })

# --- RECOMMENDATION 8: Cash Flow ---
min_cash_balance = df["cash_balance"].min()
if min_cash_balance < -5000:
    working_capital_needed = abs(min_cash_balance) * 1.2  # 20% buffer
    
    recommendations.append({
        "priority": 1,
        "category": "⚠️ Cash Flow Risk",
        "title": "Address Working Capital Gap",
        "current_state": f"Minimum cash balance: ${min_cash_balance:,.0f} (Month {df[df['cash_balance'] == min_cash_balance]['month'].values[0]})",
        "financial_impact": f"Need ${working_capital_needed:,.0f} in working capital to cover insurance delays",
        "recommendation": f"Insurance delays of {weighted_delay:.1f} months create cash flow gap",
        "action_items": [
            f"Secure ${working_capital_needed:,.0f} line of credit or capital",
            "Consider invoice factoring (advance on receivables)",
            "Increase self-pay mix (immediate payment)",
            "Negotiate faster payment terms with insurers",
            "Build cash reserves during positive months"
        ]
    })

# --- RECOMMENDATION 9: Scale Efficiency ---
profit_margin_current = (avg_monthly_profit / (df["revenue_earned"].mean())) * 100 if df["revenue_earned"].mean() > 0 else 0

if current_month["active_therapists"] < 6 and profit_margin_current < 20:
    target_therapists = 8
    estimated_profit_at_scale = avg_monthly_profit * (target_therapists / max(current_month["active_therapists"], 1)) * 1.3  # 30% margin improvement
    
    recommendations.append({
        "priority": 2,
        "category": "📈 Scale to Improve Margins",
        "title": "Grow to Achieve Better Margins",
        "current_state": f"{current_month['active_therapists']:.0f} therapists, {profit_margin_current:.1f}% margin",
        "financial_impact": f"Scaling to {target_therapists} therapists could achieve ${estimated_profit_at_scale:,.0f}/month profit",
        "recommendation": "Overhead costs spread across more therapists = better margins",
        "action_items": [
            f"Target: {target_therapists} therapists for optimal efficiency",
            "Fixed costs (tech, admin, office) don't scale linearly",
            "Break-even at ~4 therapists",
            "Optimal margin at 8-12 therapists",
            "Plan hiring timeline based on capacity needs"
        ]
    })

# Sort recommendations by priority
recommendations.sort(key=lambda x: x["priority"])

# Display recommendations
for i, rec in enumerate(recommendations):
    with st.expander(f"{'🔴' if rec['priority']==1 else '🟡' if rec['priority']==2 else '🟢'} {rec['title']}", expanded=(rec['priority']==1)):
        st.markdown(f"**Category:** {rec['category']}")
        st.markdown(f"**Current State:** {rec['current_state']}")
        st.markdown(f"**Financial Impact:** {rec['financial_impact']}")
        if 'cost' in rec:
            st.markdown(f"**Cost:** {rec['cost']}")
        if 'net_impact' in rec:
            st.markdown(f"**Net Impact:** {rec['net_impact']}")
        st.markdown(f"**Recommendation:** {rec['recommendation']}")
        
        if 'action_items' in rec:
            st.markdown("**Action Items:**")
            for item in rec['action_items']:
                st.markdown(f"- {item}")

# ==========================================
# SCENARIO ANALYSIS
# ==========================================

st.header("📊 Scenario Analysis")

st.markdown("""
Compare different strategic scenarios to see how key changes affect your practice economics.
""")

# Define scenarios
scenarios_to_run = {
    "Current (Base Case)": {
        "ongoing_churn": ongoing_churn,
        "self_pay_pct": self_pay_pct if use_simple_payer else 0.10,
        "no_show_rate": no_show_rate,
        "lmsw_pct": current_month["active_lmsw"] / max(current_month["active_therapists"], 1) if current_month["active_therapists"] > 0 else 0.7
    },
    "Conservative": {
        "ongoing_churn": min(ongoing_churn + 0.03, 0.20),
        "self_pay_pct": max((self_pay_pct if use_simple_payer else 0.10) - 0.10, 0),
        "no_show_rate": min(no_show_rate * 1.3, 0.20),
        "lmsw_pct": current_month["active_lmsw"] / max(current_month["active_therapists"], 1) if current_month["active_therapists"] > 0 else 0.7
    },
    "Optimistic": {
        "ongoing_churn": max(ongoing_churn - 0.02, 0.02),
        "self_pay_pct": min((self_pay_pct if use_simple_payer else 0.10) + 0.15, 0.50),
        "no_show_rate": max(no_show_rate * 0.7, 0.02),
        "lmsw_pct": current_month["active_lmsw"] / max(current_month["active_therapists"], 1) if current_month["active_therapists"] > 0 else 0.7
    },
    "With Group Therapy": {
        "ongoing_churn": ongoing_churn,
        "self_pay_pct": self_pay_pct if use_simple_payer else 0.10,
        "no_show_rate": no_show_rate,
        "lmsw_pct": current_month["active_lmsw"] / max(current_month["active_therapists"], 1) if current_month["active_therapists"] > 0 else 0.7,
        "group_therapy_profit": 1500  # Assumed monthly group profit
    }
}

scenario_results = []

for scenario_name, params in scenarios_to_run.items():
    # Recalculate CLV with scenario parameters
    scenario_survival = 1.0
    scenario_sessions = 0
    for i in range(1, 12):
        if i == 1:
            scenario_survival *= (1 - month1_churn)
        elif i == 2:
            scenario_survival *= (1 - month2_churn)
        elif i == 3:
            scenario_survival *= (1 - month3_churn)
        else:
            scenario_survival *= (1 - params["ongoing_churn"])
        scenario_sessions += scenario_survival * avg_sessions_per_client_per_month
    
    # Calculate weighted rate for scenario
    scenario_insurance_pct = 1 - params["self_pay_pct"]
    scenario_weighted_rate = scenario_insurance_pct * avg_insurance_rate + params["self_pay_pct"] * self_pay_rate
    
    # Calculate contribution margin
    scenario_revenue_per_session = scenario_weighted_rate * (1 - params["no_show_rate"])
    scenario_cost_per_session = params["lmsw_pct"] * lmsw_pay_per_session + (1 - params["lmsw_pct"]) * lcsw_pay_per_session
    scenario_margin_per_session = scenario_revenue_per_session - scenario_cost_per_session
    
    scenario_clv = scenario_sessions * scenario_margin_per_session
    scenario_max_cac = scenario_clv * 0.25
    
    # Estimate monthly profit (simplified)
    scenario_monthly_profit = avg_monthly_profit
    if scenario_name == "Conservative":
        scenario_monthly_profit *= 0.7
    elif scenario_name == "Optimistic":
        scenario_monthly_profit *= 1.4
    elif scenario_name == "With Group Therapy":
        scenario_monthly_profit += params.get("group_therapy_profit", 0)
    
    scenario_margin = (scenario_monthly_profit / (df["revenue_earned"].mean() + params.get("group_therapy_profit", 0))) * 100 if df["revenue_earned"].mean() > 0 else 0
    
    scenario_results.append({
        "Scenario": scenario_name,
        "CLV": scenario_clv,
        "Max CAC": scenario_max_cac,
        "Monthly Profit": scenario_monthly_profit,
        "Margin %": scenario_margin
    })

scenario_df = pd.DataFrame(scenario_results)

st.dataframe(scenario_df.style.format({
    "CLV": "${:,.0f}",
    "Max CAC": "${:,.0f}",
    "Monthly Profit": "${:,.0f}",
    "Margin %": "{:.1f}%"
}), use_container_width=True)

# Scenario assumptions
st.markdown("**Scenario Assumptions:**")
st.markdown(f"""
- **Conservative:** Ongoing churn +3% → {scenarios_to_run['Conservative']['ongoing_churn']*100:.0f}%, Self-pay -10% → {scenarios_to_run['Conservative']['self_pay_pct']*100:.0f}%, No-shows +30% → {scenarios_to_run['Conservative']['no_show_rate']*100:.0f}%
- **Optimistic:** Ongoing churn -2% → {scenarios_to_run['Optimistic']['ongoing_churn']*100:.0f}%, Self-pay +15% → {scenarios_to_run['Optimistic']['self_pay_pct']*100:.0f}%, No-shows -30% → {scenarios_to_run['Optimistic']['no_show_rate']*100:.0f}%
- **With Group Therapy:** Assumes $1,500/month additional profit from groups (30% of LCSW running groups)
""")

# Visualization
fig_scenarios, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

scenario_df.plot(x="Scenario", y=["CLV", "Max CAC"], kind="bar", ax=ax_left, color=['steelblue', 'orange'])
ax_left.set_title("Client Lifetime Value & Max CAC by Scenario")
ax_left.set_ylabel("Dollars ($)")
ax_left.set_xlabel("")
ax_left.legend(["CLV", "Max CAC"])
ax_left.grid(True, alpha=0.3, axis='y')

scenario_df.plot(x="Scenario", y="Monthly Profit", kind="bar", ax=ax_right, color='green')
ax_right.set_title("Monthly Profit by Scenario")
ax_right.set_ylabel("Profit ($)")
ax_right.set_xlabel("")
ax_right.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
st.pyplot(fig_scenarios)

st.success("✅ V2 Dashboard Complete with Strategic Recommendations & Scenario Analysis!")
