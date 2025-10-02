import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Therapy Practice Dashboard", layout="wide")
st.title("Therapy Practice Full Financial & Therapist-Level Dashboard")

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("Practice Setup")
num_therapists = st.sidebar.number_input("Number of Employed Therapists", min_value=0, value=3)
owner_sessions_per_week = st.sidebar.number_input("Owner Sessions per Week", min_value=0, value=20)
sessions_per_week_per_therapist = st.sidebar.number_input("Avg Weekly Sessions per Therapist", min_value=1, value=20)
avg_fee = st.sidebar.number_input("Avg Insurance Payout per Session ($)", min_value=50.0, value=80.0, step=10.0)
therapist_pay_per_session = st.sidebar.number_input("Therapist Pay per Session ($)", min_value=0.0, value=45.0, step=5.0)
no_show_rate = st.sidebar.slider("No-Show Rate (%)", 0, 50, 12) / 100
therapist_utilization_rate = st.sidebar.slider("Therapist Utilization Rate (%)", 0, 100, 85) / 100
owner_utilization_rate = st.sidebar.slider("Owner Utilization Rate (%)", 0, 100, 85) / 100

st.sidebar.header("Admin & Overhead")
num_admins = st.sidebar.number_input("Number of Admin Staff", min_value=0, value=1)
admin_hours_per_week = st.sidebar.number_input("Admin Hours per Week (per admin)", min_value=0, value=20)
admin_hourly_rate = st.sidebar.number_input("Admin Hourly Rate ($)", min_value=0.0, value=25.0, step=1.0)
tech_cost_per_therapist = st.sidebar.number_input("Tech Stack Cost per Therapist ($/month)", min_value=0.0, value=150.0, step=25.0)
other_overhead_monthly = st.sidebar.number_input("Other Fixed Overhead ($/month)", min_value=0.0, value=1500.0, step=50.0)

st.sidebar.header("Client & Churn")
avg_sessions_per_client_per_month = st.sidebar.number_input("Avg Sessions per Client per Month", min_value=1.0, value=2.5, step=0.1)
month1_churn = st.sidebar.slider("Month 1 Churn (%)", 0, 100, 25) / 100
month2_churn = st.sidebar.slider("Month 2 Churn (%)", 0, 100, 15) / 100
month3_churn = st.sidebar.slider("Month 3 Churn (%)", 0, 100, 10) / 100
ongoing_churn = st.sidebar.slider("Ongoing Churn (%)", 0, 20, 5) / 100
average_client_lifespan = st.sidebar.number_input("Average Client Lifespan (months)", min_value=1, value=8)

st.sidebar.header("Marketing & Acquisition")
monthly_marketing_budget = st.sidebar.number_input("Monthly Marketing Budget ($)", min_value=0.0, value=2000.0, step=100.0)
client_acquisition_cost = st.sidebar.number_input("Client Acquisition Cost ($/new client)", min_value=0.0, value=150.0, step=10.0)
insurance_payment_delay = st.sidebar.number_input("Insurance Payment Delay (months)", min_value=0.0, value=1.5, step=0.1)
seasonal_variation = st.sidebar.slider("Seasonal Variation (%) - Summer Slowdown", 0, 50, 20) / 100

st.sidebar.header("Simulation & Scenario")
months_to_simulate = st.sidebar.number_input("Months to Simulate", min_value=1, value=12)
num_simulations = st.sidebar.slider("Monte Carlo Runs", 100, 5000, 1000, 100)
scenario = st.sidebar.selectbox("Scenario", ["Base", "Conservative", "Aggressive"])

# -------------------------------
# SCENARIO MULTIPLIER
# -------------------------------
scenario_multipliers = {"Base": 1.0, "Conservative": 0.9, "Aggressive": 1.1}
scenario_mult = scenario_multipliers[scenario]
avg_fee_adjusted = avg_fee * scenario_mult
avg_sessions_adjusted = avg_sessions_per_client_per_month * scenario_mult

# -------------------------------
# CORE CALCULATIONS
# -------------------------------
weeks_per_month = 52 / 12

# Owner and therapist sessions
actual_owner_sessions_per_week = owner_sessions_per_week * owner_utilization_rate
actual_therapist_sessions_per_week = sessions_per_week_per_therapist * therapist_utilization_rate

owner_sessions_per_month = actual_owner_sessions_per_week * weeks_per_month
therapist_sessions_per_month = actual_therapist_sessions_per_week * weeks_per_month
total_sessions_per_month = owner_sessions_per_month + (num_therapists * therapist_sessions_per_month)

effective_fee_per_session = avg_fee_adjusted * (1 - no_show_rate)

# Tech costs - include owner
total_tech_cost = (num_therapists + 1) * tech_cost_per_therapist

# Admin costs
admin_cost_per_month = num_admins * admin_hours_per_week * weeks_per_month * admin_hourly_rate

# Overhead per session for CLV calculation
overhead_per_session = (total_tech_cost + admin_cost_per_month + other_overhead_monthly) / total_sessions_per_month if total_sessions_per_month > 0 else 0

contribution_margin_per_session = effective_fee_per_session - therapist_pay_per_session - overhead_per_session

# Calculate max clients capacity
max_capacity_in_clients = total_sessions_per_month / avg_sessions_adjusted if avg_sessions_adjusted > 0 else 0

# -------------------------------
# MONTH-BY-MONTH SIMULATION
# -------------------------------
active_clients = [0]
new_clients_per_month = []
churned_clients_per_month = []
monthly_revenue = []
monthly_cost = []
monthly_profit = []
cash_balance = [0]
revenue_history = []  # For delayed payments

for month in range(1, months_to_simulate + 1):
    # Determine churn rate for this month
    if month == 1:
        churn_rate = month1_churn
    elif month == 2:
        churn_rate = month2_churn
    elif month == 3:
        churn_rate = month3_churn
    else:
        churn_rate = ongoing_churn

    # Calculate surviving and churned clients
    surviving_clients = active_clients[-1] * (1 - churn_rate)
    churned_clients = active_clients[-1] - surviving_clients
    churned_clients_per_month.append(churned_clients)

    # Calculate new client acquisition
    max_new_clients_marketing = monthly_marketing_budget / client_acquisition_cost if client_acquisition_cost > 0 else 0
    required_new_clients = churned_clients  # Replace churned
    capacity_available = max_capacity_in_clients - surviving_clients
    
    # Take minimum of marketing budget, capacity, and growth targets
    new_clients = min(max_new_clients_marketing, max(required_new_clients, capacity_available))
    new_clients = max(0, new_clients)  # Can't be negative
    new_clients_per_month.append(new_clients)

    # Total active clients
    total_clients = surviving_clients + new_clients
    active_clients.append(total_clients)

    # Revenue calculation
    expected_sessions = total_clients * avg_sessions_adjusted
    revenue = expected_sessions * effective_fee_per_session
    revenue_history.append(revenue)
    
    # Cost calculation
    owner_therapist_pay = owner_sessions_per_month * therapist_pay_per_session
    therapist_pay = num_therapists * therapist_sessions_per_month * therapist_pay_per_session
    acquisition_cost = new_clients * client_acquisition_cost
    
    cost = owner_therapist_pay + therapist_pay + admin_cost_per_month + total_tech_cost + other_overhead_monthly + acquisition_cost
    
    # Profit
    profit = revenue - cost

    # Cash flow with insurance delay
    delay_months = int(insurance_payment_delay)
    delay_index = month - 1 - delay_months
    
    if delay_index >= 0 and delay_index < len(revenue_history):
        delayed_revenue = revenue_history[delay_index]
    else:
        delayed_revenue = 0
    
    cash_balance.append(cash_balance[-1] + delayed_revenue - cost)

    monthly_revenue.append(revenue)
    monthly_cost.append(cost)
    monthly_profit.append(profit)

active_clients = active_clients[1:]

# -------------------------------
# THERAPIST-LEVEL P&L
# -------------------------------
if num_therapists > 0:
    therapist_clients = np.array(active_clients) * (num_therapists * therapist_sessions_per_month / total_sessions_per_month) / num_therapists if total_sessions_per_month > 0 else np.zeros(len(active_clients))
    sessions_per_therapist = therapist_clients * avg_sessions_adjusted
    revenue_per_therapist = sessions_per_therapist * effective_fee_per_session
    therapist_session_cost = sessions_per_therapist * therapist_pay_per_session
    therapist_overhead = tech_cost_per_therapist + (admin_cost_per_month / num_therapists) + (other_overhead_monthly / num_therapists)
    cost_per_therapist = therapist_session_cost + therapist_overhead
    profit_per_therapist = revenue_per_therapist - cost_per_therapist

    therapist_pnl_df = pd.DataFrame({
        "Month": range(1, months_to_simulate + 1),
        "Clients per Therapist": np.round(therapist_clients, 2),
        "Sessions per Therapist": np.round(sessions_per_therapist, 2),
        "Revenue ($)": np.round(revenue_per_therapist, 2),
        "Cost ($)": np.round(cost_per_therapist, 2),
        "Profit ($)": np.round(profit_per_therapist, 2)
    })
else:
    therapist_pnl_df = pd.DataFrame({"Note": ["No employed therapists configured"]})

# -------------------------------
# CLIENT LIFETIME VALUE
# -------------------------------
# Build survival curve
survival_curve = []
survival_rate = 1.0

survival_rate *= (1 - month1_churn)
survival_curve.append(survival_rate)

survival_rate *= (1 - month2_churn)
survival_curve.append(survival_rate)

survival_rate *= (1 - month3_churn)
survival_curve.append(survival_rate)

for m in range(4, int(average_client_lifespan) + 1):
    survival_rate *= (1 - ongoing_churn)
    survival_curve.append(survival_rate)

# Calculate CLV
total_expected_sessions = 0
for month_idx in range(len(survival_curve)):
    survival_this_month = 1.0 if month_idx == 0 else survival_curve[month_idx - 1]
    total_expected_sessions += survival_this_month * avg_sessions_adjusted

# Present value discounting
monthly_discount_rate = 0.005
discounted_clv = 0
for month_idx in range(len(survival_curve)):
    survival_this_month = 1.0 if month_idx == 0 else survival_curve[month_idx - 1]
    monthly_contribution = survival_this_month * avg_sessions_adjusted * contribution_margin_per_session
    payment_month = month_idx + insurance_payment_delay
    present_value = monthly_contribution / ((1 + monthly_discount_rate) ** payment_month)
    discounted_clv += present_value

# Adjust for seasonality (conservative 30% of stated variation)
seasonal_adjustment = 1 - (seasonal_variation * 0.3)
adjusted_clv = discounted_clv * seasonal_adjustment

# -------------------------------
# DASHBOARD & CHARTS
# -------------------------------
summary_df = pd.DataFrame({
    "Month": range(1, months_to_simulate + 1),
    "Active Clients": np.round(active_clients, 2),
    "New Clients": np.round(new_clients_per_month, 2),
    "Churned Clients": np.round(churned_clients_per_month, 2),
    "Revenue ($)": np.round(monthly_revenue, 2),
    "Cost ($)": np.round(monthly_cost, 2),
    "Profit ($)": np.round(monthly_profit, 2),
    "Cash Balance ($)": np.round(cash_balance[1:], 2)
})

st.header("Monthly Client & Profit Summary")
st.dataframe(summary_df, use_container_width=True)

st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Risk-Adjusted CLV ($)", f"${adjusted_clv:,.0f}")
col2.metric("Total Profit ($)", f"${sum(monthly_profit):,.0f}")
col3.metric("Final Cash Balance ($)", f"${cash_balance[-1]:,.0f}")
col4.metric("Avg Profit/Month ($)", f"${np.mean(monthly_profit):,.0f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Total Sessions/Month", f"{int(total_sessions_per_month)}")
col6.metric("Contribution Margin/Session", f"${contribution_margin_per_session:.2f}")
col7.metric("Capacity Utilization", f"{(np.mean(active_clients) / max_capacity_in_clients * 100):.1f}%" if max_capacity_in_clients > 0 else "N/A")
col8.metric("Break-even Clients", f"{int((admin_cost_per_month + total_tech_cost + other_overhead_monthly) / (contribution_margin_per_session * avg_sessions_adjusted))}" if contribution_margin_per_session > 0 else "N/A")

if num_therapists > 0:
    st.header("Therapist-Level Monthly P&L")
    st.dataframe(therapist_pnl_df, use_container_width=True)

st.header("Active Clients Over Time")
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(range(1, months_to_simulate + 1), active_clients, marker='o', color='steelblue', linewidth=2)
ax1.axhline(y=max_capacity_in_clients, color='red', linestyle='--', label='Max Capacity')
ax1.set_xlabel("Month")
ax1.set_ylabel("Active Clients")
ax1.set_title("Client Growth Over Time")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

st.header("Cash Balance Over Time")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(range(1, months_to_simulate + 1), cash_balance[1:], marker='o', color='green', linewidth=2)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel("Month")
ax2.set_ylabel("Cash Balance ($)")
ax2.set_title("Cash Flow (with Insurance Payment Delay)")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

st.header("Revenue vs Cost vs Profit")
fig3, ax3 = plt.subplots(figsize=(10, 5))
months = range(1, months_to_simulate + 1)
ax3.plot(months, monthly_revenue, marker='o', label='Revenue', color='green', linewidth=2)
ax3.plot(months, monthly_cost, marker='s', label='Cost', color='red', linewidth=2)
ax3.plot(months, monthly_profit, marker='^', label='Profit', color='blue', linewidth=2)
ax3.set_xlabel("Month")
ax3.set_ylabel("Dollars ($)")
ax3.set_title("Monthly Revenue, Cost, and Profit")
ax3.legend()
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

st.header("Profit Distribution (Monte Carlo Simulation)")
simulated_profits = []
for _ in range(num_simulations):
    variation = np.random.normal(1.0, 0.05, months_to_simulate)
    simulated_profit = np.sum(np.array(monthly_profit) * variation)
    simulated_profits.append(simulated_profit)

fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.hist(simulated_profits, bins=50, color='skyblue', edgecolor='black')
ax4.axvline(x=np.mean(simulated_profits), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(simulated_profits):,.0f}')
ax4.set_title("Profit Distribution Across Monte Carlo Runs")
ax4.set_xlabel("Total Profit ($)")
ax4.set_ylabel("Frequency")
ax4.legend()
st.pyplot(fig4)

if num_therapists > 0 and len(profit_per_therapist) > 0:
    st.header("Therapist Profit Over Time")
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.plot(range(1, months_to_simulate + 1), profit_per_therapist, marker='o', color='purple', linewidth=2)
    ax5.set_xlabel("Month")
    ax5.set_ylabel("Profit per Therapist ($)")
    ax5.set_title("Average Profit per Therapist by Month")
    ax5.grid(True, alpha=0.3)
    st.pyplot(fig5)

st.header("Client Acquisition Metrics")
fig6, ax6 = plt.subplots(figsize=(10, 5))
months = range(1, months_to_simulate + 1)
ax6.bar(months, new_clients_per_month, label='New Clients', color='green', alpha=0.7)
ax6.bar(months, churned_clients_per_month, label='Churned Clients', color='red', alpha=0.7)
ax6.set_xlabel("Month")
ax6.set_ylabel("Number of Clients")
ax6.set_title("New vs Churned Clients by Month")
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')
st.pyplot(fig6)

# Summary statistics
st.header("Summary Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Revenue Metrics")
    st.write(f"**Total Revenue:** ${sum(monthly_revenue):,.0f}")
    st.write(f"**Avg Monthly Revenue:** ${np.mean(monthly_revenue):,.0f}")
    st.write(f"**Revenue per Session:** ${effective_fee_per_session:.2f}")
    st.write(f"**Total Sessions:** {int(sum([c * avg_sessions_adjusted for c in active_clients]))}")

with col2:
    st.subheader("Cost Metrics")
    st.write(f"**Total Costs:** ${sum(monthly_cost):,.0f}")
    st.write(f"**Avg Monthly Costs:** ${np.mean(monthly_cost):,.0f}")
    st.write(f"**Tech Stack (monthly):** ${total_tech_cost:,.0f}")
    st.write(f"**Admin Costs (monthly):** ${admin_cost_per_month:,.0f}")

with col3:
    st.subheader("Client Metrics")
    st.write(f"**Total New Clients:** {sum(new_clients_per_month):.1f}")
    st.write(f"**Total Churned:** {sum(churned_clients_per_month):.1f}")
    st.write(f"**Final Active Clients:** {active_clients[-1]:.1f}")
    st.write(f"**Max Capacity:** {max_capacity_in_clients:.1f} clients")
