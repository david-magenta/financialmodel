import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Therapy Practice Financial Model", layout="wide")
st.title("Therapy Practice Full Financial Dashboard")

# -------------------------------
# USER INPUTS
# -------------------------------
st.sidebar.header("Practice Setup")
num_therapists = st.sidebar.number_input("Number of Therapists", min_value=1, value=3)
sessions_per_week_per_therapist = st.sidebar.number_input("Average Weekly Sessions per Therapist", min_value=1, value=20)
avg_fee = st.sidebar.number_input("Average Collected Fee per Session ($)", min_value=50.0, value=150.0, step=10.0)
therapist_pay_per_session = st.sidebar.number_input("Therapist Pay per Session ($)", min_value=0.0, value=50.0, step=5.0)
no_show_rate = st.sidebar.slider("No-Show Rate (%)", 0, 50, 12) / 100
utilization_rate = st.sidebar.slider("Utilization Rate (%)", 0, 100, 85) / 100

st.sidebar.header("Overhead")
tech_cost_per_therapist = st.sidebar.number_input("Tech Stack Cost per Therapist ($/month)", min_value=0.0, value=150.0, step=25.0)
admin_overhead_per_therapist = st.sidebar.number_input("Admin Overhead per Therapist ($/month)", min_value=0.0, value=200.0, step=25.0)
other_overhead_per_therapist = st.sidebar.number_input("Other Overhead per Therapist ($/month)", min_value=0.0, value=125.0, step=25.0)
monthly_fixed_overhead_cash = st.sidebar.number_input("Other Fixed Overhead ($/month)", min_value=0.0, value=500.0, step=50.0)

st.sidebar.header("Client & Churn")
avg_sessions_per_client_per_month = st.sidebar.number_input("Average Sessions per Client per Month", min_value=1.0, value=2.5, step=0.1)
month1_churn = st.sidebar.slider("Month 1 Churn (%)", 0, 100, 25) / 100
month2_churn = st.sidebar.slider("Month 2 Churn (%)", 0, 100, 15) / 100
month3_churn = st.sidebar.slider("Month 3 Churn (%)", 0, 100, 10) / 100
ongoing_churn = st.sidebar.slider("Ongoing Churn (%)", 0, 20, 5) / 100
average_client_lifespan = st.sidebar.number_input("Average Client Lifespan (months)", min_value=1, value=8)

st.sidebar.header("Marketing & Acquisition")
monthly_marketing_budget = st.sidebar.number_input("Monthly Marketing Budget ($)", min_value=0.0, value=1000.0, step=50.0)
client_acquisition_cost = st.sidebar.number_input("Client Acquisition Cost ($/new client)", min_value=0.0, value=200.0, step=10.0)
target_profit_margin = st.sidebar.slider("Target Profit Margin (%)", 0, 100, 25) / 100
max_payback_months = st.sidebar.number_input("Max Payback Period (months)", min_value=1, value=4)
required_roi = st.sidebar.number_input("Required ROI (%)", min_value=0.0, value=200.0) / 100
insurance_payment_delay = st.sidebar.number_input("Insurance Payment Delay (months)", min_value=0.0, value=1.5, step=0.1)
seasonal_variation = st.sidebar.slider("Seasonal Variation (%)", 0, 50, 20) / 100
market_competition_factor = st.sidebar.number_input("Market Competition Factor", min_value=0.1, value=1.0, step=0.1)

# -------------------------------
# CORE CALCULATIONS
# -------------------------------
sessions_per_month_per_therapist = sessions_per_week_per_therapist * 4 * utilization_rate
total_sessions_per_month = sessions_per_month_per_therapist * num_therapists
effective_fee_per_session = avg_fee * (1 - no_show_rate)

# Overhead per session
overhead_per_therapist = tech_cost_per_therapist + admin_overhead_per_therapist + other_overhead_per_therapist
overhead_per_session = overhead_per_therapist / (avg_sessions_per_client_per_month * total_sessions_per_month / num_therapists)

contribution_margin_per_session = effective_fee_per_session - (therapist_pay_per_session + overhead_per_session)

# -------------------------------
# CLIENT SURVIVAL & CLV
# -------------------------------
clients_start = 100  # normalize to 100 clients for survival curve
survival = [clients_start]
for month in range(1, int(average_client_lifespan)+1):
    if month == 1:
        remaining = survival[-1] * (1 - month1_churn)
    elif month == 2:
        remaining = survival[-1] * (1 - month2_churn)
    elif month == 3:
        remaining = survival[-1] * (1 - month3_churn)
    else:
        remaining = survival[-1] * (1 - ongoing_churn)
    survival.append(remaining)

# Expected sessions per month
expected_sessions_per_month = [s * avg_sessions_per_client_per_month for s in survival[:-1]]  # exclude last month
gross_clv = sum([s * contribution_margin_per_session for s in expected_sessions_per_month])

# Present Value Adjustment for insurance delay
pv_discount_rate = 0.005  # 0.5% per month
discounted_clv = sum([s * contribution_margin_per_session / ((1 + pv_discount_rate) ** (i+1)) for i,s in enumerate(expected_sessions_per_month)])

# Risk adjustments
adjusted_clv = discounted_clv * (1 - seasonal_variation) / market_competition_factor

# Max acquisition cost constraints
profit_margin_limit = adjusted_clv * (1 - target_profit_margin)
payback_limit = min(adjusted_clv, contribution_margin_per_session * avg_sessions_per_client_per_month * max_payback_months)
roi_limit = adjusted_clv / (1 + required_roi)
max_acquisition_cost = min(profit_margin_limit, payback_limit, roi_limit)

# -------------------------------
# MARKETING & CAC LOGIC
# -------------------------------
active_clients = survival[0]
churned_clients = clients_start - survival[1]
required_new_clients = churned_clients
required_marketing = required_new_clients * client_acquisition_cost
budget_ok = required_marketing <= monthly_marketing_budget
achievable_new_clients = monthly_marketing_budget / client_acquisition_cost if client_acquisition_cost > 0 else 0

# -------------------------------
# FINANCIAL DASHBOARD
# -------------------------------
st.header("Financial Overview")
results = {
    "Effective Fee per Session ($)": round(effective_fee_per_session,2),
    "Contribution Margin per Session ($)": round(contribution_margin_per_session,2),
    "Gross CLV per Client ($)": round(gross_clv,2),
    "Discounted CLV per Client ($)": round(discounted_clv,2),
    "Adjusted CLV ($)": round(adjusted_clv,2),
    "Max Acquisition Cost ($)": round(max_acquisition_cost,2),
    "Required New Clients": round(required_new_clients,2),
    "Marketing Budget Suffices?": "Yes" if budget_ok else "No",
    "Achievable New Clients with Budget": round(achievable_new_clients,2)
}
st.table(pd.DataFrame(results.items(), columns=["Metric", "Value"]))

# -------------------------------
# SCENARIO ANALYSIS
# -------------------------------
st.header("Scenario Analysis")
scenarios = {"Conservative": 0.9, "Base": 1.0, "Aggressive": 1.1}
scenario_data = []
for name,factor in scenarios.items():
    rev = total_sessions_per_month * avg_fee * factor
    profit = rev - (num_therapists * (therapist_pay_per_session * sessions_per_month_per_therapist + overhead_per_therapist) + monthly_fixed_overhead_cash + monthly_marketing_budget)
    scenario_data.append([name, round(rev,2), round(profit,2)])
st.table(pd.DataFrame(scenario_data, columns=["Scenario", "Revenue ($)", "Profit ($)"]))

# -------------------------------
# MONTE CARLO SIMULATION
# -------------------------------
st.header("Monte Carlo Simulation")
num_simulations = st.slider("Number of Monte Carlo Runs", 100, 5000, 1000, 100)
run_mc = st.checkbox("Run Monte Carlo Simulation")

if run_mc:
    revenue_sim = np.random.normal(total_sessions_per_month * avg_fee, total_sessions_per_month * avg_fee * 0.1, num_simulations)
    costs_sim = np.random.normal(num_therapists * (therapist_pay_per_session * sessions_per_month_per_therapist + overhead_per_therapist) + monthly_fixed_overhead_cash + monthly_marketing_budget, 0.05*(num_therapists * (therapist_pay_per_session * sessions_per_month_per_therapist + overhead_per_therapist) + monthly_fixed_overhead_cash + monthly_marketing_budget), num_simulations)
    profit_sim = revenue_sim - costs_sim

    mc_results = pd.DataFrame({
        "Mean Profit ($)": [round(profit_sim.mean(),2)],
        "5th Percentile ($)": [round(np.percentile(profit_sim,5),2)],
        "95th Percentile ($)": [round(np.percentile(profit_sim,95),2)]
    })
    st.table(mc_results)

    # Profit distribution chart
    fig, ax = plt.subplots(figsize=(10,4))
    ax.hist(profit_sim, bins=50, color='skyblue', edgecolor='black')
    ax.set_title("Monte Carlo Profit Distribution")
    ax.set_xlabel("Monthly Profit ($)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# -------------------------------
# CLIENT SURVIVAL CHART
# -------------------------------
st.header("Client Survival Over Time")
survival_df = pd.DataFrame({"Month": list(range(1,len(survival))), "Remaining Clients": survival[1:]})
st.line_chart(survival_df.set_index("Month"))

# -------------------------------
# MARKETING BUDGET CHART
# -------------------------------
st.header("Marketing Budget vs Required New Clients")
fig2, ax2 = plt.subplots()
ax2.bar(["Required", "Achievable"], [required_new_clients, achievable_new_clients], color=["orange","green"])
ax2.set_ylabel("Number of Clients")
ax2.set_title("New Clients: Required vs Achievable with Marketing Budget")
st.pyplot(fig2)
