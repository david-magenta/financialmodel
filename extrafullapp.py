import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Therapy Practice Financial Model", layout="wide")
st.title("Therapy Practice Financial Dashboard")

# -------------------------------
# USER INPUTS
# -------------------------------
st.sidebar.header("Practice Setup")
num_therapists = st.sidebar.number_input("Number of Therapists", min_value=1, value=3)
sessions_per_week = st.sidebar.number_input("Average Weekly Sessions per Therapist", min_value=1, value=25)
avg_fee = st.sidebar.number_input("Average Collected Fee per Session ($)", min_value=50.0, value=150.0, step=10.0)
utilization_rate = st.sidebar.slider("Utilization Rate (%)", 0, 100, 85) / 100

st.sidebar.header("Compensation & Costs")
therapist_salary = st.sidebar.number_input("Average Annual Therapist Salary ($)", min_value=0, value=70000, step=5000)
therapist_benefits = st.sidebar.number_input("Benefits per Therapist ($)", min_value=0, value=10000, step=1000)
fixed_overhead_cash = st.sidebar.number_input("Fixed Monthly Overhead ($)", min_value=0.0, value=5000.0, step=500.0)
marketing_budget = st.sidebar.number_input("Monthly Marketing Budget ($)", min_value=0.0, value=1000.0, step=100.0)
client_acquisition_cost = st.sidebar.number_input("Client Acquisition Cost ($ per new client)", min_value=0.0, value=200.0, step=10.0)

st.sidebar.header("Strategic Levers")
retention_rate = st.sidebar.slider("Client Retention Rate (%)", 0, 100, 75) / 100
conversion_rate = st.sidebar.slider("Lead-to-Client Conversion Rate (%)", 0, 100, 25) / 100
annual_growth_rate = st.sidebar.slider("Annual Growth Rate (%)", 0, 100, 10) / 100
churn_rate = 1 - retention_rate

# -------------------------------
# CORE CALCULATIONS
# -------------------------------

# Sessions and revenue
sessions_per_month_per_therapist = sessions_per_week * 4 * utilization_rate
total_sessions_per_month = sessions_per_month_per_therapist * num_therapists
monthly_revenue = total_sessions_per_month * avg_fee

# Therapist costs
monthly_therapist_salary = (therapist_salary + therapist_benefits) / 12
total_therapist_costs = monthly_therapist_salary * num_therapists

# Fixed overhead
monthly_overhead = fixed_overhead_cash + marketing_budget

# Monthly profit
monthly_profit = monthly_revenue - (total_therapist_costs + monthly_overhead)
profit_per_therapist = monthly_profit / num_therapists if num_therapists > 0 else 0

# Churn & new client requirements
# Estimate active clients
avg_clients_per_therapist = total_sessions_per_month / (4 * 2.5)  # assuming avg 2.5 sessions per client per month
active_clients = avg_clients_per_therapist * num_therapists
churned_clients = active_clients * churn_rate
required_new_clients = churned_clients
required_marketing = required_new_clients * client_acquisition_cost
budget_ok = required_marketing <= marketing_budget
achievable_new_clients = marketing_budget / client_acquisition_cost if client_acquisition_cost > 0 else 0

# CAC payback in months
cac_payback_months = client_acquisition_cost / (avg_fee * sessions_per_month_per_therapist / 4)  # approximate per-client contribution

# -------------------------------
# RESULTS TABLES
# -------------------------------
st.header("Financial Overview")
results = {
    "Monthly Revenue ($)": round(monthly_revenue or 0, 2),
    "Therapist Costs ($)": round(total_therapist_costs or 0, 2),
    "Fixed Overhead + Marketing ($)": round(monthly_overhead or 0, 2),
    "Monthly Profit ($)": round(monthly_profit or 0, 2),
    "Profit per Therapist ($)": round(profit_per_therapist or 0, 2),
    "Active Clients": round(active_clients or 0, 2),
    "Churned Clients": round(churned_clients or 0, 2),
    "Required New Clients": round(required_new_clients or 0, 2),
    "Achievable New Clients (Budget Limited)": round(achievable_new_clients or 0, 2),
    "Marketing Budget Suffices?": "Yes" if budget_ok else "No",
    "CAC Payback (Months)": round(cac_payback_months or 0, 2)
}
st.table(pd.DataFrame(results.items(), columns=["Metric", "Value"]))

# -------------------------------
# SCENARIO ANALYSIS
# -------------------------------
st.header("Scenario Analysis")
scenarios = {"Conservative": 0.9, "Base": 1.0, "Aggressive": 1.1}
scenario_data = []
for name, factor in scenarios.items():
    rev = monthly_revenue * factor
    profit = rev - (total_therapist_costs + monthly_overhead)
    scenario_data.append([name, round(rev, 2), round(profit, 2)])
st.table(pd.DataFrame(scenario_data, columns=["Scenario", "Revenue ($)", "Profit ($)"]))

# Scenario chart
scenario_df = pd.DataFrame(scenario_data, columns=["Scenario", "Revenue", "Profit"])
st.bar_chart(scenario_df.set_index("Scenario")[["Revenue", "Profit"]])

# -------------------------------
# MONTE CARLO SIMULATION
# -------------------------------
st.header("Monte Carlo Simulation")

num_simulations = st.slider("Number of Monte Carlo Runs", 100, 5000, 1000, 100)
run_mc = st.checkbox("Run Monte Carlo Simulation")

if run_mc:
    # Randomize revenue ±10%
    revenue_sim = np.random.normal(monthly_revenue, monthly_revenue * 0.1, num_simulations)
    # Randomize costs ±5%
    costs_sim = np.random.normal(total_therapist_costs + monthly_overhead, (total_therapist_costs + monthly_overhead) * 0.05, num_simulations)
    profit_sim = revenue_sim - costs_sim

    mc_results = pd.DataFrame({
        "Mean Profit ($)": [round(profit_sim.mean(), 2)],
        "5th Percentile ($)": [round(np.percentile(profit_sim, 5), 2)],
        "95th Percentile ($)": [round(np.percentile(profit_sim, 95), 2)]
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
# MARKETING BUDGET VS REQUIRED CLIENTS
# -------------------------------
st.header("Marketing Budget Analysis")
marketing_analysis = {
    "Marketing Budget ($)": round(marketing_budget or 0, 2),
    "Required Marketing for Churn Replacement ($)": round(required_marketing or 0, 2),
    "Budget Suffices?": "Yes" if budget_ok else "No",
    "Achievable New Clients with Budget": round(achievable_new_clients or 0, 2),
    "Required New Clients": round(required_new_clients or 0, 2)
}
st.table(pd.DataFrame(marketing_analysis.items(), columns=["Metric", "Value"]))

# Optional chart: New clients vs achievable
fig2, ax2 = plt.subplots()
ax2.bar(["Required", "Achievable"], [required_new_clients, achievable_new_clients], color=["orange", "green"])
ax2.set_ylabel("Number of Clients")
ax2.set_title("New Clients: Required vs Achievable with Marketing Budget")
st.pyplot(fig2)
