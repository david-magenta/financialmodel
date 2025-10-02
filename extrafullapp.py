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
num_therapists = st.sidebar.number_input("Number of Therapists", min_value=1, value=3)
sessions_per_week_per_therapist = st.sidebar.number_input("Avg Weekly Sessions per Therapist", min_value=1, value=20)
avg_fee = st.sidebar.number_input("Avg Collected Fee per Session ($)", min_value=50.0, value=150.0, step=10.0)
therapist_pay_per_session = st.sidebar.number_input("Therapist Pay per Session ($)", min_value=0.0, value=50.0, step=5.0)
no_show_rate = st.sidebar.slider("No-Show Rate (%)", 0,50,12)/100
utilization_rate = st.sidebar.slider("Utilization Rate (%)",0,100,85)/100

st.sidebar.header("Overhead")
tech_cost_per_therapist = st.sidebar.number_input("Tech Stack Cost per Therapist ($/month)", min_value=0.0, value=150.0, step=25.0)
admin_overhead_per_therapist = st.sidebar.number_input("Admin Overhead per Therapist ($/month)", min_value=0.0, value=200.0, step=25.0)
other_overhead_per_therapist = st.sidebar.number_input("Other Overhead per Therapist ($/month)", min_value=0.0, value=125.0, step=25.0)
monthly_fixed_overhead_cash = st.sidebar.number_input("Other Fixed Overhead ($/month)", min_value=0.0, value=500.0, step=50.0)

st.sidebar.header("Client & Churn")
avg_sessions_per_client_per_month = st.sidebar.number_input("Avg Sessions per Client per Month", min_value=1.0, value=2.5, step=0.1)
month1_churn = st.sidebar.slider("Month 1 Churn (%)", 0,100,25)/100
month2_churn = st.sidebar.slider("Month 2 Churn (%)", 0,100,15)/100
month3_churn = st.sidebar.slider("Month 3 Churn (%)", 0,100,10)/100
ongoing_churn = st.sidebar.slider("Ongoing Churn (%)", 0,20,5)/100
average_client_lifespan = st.sidebar.number_input("Average Client Lifespan (months)", min_value=1, value=8)

st.sidebar.header("Marketing & Acquisition")
monthly_marketing_budget = st.sidebar.number_input("Monthly Marketing Budget ($)", min_value=0.0, value=1000.0, step=50.0)
client_acquisition_cost = st.sidebar.number_input("Client Acquisition Cost ($/new client)", min_value=0.0, value=200.0, step=10.0)
target_profit_margin = st.sidebar.slider("Target Profit Margin (%)",0,100,25)/100
max_payback_months = st.sidebar.number_input("Max Payback Period (months)", min_value=1, value=4)
required_roi = st.sidebar.number_input("Required ROI (%)", min_value=0.0, value=200.0)/100
insurance_payment_delay = st.sidebar.number_input("Insurance Payment Delay (months)", min_value=0.0, value=1.5, step=0.1)
seasonal_variation = st.sidebar.slider("Seasonal Variation (%)", 0,50,20)/100
market_competition_factor = st.sidebar.number_input("Market Competition Factor", min_value=0.1, value=1.0, step=0.1)

st.sidebar.header("Simulation & Scenario")
months_to_simulate = st.sidebar.number_input("Months to Simulate", min_value=1, value=12)
num_simulations = st.sidebar.slider("Monte Carlo Runs", 100,5000,1000,100)
scenario = st.sidebar.selectbox("Scenario", ["Base","Conservative","Aggressive"])

# -------------------------------
# SCENARIO MULTIPLIER
# -------------------------------
scenario_multiplier = {"Base":1.0,"Conservative":0.9,"Aggressive":1.1}[scenario]
avg_fee *= scenario_multiplier
avg_sessions_per_client_per_month *= scenario_multiplier
therapist_pay_per_session *= scenario_multiplier

# -------------------------------
# CORE CALCULATIONS
# -------------------------------
sessions_per_month_per_therapist = sessions_per_week_per_therapist*4*utilization_rate
total_sessions_capacity = sessions_per_month_per_therapist*num_therapists
effective_fee_per_session = avg_fee*(1-no_show_rate)
overhead_per_therapist = tech_cost_per_therapist + admin_overhead_per_therapist + other_overhead_per_therapist
contribution_margin_per_session = effective_fee_per_session - (therapist_pay_per_session + overhead_per_therapist/total_sessions_capacity)

# -------------------------------
# MONTH-BY-MONTH SIMULATION
# -------------------------------
active_clients=[0]
new_clients_per_month=[]
churned_clients_per_month=[]
monthly_revenue=[]
monthly_cost=[]
monthly_profit=[]
cash_balance = [0]

for month in range(1, months_to_simulate+1):
    if month==1: churn_rate = month1_churn
    elif month==2: churn_rate = month2_churn
    elif month==3: churn_rate = month3_churn
    else: churn_rate = ongoing_churn

    surviving_clients = active_clients[-1]*(1-churn_rate)
    churned_clients = active_clients[-1]-surviving_clients
    churned_clients_per_month.append(churned_clients)

    max_new_clients = monthly_marketing_budget/client_acquisition_cost if client_acquisition_cost>0 else 0
    required_new_clients = churned_clients
    new_clients = min(max_new_clients, required_new_clients)
    new_clients_per_month.append(new_clients)

    total_clients = surviving_clients+new_clients
    active_clients.append(total_clients)

    expected_sessions = total_clients*avg_sessions_per_client_per_month
    revenue = expected_sessions*effective_fee_per_session
    cost = expected_sessions*(therapist_pay_per_session+overhead_per_therapist/total_sessions_capacity)+monthly_fixed_overhead_cash+new_clients*client_acquisition_cost
    profit = revenue-cost

    # Cash flow with insurance delay
    delayed_revenue = revenue if month > insurance_payment_delay else 0
    cash_balance.append(cash_balance[-1]+delayed_revenue-cost)

    monthly_revenue.append(revenue)
    monthly_cost.append(cost)
    monthly_profit.append(profit)

active_clients=active_clients[1:]

# -------------------------------
# THERAPIST-LEVEL P&L
# -------------------------------
clients_per_therapist=np.array(active_clients)/num_therapists
sessions_per_therapist=clients_per_therapist*avg_sessions_per_client_per_month
revenue_per_therapist=sessions_per_therapist*effective_fee_per_session
therapist_session_cost=sessions_per_therapist*therapist_pay_per_session
therapist_overhead=overhead_per_therapist
cost_per_therapist=therapist_session_cost+therapist_overhead
profit_per_therapist=revenue_per_therapist-cost_per_therapist

therapist_pnl_df=pd.DataFrame({
    "Month":range(1,months_to_simulate+1),
    "Clients per Therapist":np.round(clients_per_therapist,2),
    "Sessions per Therapist":np.round(sessions_per_therapist,2),
    "Revenue ($)":np.round(revenue_per_therapist,2),
    "Cost ($)":np.round(cost_per_therapist,2),
    "Profit ($)":np.round(profit_per_therapist,2)
})

# -------------------------------
# CLIENT LIFETIME VALUE
# -------------------------------
clv_per_client=[]
for month_idx in range(months_to_simulate):
    survival=1.0
    client_sessions=[]
    for m in range(month_idx,month_idx+int(average_client_lifespan)):
        if m==0: churn_rate=month1_churn
        elif m==1: churn_rate=month2_churn
        elif m==2: churn_rate=month3_churn
        else: churn_rate=ongoing_churn
        survival*=(1-churn_rate)
        client_sessions.append(avg_sessions_per_client_per_month*survival)
    discounted_value=sum([s*contribution_margin_per_session/((1+0.005)**(i+1)) for i,s in enumerate(client_sessions)])
    adjusted_value=discounted_value*(1-seasonal_variation)/market_competition_factor
    clv_per_client.append(adjusted_value)
average_clv=np.mean(clv_per_client)
profit_margin_limit=average_clv*(1-target_profit_margin)
payback_limit=sum([avg_sessions_per_client_per_month*contribution_margin_per_session for _ in range(max_payback_months)])
roi_limit=average_clv/(1+required_roi)
max_acquisition_cost=min(profit_margin_limit,payback_limit,roi_limit)

# -------------------------------
# DASHBOARD & CHARTS
# -------------------------------
summary_df=pd.DataFrame({
    "Month":range(1,months_to_simulate+1),
    "Active Clients":np.round(active_clients,2),
    "New Clients":np.round(new_clients_per_month,2),
    "Churned Clients":np.round(churned_clients_per_month,2),
    "Revenue ($)":np.round(monthly_revenue,2),
    "Cost ($)":np.round(monthly_cost,2),
    "Profit ($)":np.round(monthly_profit,2),
    "Cash Balance ($)":np.round(cash_balance[1:],2)
})

st.header("Monthly Client & Profit Summary")
st.dataframe(summary_df)

st.header("Key Metrics")
col1,col2,col3,col4=st.columns(4)
col1.metric("Average CLV per Client ($)",np.round(average_clv,2))
col2.metric("Max Acquisition Cost ($)",np.round(max_acquisition_cost,2))
col3.metric("Total Profit ($)",np.round(sum(monthly_profit),2))
col4.metric("Marketing Budget Suffices?","Yes" if sum(new_clients_per_month)*client_acquisition_cost<=monthly_marketing_budget*months_to_simulate else "No")

st.header("Therapist-Level Monthly P&L")
st.dataframe(therapist_pnl_df)

st.header("Active Clients Over Time")
st.line_chart(summary_df.set_index("Month")["Active Clients"])

st.header("Cash Balance Over Time")
st.line_chart(summary_df.set_index("Month")["Cash Balance ($)"])

st.header("New vs Required Clients vs Marketing Budget")
achievable_new_clients=monthly_marketing_budget/client_acquisition_cost if client_acquisition_cost>0 else 0
fig2,ax2=plt.subplots()
ax2.bar(["Required New Clients","Achievable New Clients"],[np.round(np.mean(new_clients_per_month),2),np.round(achievable_new_clients,2)],color=["orange","green"])
ax2.set_ylabel("Clients")
ax2.set_title("Monthly New Clients vs Marketing Budget")
st.pyplot(fig2)

st.header("Profit Distribution (Monte Carlo Simulation)")
simulated_profits=[]
for _ in range(num_simulations):
    variation=np.random.normal(1.0,0.05,months_to_simulate)
    simulated_profit=np.sum(np.array(monthly_profit)*variation)
    simulated_profits.append(simulated_profit)
fig3,ax3=plt.subplots()
ax3.hist(simulated_profits,bins=50,color="skyblue")
ax3.set_title("Profit Distribution Across Monte Carlo Runs")
ax3.set_xlabel("Total Profit ($)")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)

st.header("Therapist Profit Heatmap")
profit_matrix=profit_per_therapist.reshape(-1,1)
fig4,ax4=plt.subplots()
sns.heatmap(profit_matrix,annot=True,fmt=".0f",cmap="YlGnBu",ax=ax4)
ax4.set_ylabel("Month")
ax4.set_xlabel("Therapist")
ax4.set_title("Therapist Profit by Month")
st.pyplot(fig4)

st.header("Revenue vs Cost per Therapist (Stacked Bar)")
fig5,ax5=plt.subplots()
ax5.bar(range(1,months_to_simulate+1),revenue_per_therapist[0],label="Revenue",color="green")
ax5.bar(range(1,months_to_simulate+1),cost_per_therapist[0],bottom=revenue_per_therapist[0],label="Cost",color="red",alpha=0.6)
ax5.set_xlabel("Month")
ax5.set_ylabel("$")
ax5.set_title("Revenue vs Cost per Therapist (Sample Therapist 1)")
ax5.legend()
st.pyplot(fig5)
