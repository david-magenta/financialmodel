import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Therapy Practice Financial Model", layout="wide")
st.title("🧮 Therapy Practice Financial Model")

st.sidebar.header("Input Assumptions")

# Session Economics
insurance_payout = st.sidebar.number_input("Insurance payout per session ($)", 50, 300, 80)
therapist_pay = st.sidebar.number_input("Therapist pay per session ($)", 20, 200, 45)
self_pay_rate = st.sidebar.number_input("Self-pay rate per session ($)", 50, 300, 150)
percent_self_pay = st.sidebar.slider("Percent of clients self-pay (%)", 0, 100, 20)
sessions_per_client = st.sidebar.number_input("Client session frequency (per month)", 1.0, 10.0, 2.5)
no_show_rate = st.sidebar.slider("No-show rate (%)", 0, 50, 12)

# Overhead
tech_cost = st.sidebar.number_input("Tech cost per therapist per month ($)", 0, 1000, 150)
admin_cost = st.sidebar.number_input("Admin overhead per therapist per month ($)", 0, 2000, 200)
other_overhead = st.sidebar.number_input("Other overhead per therapist per month ($)", 0, 2000, 125)

# Therapist assumptions
therapists = st.sidebar.number_input("Number of therapists", 1, 50, 3)
clients_per_therapist = st.sidebar.number_input("Unique clients per therapist per month", 1, 100, 26)
utilization_rate = st.sidebar.slider("Therapist utilization rate (%)", 10, 100, 75)

# Churn / CLV
month1_churn = st.sidebar.slider("Month 1 churn (%)", 0, 100, 25)
month2_churn = st.sidebar.slider("Month 2 churn (%)", 0, 100, 15)
month3_churn = st.sidebar.slider("Month 3 churn (%)", 0, 100, 10)
ongoing_churn = st.sidebar.slider("Ongoing churn (%)", 0, 100, 5)
avg_lifespan_months = st.sidebar.number_input("Average client lifespan (months)", 1, 36, 8)

# Marketing
marketing_budget = st.sidebar.number_input("Monthly marketing budget ($)", 0, 20000, 1000)
cost_per_lead = st.sidebar.number_input("Cost per lead ($)", 1, 500, 35)
conversion_rate = st.sidebar.slider("Lead-to-client conversion rate (%)", 1, 100, 25)

# Business targets
target_profit_margin = st.sidebar.slider("Target profit margin (%)", 0, 100, 25)
max_payback = st.sidebar.number_input("Max payback period (months)", 1, 12, 4)
required_roi = st.sidebar.number_input("Required ROI (%)", 0, 500, 200)
insurance_delay = st.sidebar.number_input("Insurance payment delay (months)", 0.0, 6.0, 1.5)
seasonal_variation = st.sidebar.slider("Seasonal variation (%)", 0, 100, 20)
competition_factor = st.sidebar.number_input("Market competition factor", 0.1, 5.0, 1.0)

# Effective revenue per session
effective_revenue_per_session = (
    (insurance_payout * (100 - percent_self_pay) / 100) +
    (self_pay_rate * (percent_self_pay / 100))
) * (1 - no_show_rate/100)

# Overhead per therapist
overhead_per_therapist = tech_cost + admin_cost + other_overhead
sessions_per_therapist = clients_per_therapist * sessions_per_client * (utilization_rate/100)

# Contribution margin per session
contribution_per_session = effective_revenue_per_session - therapist_pay - (overhead_per_therapist / sessions_per_therapist)

# CLV calculation with churn
months = np.arange(1, avg_lifespan_months + 1)
survival = np.ones(avg_lifespan_months)
survival[0] = 1 - month1_churn/100
if avg_lifespan_months > 1:
    survival[1] = survival[0] * (1 - month2_churn/100)
if avg_lifespan_months > 2:
    survival[2] = survival[1] * (1 - month3_churn/100)
for i in range(3, avg_lifespan_months):
    survival[i] = survival[i-1] * (1 - ongoing_churn/100)

# Expected sessions and CLV
expected_sessions = survival * sessions_per_client
discount_factor = 1 / ((1 + 0.005) ** (months + insurance_delay))  # 0.5% monthly discount
clv_per_client = np.sum(expected_sessions * contribution_per_session * discount_factor)

# Adjust for seasonality and competition
clv_per_client_adjusted = clv_per_client * (1 - seasonal_variation*0.5/100) / competition_factor

# Acquisition cost constraints
max_cac_profit = clv_per_client_adjusted * (1 - target_profit_margin/100)
payback_sessions = min(max_payback, avg_lifespan_months)
max_cac_payback = np.sum(expected_sessions[:payback_sessions] * contribution_per_session)
max_cac_roi = clv_per_client_adjusted / (1 + required_roi/100)
max_cac = min(max_cac_profit, max_cac_payback, max_cac_roi)

# Marketing -> new clients
new_clients = (marketing_budget / cost_per_lead) * (conversion_rate/100)
cac_actual = marketing_budget / new_clients if new_clients > 0 else 0

# Outputs
st.subheader("Client Lifetime Value (CLV)")
st.metric("CLV per client (adjusted)", f"${clv_per_client_adjusted:,.2f}")
st.metric("Max Acquisition Cost per client", f"${max_cac:,.2f}")

st.subheader("Per Therapist Financials")
revenue_per_therapist = sessions_per_therapist * effective_revenue_per_session
cost_per_therapist = (sessions_per_therapist * therapist_pay) + overhead_per_therapist
profit_per_therapist = revenue_per_therapist - cost_per_therapist
st.metric("Revenue", f"${revenue_per_therapist:,.0f}")
st.metric("Cost", f"${cost_per_therapist:,.0f}")
st.metric("Profit", f"${profit_per_therapist:,.0f}")

st.subheader("Practice Totals")
st.metric("Total Revenue", f"${revenue_per_therapist * therapists:,.0f}")
st.metric("Total Profit", f"${profit_per_therapist * therapists:,.0f}")

st.subheader("Marketing & CAC")
st.write(f"Estimated new clients per month: **{new_clients:.1f}**")
st.write(f"Customer Acquisition Cost (CAC): **${cac_actual:,.2f}**")

# Optional: Monthly CLV curve
st.subheader("Month-by-Month Expected Contribution per Client")
clv_df = pd.DataFrame({
    "Month": months,
    "Survival Rate": survival,
    "Expected Sessions": expected_sessions,
    "Contribution": expected_sessions * contribution_per_session,
    "Discounted Contribution": expected_sessions * contribution_per_session * discount_factor
})
st.line_chart(clv_df.set_index("Month")["Discounted Contribution"])
