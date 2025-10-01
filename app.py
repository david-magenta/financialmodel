import streamlit as st

st.set_page_config(page_title="Therapy Practice Financial Model", layout="wide")

st.title("🧮 Therapy Practice Financial Model")

# Sidebar inputs
st.sidebar.header("Input Assumptions")

# Session economics
insurance_payout = st.sidebar.number_input("Insurance payout per session ($)", 50, 300, 80)
therapist_pay = st.sidebar.number_input("Therapist pay per session ($)", 20, 200, 45)
self_pay_rate = st.sidebar.number_input("Self-pay rate per session ($)", 50, 300, 150)
percent_self_pay = st.sidebar.slider("Percent of clients self-pay (%)", 0, 100, 20)
sessions_per_client = st.sidebar.number_input("Client session frequency (per month)", 1.0, 10.0, 2.5)
no_show_rate = st.sidebar.slider("No-show rate (%)", 0, 50, 12)

# Therapist assumptions
therapists = st.sidebar.number_input("Number of therapists", 1, 50, 3)
clients_per_therapist = st.sidebar.number_input("Unique clients per therapist per month", 1, 100, 26)
utilization_rate = st.sidebar.slider("Therapist utilization rate (%)", 10, 100, 75)

# Overhead
tech_cost = st.sidebar.number_input("Tech cost per therapist per month ($)", 0, 1000, 150)
admin_cost = st.sidebar.number_input("Admin overhead per therapist per month ($)", 0, 2000, 200)
other_overhead = st.sidebar.number_input("Other overhead per therapist per month ($)", 0, 2000, 125)

# Marketing
marketing_budget = st.sidebar.number_input("Monthly marketing budget ($)", 0, 20000, 1000)
cost_per_lead = st.sidebar.number_input("Cost per lead ($)", 1, 500, 35)
conversion_rate = st.sidebar.slider("Lead-to-client conversion rate (%)", 1, 100, 25)

# Calculations
effective_revenue_per_session = (
    (insurance_payout * (100 - percent_self_pay) / 100) +
    (self_pay_rate * (percent_self_pay / 100))
) * (1 - no_show_rate/100)

overhead_per_therapist = tech_cost + admin_cost + other_overhead
sessions_per_therapist = clients_per_therapist * sessions_per_client * (utilization_rate/100)

revenue_per_therapist = sessions_per_therapist * effective_revenue_per_session
cost_per_therapist = (sessions_per_therapist * therapist_pay) + overhead_per_therapist
profit_per_therapist = revenue_per_therapist - cost_per_therapist

# CAC and clients from marketing
new_clients = (marketing_budget / cost_per_lead) * (conversion_rate/100)
if new_clients > 0:
    cac = marketing_budget / new_clients
else:
    cac = 0

# Outputs
st.subheader("Practice Financials (per month)")
st.metric("Revenue per therapist", f"${revenue_per_therapist:,.0f}")
st.metric("Cost per therapist", f"${cost_per_therapist:,.0f}")
st.metric("Profit per therapist", f"${profit_per_therapist:,.0f}")

st.subheader("Practice Totals")
st.metric("Total Revenue", f"${revenue_per_therapist * therapists:,.0f}")
st.metric("Total Profit", f"${profit_per_therapist * therapists:,.0f}")

st.subheader("Marketing & CAC")
st.write(f"Estimated new clients per month: **{new_clients:.1f}**")
st.write(f"Customer Acquisition Cost (CAC): **${cac:,.2f}**")
