# upgraded_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Therapy Practice Financial Model — Full", layout="wide")
st.title("💼 Therapy Practice Financial Model — Full Upgrade")

# ---------------------------
# Helper functions
# ---------------------------
def safe_div(a, b):
    return a / b if b != 0 else 0.0

def compute_effective_revenue_per_session(insurance_payout, self_pay_rate, percent_self_pay,
                                          copay_per_session, copay_collected_rate, no_show_rate):
    """
    Returns tuple:
      (effective_revenue_per_session, revenue_breakdown_dict)
    Definition:
      - percent_self_pay is percent of sessions that are self-pay (0..100)
      - remaining sessions are 'insurance' sessions and include copay (if applicable)
      - no_show_rate reduces effective sessions (so revenue)
      - copay_collected_rate is fraction of copays actually collected at time of service
    """
    p_self = percent_self_pay / 100.0
    p_ins = 1.0 - p_self

    # Self-pay revenue per scheduled session (after no-show)
    self_pay_rev = self_pay_rate * (1 - no_show_rate/100.0)

    # Insurance revenue per scheduled session (payout + copay portion)
    # insurance payout assumed to include what the payer pays for the service (not including copay)
    insurance_rev = (insurance_payout + copay_per_session * copay_collected_rate) * (1 - no_show_rate/100.0)

    effective = self_pay_rev * p_self + insurance_rev * p_ins
    breakdown = {
        "self_pay_rev_per_session": self_pay_rev,
        "insurance_rev_per_session": insurance_rev,
        "pct_self_pay": p_self,
        "pct_insurance": p_ins,
        "effective_revenue_per_session": effective
    }
    return effective, breakdown

def compute_overhead_per_session(tech, admin_alloc, other, clients_per_therapist, sessions_per_client):
    # allocate monthly overhead per therapist to per-session level
    sessions_per_month = clients_per_therapist * sessions_per_client if sessions_per_client > 0 else 0
    monthly_overhead = tech + admin_alloc + other
    return safe_div(monthly_overhead, sessions_per_month) if sessions_per_month>0 else float('inf')

def survival_curve(month1, month2, month3, ongoing, months):
    """
    Returns numpy array of survival probabilities for months 1..months.
    month1..ongoing are percents (0..100).
    """
    surv = np.zeros(months)
    if months >= 1:
        surv[0] = 1.0 * (1 - month1/100.0)
    if months >= 2:
        surv[1] = surv[0] * (1 - month2/100.0)
    if months >= 3:
        surv[2] = surv[1] * (1 - month3/100.0)
    for i in range(3, months):
        surv[i] = surv[i-1] * (1 - ongoing/100.0)
    # Prepend month 0 survival = 1 if needed by caller. Current convention: index 0 is month1 survival.
    return surv

def compute_expected_sessions_per_client(survival, sessions_per_client):
    # survival is array month1..n survival probabilities
    return survival * sessions_per_client

def pv_discount_factors(months_array, monthly_discount_rate, payment_delay_months):
    """
    months_array: numpy array [1..N] representing month numbers
    monthly_discount_rate: decimal (e.g., 0.005 for 0.5%/month)
    payment_delay_months: months of delay before collection
    """
    # discount factor for cash received in month t is (1 + r)^(t + delay)
    return 1.0 / ((1.0 + monthly_discount_rate) ** (months_array + payment_delay_months))

def compute_clv(expected_sessions, contrib_per_session, monthly_discount_rate, payment_delay_months, seasonal_variation_pct, competition_factor):
    months = np.arange(1, len(expected_sessions)+1)
    dfs = pv_discount_factors(months, monthly_discount_rate, payment_delay_months)
    undiscounted = expected_sessions * contrib_per_session
    discounted = undiscounted * dfs
    gross = np.sum(discounted)
    # Adjustments
    seasonal_adj = 1.0 - (seasonal_variation_pct * 0.5/100.0)
    adjusted = gross * seasonal_adj / competition_factor
    return {
        "gross_clv": float(gross),
        "adjusted_clv": float(adjusted),
        "undiscounted_breakdown": undiscounted.tolist(),
        "discounted_breakdown": discounted.tolist()
    }

def compute_max_cac(clv_adjusted, target_profit_margin_pct, payback_months, expected_sessions, contrib_per_session, required_roi_pct):
    # 1) profit margin constraint
    margin_limit = clv_adjusted * (1 - target_profit_margin_pct/100.0)
    # 2) payback constraint - sum of contribution in first payback_months (undiscounted contributions)
    payback_months = int(min(payback_months, len(expected_sessions)))
    payback_value = float(np.sum(expected_sessions[:payback_months] * contrib_per_session))
    # 3) ROI constraint
    roi_limit = clv_adjusted / (1 + required_roi_pct/100.0)
    conservative = min(margin_limit, payback_value, roi_limit)
    return {
        "profit_margin_limit": margin_limit,
        "payback_limit": payback_value,
        "roi_limit": roi_limit,
        "max_cac_conservative": conservative
    }

def build_monthly_cohorts(initial_clients, monthly_new_clients_series, months_projection, sessions_per_client, churn_params):
    """
    Build a matrix cohorts x months of active clients counts.
    - initial_clients: number at month 0 (starting point)
    - monthly_new_clients_series: length M array with number of new clients added in months 1..M
    - months_projection: total months to project
    - sessions_per_client: sessions per client per month (assumed constant)
    - churn_params: dict with month1, month2, month3, ongoing churn % (per-month)
    Returns:
      cohorts_active_clients: shape (num_cohorts, months_projection)
        where cohort 0 = initial cohort, cohort i = clients added in month i (1-indexed)
    """
    months = months_projection
    # Number of cohorts is 1 + months (initial + each month new cohort)
    cohorts = 1 + months
    mat = np.zeros((cohorts, months))
    # initial cohort index 0 starts at month 0 (we will align: mat[:,month_index])
    # but we model survival starting month1 as the first month after acquisition or initial month?
    # We'll set cohort added in month m (1-based) to begin yielding sessions in that month.
    # For initial cohort we assume they exist at month 1 (index 0).
    # Build survival array length months for churn starting at month1.
    surv = survival_curve(churn_params["month1"], churn_params["month2"], churn_params["month3"], churn_params["ongoing"], months)
    # initial cohort: initial_clients * survival curve starting at month1
    for t in range(months):
        mat[0, t] = initial_clients * surv[t] if t < len(surv) else 0.0
    # monthly new cohorts
    for m in range(1, months+1):  # m=1..months
        # cohort index m corresponds to clients acquired in month m (1-based)
        cohort_idx = m
        clients_added = monthly_new_clients_series[m-1] if m-1 < len(monthly_new_clients_series) else 0.0
        # in month m (index m-1) they experience survival[0] fraction, month m+1 -> survival[1], etc.
        for t in range(m-1, months):
            survival_index = t - (m-1)  # 0-based into survival array
            if survival_index < len(surv):
                mat[cohort_idx, t] = clients_added * surv[survival_index]
            else:
                mat[cohort_idx, t] = 0.0
    return mat  # rows = cohorts, columns = months

def schedule_collections_for_billed(billed_amounts_by_month, ar_distribution):
    """
    billed_amounts_by_month: array length M with amounts billed in month m (accrual)
    ar_distribution: dict of percentages e.g. {"30":0.6,"60":0.3,"90":0.1}
      meaning collected in next months with these shares.
    Return: collected_cash_by_month array length M+max_delay.
    For billed in month t (0-based), collections:
      month t + 1 receives billed * ar_distribution["30"]
      month t + 2 receives billed * ar_distribution["60"]
      etc.
    """
    max_delay_months = max([int(k) for k in ar_distribution.keys()]) // 30
    total_months = len(billed_amounts_by_month) + max_delay_months + 1
    collected = np.zeros(total_months)
    for t, billed in enumerate(billed_amounts_by_month):
        for k, pct in ar_distribution.items():
            try:
                delay = int(k) // 30
            except:
                delay = 1
            collect_month = t + delay
            if collect_month < len(collected):
                collected[collect_month] += billed * pct
    return collected

# ---------------------------
# Inputs (organized)
# ---------------------------
st.sidebar.header("Quick scenario")
scenario_name = st.sidebar.text_input("Scenario name", "Base scenario")

with st.sidebar.expander("Session & pricing"):
    insurance_payout = st.number_input("Avg insurance payout per session ($)", min_value=10.0, value=80.0, step=1.0)
    self_pay_rate = st.number_input("Self-pay session rate ($)", min_value=20.0, value=150.0, step=1.0)
    percent_self_pay = st.slider("Percent of sessions that are self-pay (%)", min_value=0, max_value=100, value=20)
    copay_per_session = st.number_input("Average copay per session ($)", min_value=0.0, value=20.0)
    copay_collected_rate = st.slider("Percentage of copays collected at time of service (%)", 0, 100, 90)

with st.sidebar.expander("Therapists & productivity"):
    therapists = st.number_input("Number of therapists (FTE)", min_value=1, max_value=200, value=3)
    clients_per_therapist = st.number_input("Unique clients per therapist (active caseload) per month", min_value=1, max_value=400, value=26)
    session_frequency_per_client = st.number_input("Sessions per client per month", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
    utilization_pct = st.slider("Therapist utilization (booked clinical hours / available) (%)", 10, 100, 75)
    caseload_cap = st.number_input("Caseload cap per therapist (clients)", min_value=1, max_value=400, value=30)
    onboarding_months = st.number_input("Onboarding months to full caseload", min_value=0, max_value=12, value=3)
    annual_turnover_pct = st.number_input("Therapist annual turnover (%)", min_value=0.0, max_value=200.0, value=15.0)

with st.sidebar.expander("Comp & payroll"):
    therapist_pay_per_session = st.number_input("Therapist pay per session ($)", min_value=5.0, value=45.0)
    payroll_tax_and_benefits_pct = st.number_input("Payroll taxes & benefits (% of pay)", min_value=0.0, max_value=200.0, value=10.0)

with st.sidebar.expander("Overhead & admin"):
    tech_cost_per_therapist = st.number_input("Tech stack $ / therapist / month", min_value=0.0, value=150.0)
    admin_alloc_per_therapist = st.number_input("Admin allocation $ / therapist / month", min_value=0.0, value=200.0)
    other_overhead_per_therapist = st.number_input("Other overhead $ / therapist / month", min_value=0.0, value=125.0)
    admin_ratio = st.number_input("Therapists per admin (hire when exceeded)", min_value=1, value=8)
    recruiting_cost_per_hire = st.number_input("Recruiting cost per new therapist ($)", min_value=0.0, value=2000.0)
    onboarding_cost_per_hire = st.number_input("Onboarding cost per new therapist ($)", min_value=0.0, value=1500.0)
    switch_to_internal_billing_threshold = st.number_input("Switch to internal billing at N therapists", min_value=1, value=15)

with st.sidebar.expander("Billing & finance"):
    external_biller_pct_collections = st.number_input("External biller fee (% of insurance collections)", min_value=0.0, max_value=50.0, value=6.0)
    cc_processing_pct = st.number_input("Credit card fee on self-pay & copay (%)", min_value=0.0, max_value=10.0, value=0.029)
    bad_debt_pct = st.number_input("Bad debt / write-offs (% of billed)", min_value=0.0, max_value=100.0, value=3.0)
    monthly_discount_rate_pct = st.number_input("Monthly discount rate (%)", min_value=0.0, max_value=100.0, value=0.5)
    insurance_payment_delay_months = st.number_input("Insurance payment delay (months)", min_value=0.0, max_value=12.0, value=1.5)
    ar_dist_30 = st.number_input("AR distribution: % collected at ~30d", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
    ar_dist_60 = st.number_input("AR distribution: % collected at ~60d", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
    ar_dist_90 = st.number_input("AR distribution: % collected at ~90d", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
    # normalize later

with st.sidebar.expander("Marketing & acquisition"):
    monthly_marketing_budget = st.number_input("Monthly marketing budget ($)", min_value=0.0, value=1000.0)
    cost_per_lead = st.number_input("Cost per lead (CPL $)", min_value=0.1, value=35.0)
    lead_to_intake_rate_pct = st.number_input("Lead → intake conversion (%)", min_value=0.0, max_value=100.0, value=25.0)
    intake_to_first_session_pct = st.number_input("Intake → first session (%)", min_value=0.0, max_value=100.0, value=60.0)
    referral_pct = st.slider("Percent of new clients from referral (%)", 0, 100, 20)

with st.sidebar.expander("Churn & CLV"):
    month1_churn = st.number_input("Month1 churn (%)", min_value=0.0, max_value=100.0, value=25.0)
    month2_churn = st.number_input("Month2 churn (%)", min_value=0.0, max_value=100.0, value=15.0)
    month3_churn = st.number_input("Month3 churn (%)", min_value=0.0, max_value=100.0, value=10.0)
    ongoing_churn = st.number_input("Ongoing monthly churn (%)", min_value=0.0, max_value=100.0, value=5.0)
    avg_lifespan_months = st.number_input("Model client lifespan (months) for CLV", min_value=1, max_value=60, value=12)

with st.sidebar.expander("Targets & constraints"):
    target_profit_margin_pct = st.number_input("Target profit margin (%)", min_value=0.0, max_value=100.0, value=25.0)
    max_payback_months = st.number_input("Max acceptable CAC payback (months)", min_value=1, max_value=36, value=4)
    required_roi_pct = st.number_input("Required ROI on acquisition (%)", min_value=0.0, max_value=1000.0, value=200.0)
    seasonal_variation_pct = st.number_input("Seasonal variation (%)", min_value=0.0, max_value=100.0, value=20.0)
    competition_factor = st.number_input("Competition adjustment factor (1 = base)", min_value=0.1, value=1.0)

with st.sidebar.expander("Run options / Misc"):
    months_projection = st.number_input("Months to project (P&L / Cashflow)", min_value=1, max_value=60, value=24)
    run_monte_carlo = st.checkbox("Run Monte Carlo on CLV & Max CAC (may be slow)", value=False)
    monte_carlo_samples = st.number_input("Monte Carlo samples", min_value=100, max_value=20000, value=2000)

# AR normalized
ar_sum = ar_dist_30 + ar_dist_60 + ar_dist_90
if ar_sum <= 0:
    st.sidebar.error("AR distribution must sum to > 0")
ar_dist = {"30": ar_dist_30 / ar_sum, "60": ar_dist_60 / ar_sum, "90": ar_dist_90 / ar_sum}

# ---------------------------
# Core calculations
# ---------------------------
with st.spinner("Calculating..."):
    # Effective revenue per scheduled session
    effective_revenue_per_session, rev_breakdown = compute_effective_revenue_per_session(
        insurance_payout, self_pay_rate, percent_self_pay, copay_per_session, copay_collected_rate, no_show_rate=0.0
    )
    # Note: we set no_show_rate=0 here for revenue per scheduled session calculation and handle no-show by reducing expected sessions.

    # Overhead per session (per therapist)
    overhead_per_session = compute_overhead_per_session(tech_cost_per_therapist, admin_alloc_per_therapist, other_overhead_per_therapist,
                                                        clients_per_therapist, session_frequency_per_client)

    # Sessions per therapist per month (booked sessions)
    sessions_per_therapist = clients_per_therapist * session_frequency_per_client * (utilization_pct / 100.0)

    # Contribution per scheduled session (before biller fees & cc fees)
    # We'll treat cc_processing_pct and external biller fees as applied on collections later.
    # Therapist total cost per session = therapist pay + payroll taxes & benefits per session + overhead per session.
    payroll_multiplier = 1.0 + payroll_tax_and_benefits_pct / 100.0
    therapist_total_cost_per_session = therapist_pay_per_session * payroll_multiplier
    base_contrib_per_session = effective_revenue_per_session - therapist_total_cost_per_session - overhead_per_session

    # Adjust for no-show (reduce expected sessions)
    effective_no_show_rate = st.sidebar.slider("No-show rate (%)", 0, 50, 12)  # small UI convenience
    # survival + expected sessions for CLV
    surv = survival_curve(month1_churn, month2_churn, month3_churn, ongoing_churn, int(avg_lifespan_months))
    expected_sessions_per_month_per_client = compute_expected_sessions_per_client(surv, session_frequency_per_client * (1 - effective_no_show_rate/100.0))

    # CLV (discounted) and adjustments
    clv = compute_clv(expected_sessions_per_month_per_client, base_contrib_per_session, monthly_discount_rate_pct/100.0,
                      insurance_payment_delay_months, seasonal_variation_pct, competition_factor)

    # CAC constraints
    cac_constraints = compute_max_cac(clv["adjusted_clv"], target_profit_margin_pct, max_payback_months,
                                      expected_sessions_per_month_per_client, base_contrib_per_session, required_roi_pct)

    # Marketing-derived new clients
    leads_per_month = safe_div(monthly_marketing_budget, cost_per_lead)
    new_clients_from_marketing = leads_per_month * (lead_to_intake_rate_pct/100.0) * (intake_to_first_session_pct/100.0)
    # referral clients add cheaply; model referral as multiplier
    monthly_new_clients = new_clients_from_marketing * (1 - referral_pct/100.0) + (monthly_marketing_budget * referral_pct/100.0 * 0.0)  # referrals not paid in this model

    # Build cohort matrix for P&L
    # Initial active clients in month 1
    initial_active_clients = therapists * clients_per_therapist
    monthly_new_clients_series = [monthly_new_clients] * months_projection
    cohort_matrix = build_monthly_cohorts(initial_active_clients, monthly_new_clients_series, months_projection, session_frequency_per_client * (1 - effective_no_show_rate/100.0),
                                          {"month1": month1_churn, "month2": month2_churn, "month3": month3_churn, "ongoing": ongoing_churn})

    # sessions each month = sum across cohorts of active clients * sessions per client
    sessions_by_month = np.sum(cohort_matrix, axis=0) * session_frequency_per_client * (1 - effective_no_show_rate/100.0) / session_frequency_per_client
    # simplified: sessions_by_month computed as sum(active clients) * sessions_per_client (but we already adjusted cohort building)
    # More directly:
    active_clients_by_month = np.sum(cohort_matrix, axis=0)
    sessions_by_month = active_clients_by_month * session_frequency_per_client  # sessions actually scheduled (no-show handled earlier in expected_sessions calculation)

    # billed amounts by month (accrual)
    # Use revenue breakouts to separate self-pay vs insurance
    pct_self = percent_self_pay / 100.0
    pct_insurance = 1.0 - pct_self
    billed_per_session_self = self_pay_rate * (1 - cc_processing_pct)
    billed_per_session_ins = insurance_payout  # copays handled below
    # copays: only on insurance sessions, some collected upfront (copay_collected_rate)
    billed_copay_collected_upfront_per_session = copay_per_session * copay_collected_rate

    billed_amounts_by_month = []
    for s in sessions_by_month:
        insurance_sessions = s * pct_insurance
        selfpay_sessions = s * pct_self
        billed_insurance = insurance_sessions * (billed_per_session_ins + billed_copay_collected_upfront_per_session)
        billed_selfpay = selfpay_sessions * billed_per_session_self
        total_billed = billed_insurance + billed_selfpay
        billed_amounts_by_month.append(total_billed)

    billed_amounts_by_month = np.array(billed_amounts_by_month)

    # Schedule collections according to AR distribution
    # Contract: ar_dist keys "30","60","90" expressed as fractional shares
    collected = schedule_collections_for_billed(billed_amounts_by_month, ar_dist)

    # Biller fees applied only on insurance collections when they are collected (not copays)
    # For simplicity, compute insurance portion collected by splitting billed amounts by pct_insurance and assuming copay portion is not subject to biller fee
    insurance_billed_by_month = sessions_by_month * pct_insurance * insurance_payout
    insurance_collected_schedule = schedule_collections_for_billed(insurance_billed_by_month, ar_dist)
    biller_fees_by_month = insurance_collected_schedule * (external_biller_pct_collections / 100.0)

    # credit card fees on self-pay and copay collected at time of service (we treated copay collected upfront in billed)
    selfpay_billed_by_month = sessions_by_month * pct_self * self_pay_rate
    # We assumed copay collected upfront: compute its immediate collected portion (month index aligns)
    copay_billed_by_month = sessions_by_month * pct_insurance * copay_per_session * copay_collected_rate
    immediate_cc_fees_by_month = (selfpay_billed_by_month + copay_billed_by_month) * cc_processing_pct

    # Bad debt write-offs assumed to apply to billed amounts (we'll subtract at accrual or collection depending on method)
    bad_debt_by_month = billed_amounts_by_month * (bad_debt_pct / 100.0)

    # Monthly payroll (cash) estimate: pay therapists as sessions occur (therapist_pay_per_session * sessions_by_month) + payroll taxes & benefits as cash outflow monthly
    payroll_cash_by_month = sessions_by_month * therapist_pay_per_session * payroll_multiplier

    # Monthly overhead cash (tech/admin/other) prorated monthly
    monthly_fixed_overhead_cash = (tech_cost_per_therapist + admin_alloc_per_therapist + other_overhead_per_therapist) * therapists

    # Marketing spend cash outflow monthly
    marketing_cash_by_month = np.array([monthly_marketing_budget] * months_projection)

    # Collections length may be larger than months_projection due to delays; trim to the length we want for cashflow view
    collected_trim = collected[:months_projection]
    biller_fees_trim = biller_fees_by_month[:months_projection]

    # Cashflow by month
    cash_collections = collected_trim
    cash_outflows = payroll_cash_by_month + monthly_fixed_overhead_cash + marketing_cash_by_month + immediate_cc_fees_by_month[:months_projection] + biller_fees_trim
    net_cash_by_month = cash_collections - cash_outflows

    # Accrual P&L (recognize billed amounts as revenue, expenses = payroll incurred + overhead)
    accrual_revenue = billed_amounts_by_month
    accrual_expenses = payroll_cash_by_month + monthly_fixed_overhead_cash + immediate_cc_fees_by_month[:months_projection] + biller_fees_trim + bad_debt_by_month
    accrual_net_income = accrual_revenue - accrual_expenses

    # Staffing plan
    needed_admins = int(np.ceil(therapists / admin_ratio))
    hire_admin_cost = 0.0  # we don't compute admin hire cash here beyond monthly salary allocation
    # Determine if switching to internal billing (rough decision)
    use_internal_billing = therapists >= switch_to_internal_billing_threshold

    # Checks & warnings
    warnings = []
    if base_contrib_per_session < 0:
        warnings.append("CONTRIBUTION PER SESSION IS NEGATIVE — check therapist pay, overhead, or payer mix.")
    if clv["adjusted_clv"] <= 0:
        warnings.append("ADJUSTED CLV <= 0 — check inputs.")
    if cac_constraints["max_cac_conservative"] <= 0:
        warnings.append("MAX CAC (conservative) <= 0 — acquisition spend likely unaffordable under these assumptions.")
    if sessions_per_therapist <= 0:
        warnings.append("SESSIONS PER THERAPIST <= 0 — check utilization or clients per therapist.")
    if therapists < 1:
        warnings.append("Less than 1 therapist — results not meaningful.")

# ---------------------------
# Layout: Summary KPIs + tabs
# ---------------------------
left_col, right_col = st.columns([1,2])

with left_col:
    st.header("Quick KPIs")
    st.metric("CLV (adjusted)", f"${clv['adjusted_clv']:.2f}")
    st.metric("Max CAC (conservative)", f"${cac_constraints['max_cac_conservative']:.2f}")
    st.metric("Contribution / session", f"${base_contrib_per_session:.2f}")
    st.metric("Sessions / therapist / month", f"{sessions_per_therapist:.1f}")

    st.markdown("**Staffing**")
    st.write(f"Therapists: {therapists}")
    st.write(f"Admins required (estimated): {needed_admins}")
    st.write("Switch to internal billing:" + (" Yes" if use_internal_billing else " No"))

    if warnings:
        st.warning(" · ".join(warnings))

with right_col:
    st.header("High level practice numbers (monthly)")
    total_revenue_month0 = float(np.sum(accrual_revenue[:1])) if len(accrual_revenue)>0 else 0.0
    total_revenue_month1 = float(np.sum(accrual_revenue[:months_projection]))
    total_profit_month1 = float(np.sum(accrual_net_income[:months_projection]))
    st.metric("Projected total revenue (next {} months)".format(months_projection), f"${total_revenue_month1:,.0f}")
    st.metric("Projected net income (accrual)".format(months_projection), f"${total_profit_month1:,.0f}")

# Tabs for detailed views
tabs = st.tabs(["CLV & Acquisition", "Monthly P&L & Cash Flow", "Therapist Level", "Scenarios & Sensitivity", "Audit & Tests"])

# ---------------------------
# Tab 1: CLV & Acquisition
# ---------------------------
with tabs[0]:
    st.subheader("CLV & Acquisition constraints")
    st.write("CLV (gross, discounted):", f"${clv['gross_clv']:.2f}")
    st.write("CLV (adjusted for seasonality & competition):", f"${clv['adjusted_clv']:.2f}")
    st.write("---")
    st.write("Acquisition constraints (three limits used to pick conservative max CAC):")
    st.write(pd.DataFrame({
        "Constraint": ["Profit margin limit", "Payback limit (first N months)", "ROI limit"],
        "Value ($)": [cac_constraints["profit_margin_limit"], cac_constraints["payback_limit"], cac_constraints["roi_limit"]]
    }))
    st.info(f"Conservative max CAC = ${cac_constraints['max_cac_conservative']:.2f}")

    st.markdown("### Marketing funnel & CAC")
    st.write(f"Estimated leads / month: {leads_per_month:.1f}")
    st.write(f"Estimated new clients / month (marketing): {new_clients_from_marketing:.2f}")
    effective_cac = safe_div(monthly_marketing_budget, new_clients_from_marketing) if new_clients_from_marketing>0 else float('inf')
    st.write(f"Effective CAC from current marketing: ${effective_cac:.2f}")
    st.markdown("### Monte Carlo (optional)")
    if run_monte_carlo:
        st.write("Running Monte Carlo — this may take a few seconds...")
        mc_samples = int(monte_carlo_samples)
        mc_clvs = []
        mc_max_cacs = []
        rng = np.random.default_rng(12345)
        for i in range(mc_samples):
            # sample churn and payout around base with small stdev
            samp_month1 = max(0.0, rng.normal(month1_churn, max(1.0, 0.1*month1_churn)))
            samp_month2 = max(0.0, rng.normal(month2_churn, max(1.0, 0.1*month2_churn)))
            samp_month3 = max(0.0, rng.normal(month3_churn, max(1.0, 0.1*month3_churn)))
            samp_ongoing = max(0.0, rng.normal(ongoing_churn, max(0.5, 0.05*ongoing_churn)))
            surv_samp = survival_curve(samp_month1, samp_month2, samp_month3, samp_ongoing, int(avg_lifespan_months))
            expected_samp = compute_expected_sessions_per_client(surv_samp, session_frequency_per_client * (1 - effective_no_show_rate/100.0))
            clv_s = compute_clv(expected_samp, base_contrib_per_session, monthly_discount_rate_pct/100.0, insurance_payment_delay_months,
                               seasonal_variation_pct, competition_factor)
            mc_clvs.append(clv_s["adjusted_clv"])
            mc_limits = compute_max_cac(clv_s["adjusted_clv"], target_profit_margin_pct, max_payback_months, expected_samp, base_contrib_per_session, required_roi_pct)
            mc_max_cacs.append(mc_limits["max_cac_conservative"])
        st.write(f"Monte Carlo samples: {mc_samples}")
        st.write(pd.Series(mc_max_cacs).describe())
        fig, ax = plt.subplots()
        ax.hist(mc_max_cacs, bins=30)
        ax.set_title("Distribution of max CAC (Monte Carlo)")
        ax.set_xlabel("Max CAC")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

# ---------------------------
# Tab 2: Monthly P&L & Cash Flow
# ---------------------------
with tabs[1]:
    st.subheader("Monthly P&L (accrual) & Cash Flow (collections schedule)")
    df = pd.DataFrame({
        "Month": [f"Month {i+1}" for i in range(months_projection)],
        "Active clients (est)": active_clients_by_month.round(1),
        "Sessions": sessions_by_month.round(1),
        "Billed (accrual)": billed_amounts_by_month.round(2),
        "Payroll cash": payroll_cash_by_month.round(2),
        "Fixed overhead cash": (monthly_fixed_overhead_cash).round(2),
        "Immediate CC fees": immediate_cc_fees_by_month[:months_projection].round(2),
        "Biller fees (collected month)": biller_fees_trim.round(2),
        "Accrual Expenses": accrual_expenses[:months_projection].round(2),
        "Accrual Net Income": accrual_net_income[:months_projection].round(2),
        "Collections (cash)": cash_collections.round(2),
        "Net cash (collections - cash outflows)": net_cash_by_month.round(2)
    })
    st.dataframe(df.style.format("{:.2f}"), height=360)

    st.download_button("Export monthly P&L CSV", df.to_csv(index=False).encode('utf-8'), file_name=f"monthly_pnl_{scenario_name}.csv")

    st.markdown("### Charts")
    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(1, months_projection+1), np.array(billed_amounts_by_month), label="Billed (accrual)")
    ax1.plot(np.arange(1, months_projection+1), np.array(cash_collections), label="Collections (cash)")
    ax1.plot(np.arange(1, months_projection+1), np.array(net_cash_by_month), label="Net cash")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("USD")
    ax1.set_title("Revenue vs Collections vs Net Cash")
    ax1.legend()
    st.pyplot(fig1)

# ---------------------------
# Tab 3: Therapist Level
# ---------------------------
with tabs[2]:
    st.subheader("Therapist-level economics")
    per_therapist = pd.DataFrame({
        "Metric": ["Sessions / month (est)", "Revenue / month (est)", "Payroll / month (est)", "Overhead / month", "Profit / month (est)"],
        "Value": [
            sessions_per_therapist,
            sessions_per_therapist * effective_revenue_per_session,
            sessions_per_therapist * therapist_pay_per_session * (1 + payroll_tax_and_benefits_pct / 100.0),
            (tech_cost_per_therapist + admin_alloc_per_therapist + other_overhead_per_therapist),
            sessions_per_therapist * effective_revenue_per_session - (sessions_per_therapist * therapist_pay_per_session * (1 + payroll_tax_and_benefits_pct / 100.0)) - (tech_cost_per_therapist + admin_alloc_per_therapist + other_overhead_per_therapist)
        ]
    })
    st.table(per_therapist.style.format({"Value": "${:,.2f}"}))

    st.markdown("### Hiring / ramping")
    st.write(f"Recruiting cost per hire: ${recruiting_cost_per_hire:,.0f}, onboarding cost per hire: ${onboarding_cost_per_hire:,.0f}")
    st.write("Estimated therapists needed before hiring admin:", admin_ratio)
    st.write(f"Estimated admins: {needed_admins}")

# ---------------------------
# Tab 4: Scenarios & Sensitivity
# ---------------------------
with tabs[3]:
    st.subheader("Scenario manager")
    st.write("You can save this scenario as a JSON snapshot and load/compare scenarios.")
    scenario_snapshot = {
        "scenario_name": scenario_name,
        "inputs": {
            "insurance_payout": insurance_payout,
            "self_pay_rate": self_pay_rate,
            "percent_self_pay": percent_self_pay,
            "copay_per_session": copay_per_session,
            "copay_collected_rate": copay_collected_rate,
            "therapists": therapists,
            "clients_per_therapist": clients_per_therapist,
            "session_frequency_per_client": session_frequency_per_client,
            "utilization_pct": utilization_pct,
            "therapist_pay_per_session": therapist_pay_per_session,
            "payroll_tax_and_benefits_pct": payroll_tax_and_benefits_pct,
            "tech_cost_per_therapist": tech_cost_per_therapist,
            "admin_alloc_per_therapist": admin_alloc_per_therapist,
            "other_overhead_per_therapist": other_overhead_per_therapist,
            "monthly_marketing_budget": monthly_marketing_budget,
            "cost_per_lead": cost_per_lead,
            "lead_to_intake_rate_pct": lead_to_intake_rate_pct,
            "intake_to_first_session_pct": intake_to_first_session_pct,
            "month1_churn": month1_churn,
            "month2_churn": month2_churn,
            "month3_churn": month3_churn,
            "ongoing_churn": ongoing_churn,
            "avg_lifespan_months": avg_lifespan_months,
            "target_profit_margin_pct": target_profit_margin_pct
        },
        "outputs": {
            "clv_adjusted": clv["adjusted_clv"],
            "max_cac_conservative": cac_constraints["max_cac_conservative"]
        }
    }
    st.download_button("Download scenario JSON", json.dumps(scenario_snapshot, indent=2).encode('utf-8'), file_name=f"scenario_{scenario_name}.json", mime="application/json")

    st.markdown("### Quick sensitivity (vary churn and payer mix)")
    churn_range = np.linspace(max(0.1, month1_churn - 10), month1_churn + 10, 9)
    sens_df = []
    for c in churn_range:
        surv_c = survival_curve(c, month2_churn, month3_churn, ongoing_churn, int(avg_lifespan_months))
        expected_c = compute_expected_sessions_per_client(surv_c, session_frequency_per_client * (1 - effective_no_show_rate/100.0))
        clv_c = compute_clv(expected_c, base_contrib_per_session, monthly_discount_rate_pct/100.0, insurance_payment_delay_months, seasonal_variation_pct, competition_factor)
        limits_c = compute_max_cac(clv_c["adjusted_clv"], target_profit_margin_pct, max_payback_months, expected_c, base_contrib_per_session, required_roi_pct)
        sens_df.append({"month1_churn": c, "clv": clv_c["adjusted_clv"], "max_cac": limits_c["max_cac_conservative"]})
    sens_df = pd.DataFrame(sens_df)
    st.line_chart(sens_df.set_index("month1_churn")["max_cac"])

# ---------------------------
# Tab 5: Audit & Tests
# ---------------------------
with tabs[4]:
    st.subheader("Formulas & Audit trail")
    st.markdown("""
    **Key formulas used**:
    - Effective revenue per session = weighted(self-pay revenue per session, insurance revenue per session including collected copay) * (1 - no-show)
    - Overhead per session = (tech + admin_alloc + other) / (clients_per_therapist * sessions_per_client)
    - Therapist total cost per session = therapist_pay_per_session * (1 + payroll_tax_and_benefits_pct)
    - Contribution per session = effective_revenue_per_session - therapist_total_cost_per_session - overhead_per_session
    - Survival curve: month1, month2, month3, then ongoing monthly churn applied multiplicatively
    - Expected sessions per client per month = survival_month * sessions_per_client (no-show adjusted)
    - CLV (PV) = sum_over_months(expected_sessions_month * contribution_per_session / (1 + monthly_discount)^ (month + payment_delay))
    - CAC constraints: profit margin limit, payback limit (sum of contributions in first N months), ROI limit (CLV / (1 + required ROI))
    """)
    st.markdown("**Checks performed**:")
    st.write("- Negative contribution per session => warning")
    st.write("- CLV <= 0 => warning")
    st.write("- CAC conservative <= 0 => warning")
    st.write("- AR distribution normalized to sum=1")

    st.subheader("Run basic unit tests")
    if st.button("Run built-in checks"):
        test_msgs = []
        # Check 1: overhead per session not infinite
        try:
            assert np.isfinite(overhead_per_session), "Overhead per session is infinite or NaN"
            test_msgs.append("OK: overhead per session finite")
            assert sessions_per_therapist > 0, "Sessions per therapist <= 0"
            test_msgs.append("OK: sessions per therapist > 0")
            # basic clv sanity
            assert clv["adjusted_clv"] > -1e6, "CLV suspiciously negative"
            test_msgs.append("OK: CLV numeric sanity check")
            test_msgs.append("All quick checks passed")
        except AssertionError as e:
            test_msgs.append("TEST FAILURE: " + str(e))
        st.write("\n".join(test_msgs))

st.markdown("---")
st.caption("Model built from your specification. Replace defaults with historical data to calibrate. Always validate results with an accountant.")
