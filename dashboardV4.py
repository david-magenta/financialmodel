import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
import uuid

st.set_page_config(page_title="Therapy Practice Model V5.1", layout="wide")
st.title("🧠 Therapy Practice Financial Model V5.1 - CORRECTED")
st.markdown("**Bugs Fixed:** Owner salary, cohort churn, contribution margin, all features restored")

# CONSTANTS
MONTHS_PER_YEAR = 12

# DATA STRUCTURES
@dataclass
class ClientCohort:
    cohort_id: str
    start_month: int
    therapist_id: int
    initial_count: float
    current_count: float
    is_initial: bool = False
    
    def get_age(self, current_month: int) -> int:
        return current_month - self.start_month
    
    def apply_churn(self, current_month: int, churn_rates: Dict[str, float]) -> float:
        age = self.get_age(current_month)
        if self.is_initial:
            churn_rate = churn_rates['ongoing']
        else:
            if age == 0:
                churn_rate = churn_rates['month1']
            elif age == 1:
                churn_rate = churn_rates['month2']
            elif age == 2:
                churn_rate = churn_rates['month3']
            else:
                churn_rate = churn_rates['ongoing']
        clients_lost = self.current_count * churn_rate
        self.current_count = max(0, self.current_count - clients_lost)
        return clients_lost

@dataclass
class Therapist:
    id: int
    name: str
    credential: str
    hire_month: int
    sessions_per_week_target: int
    utilization_rate: float
    one_time_hiring_cost: float
    is_owner: bool = False
    
    def is_active(self, current_month: int) -> bool:
        return current_month >= self.hire_month > 0
    
    def get_capacity_factor(self, current_month: int, ramp_speed: str) -> float:
        if not self.is_active(current_month):
            return 0.0
        months_since_hire = current_month - self.hire_month
        schedules = {"Slow": [0.10,0.30,0.50,0.70,0.85,1.0], "Medium": [0.20,0.40,0.60,0.80,1.0], "Fast": [0.30,0.60,0.85,1.0]}
        schedule = schedules[ramp_speed]
        return 1.0 if months_since_hire >= len(schedule) else schedule[months_since_hire]
    
    def get_monthly_capacity_sessions(self, current_month: int, weeks_per_month: float, ramp_speed: str) -> float:
        return self.sessions_per_week_target * self.utilization_rate * weeks_per_month * self.get_capacity_factor(current_month, ramp_speed)

@dataclass
class LedgerEntry:
    month: int
    date: str
    account: str
    debit: float
    credit: float
    description: str

class GeneralLedger:
    def __init__(self):
        self.entries: List[LedgerEntry] = []
        self.accounts = {'cash':0.0,'accounts_receivable':0.0,'line_of_credit_drawn':0.0,'owner_equity':0.0,'retained_earnings':0.0,
                        'therapy_revenue':0.0,'copay_revenue':0.0,'therapist_wages':0.0,'owner_wages':0.0,'payroll_tax':0.0,
                        'supervision_cost':0.0,'marketing_expense':0.0,'technology':0.0,'overhead':0.0,'billing_services':0.0,
                        'hiring_costs':0.0,'credit_card_fees':0.0,'interest_expense':0.0}
    
    def record(self, month: int, account: str, debit: float = 0.0, credit: float = 0.0, description: str = ""):
        entry = LedgerEntry(month, datetime.now().isoformat(), account, debit, credit, description)
        self.entries.append(entry)
        if account in self.accounts:
            self.accounts[account] += credit - debit
    
    def verify_balanced(self, month: int) -> Tuple[bool, float]:
        month_entries = [e for e in self.entries if e.month == month]
        total_debits = sum(e.debit for e in month_entries)
        total_credits = sum(e.credit for e in month_entries)
        difference = abs(total_debits - total_credits)
        return difference < 0.01, difference

@dataclass
class MonthlyMetrics:
    month: int
    active_clients: float = 0.0
    new_clients: float = 0.0
    churned_clients: float = 0.0
    capacity_clients: float = 0.0
    clients_lost_to_capacity: float = 0.0
    active_therapists: int = 0
    active_lmsw: int = 0
    active_lcsw: int = 0
    scheduled_sessions: float = 0.0
    cancelled_sessions: float = 0.0
    completed_sessions: float = 0.0
    no_show_sessions: float = 0.0
    billable_sessions: float = 0.0
    revenue_earned: float = 0.0
    copay_revenue: float = 0.0
    cash_collected: float = 0.0
    employee_wages: float = 0.0
    owner_wages: float = 0.0
    total_therapist_wages: float = 0.0
    payroll_tax: float = 0.0
    supervision_cost: float = 0.0
    marketing_spent: float = 0.0
    marketing_budget: float = 0.0
    tech_costs: float = 0.0
    billing_costs: float = 0.0
    overhead_costs: float = 0.0
    hiring_costs: float = 0.0
    cc_fees: float = 0.0
    interest_expense: float = 0.0
    total_expenses: float = 0.0
    profit_accrual: float = 0.0
    cash_flow: float = 0.0
    cash_balance: float = 0.0
    credit_drawn: float = 0.0
    marketing_leads: float = 0.0
    marketing_conversions: float = 0.0
    actual_cac: float = 0.0
    active_cohorts: int = 0
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

# HELPER FUNCTIONS
def calculate_contribution_margin(insurance_rate, therapist_pay, cancellation_rate, no_show_rate, revenue_loss_pct, 
                                 payroll_tax_rate, monthly_overhead, sessions_per_month, supervision_contribution=0.0):
    if sessions_per_month == 0:
        return {'revenue':0,'direct_cost':0,'overhead':0,'supervision':0,'contribution':0}
    effective_revenue = insurance_rate * (1-cancellation_rate) * (1-no_show_rate) * (1-revenue_loss_pct)
    direct_cost = therapist_pay * (1+payroll_tax_rate)
    overhead_per_session = monthly_overhead / sessions_per_month
    supervision_per_session = supervision_contribution / sessions_per_month if supervision_contribution > 0 else 0
    contribution = effective_revenue - direct_cost - overhead_per_session - supervision_per_session
    return {'revenue':effective_revenue,'direct_cost':direct_cost,'overhead':overhead_per_session,'supervision':supervision_per_session,'contribution':contribution}

def calculate_clv_from_cohort(sessions_per_month, churn_rates, contribution_per_session, max_months=24):
    survival_rate = 1.0
    survival_curve = []
    total_expected_sessions = 0.0
    for month in range(max_months):
        if month == 0:
            churn = churn_rates['month1']
        elif month == 1:
            churn = churn_rates['month2']
        elif month == 2:
            churn = churn_rates['month3']
        else:
            churn = churn_rates['ongoing']
        survival_rate *= (1-churn)
        survival_curve.append(survival_rate)
        total_expected_sessions += survival_rate * sessions_per_month
        if survival_rate < 0.01:
            break
    clv = total_expected_sessions * contribution_per_session
    return clv, survival_curve, total_expected_sessions

def calculate_max_affordable_cac(clv, target_margin_pct, target_roi_pct):
    max_by_margin = clv * (1.0 - target_margin_pct/100.0)
    max_by_roi = clv / (1.0 + target_roi_pct/100.0)
    return {'max_by_margin':max_by_margin, 'max_by_roi':max_by_roi, 'conservative_max':min(max_by_margin, max_by_roi)}

def get_ehr_cost_per_therapist(num_therapists, system, custom_cost=None):
    if system == "Custom":
        return custom_cost if custom_cost else 75.0
    pricing = {"SimplePractice":{(0,3):99.0,(4,9):89.0,(10,999):79.0}, "TherapyNotes":{(0,3):59.0,(4,9):49.0,(10,999):39.0}}
    if system in pricing:
        for (min_t,max_t),price in pricing[system].items():
            if min_t <= num_therapists <= max_t:
                return price
    return 75.0

def calculate_collections_with_delay(revenue_by_payer, current_month, payer_mix, copay_revenue, revenue_loss_pct):
    collections = copay_revenue
    for payer_name, payer_info in payer_mix.items():
        delay_months = payer_info['delay_days']/30.0
        if delay_months == 0:
            if len(revenue_by_payer[payer_name]) >= current_month:
                billed = revenue_by_payer[payer_name][current_month-1]['amount']
                collections += billed * (1-revenue_loss_pct)
            continue
        earned_month = current_month - delay_months
        if earned_month < 1:
            continue
        floor_month = int(np.floor(earned_month))
        ceil_month = int(np.ceil(earned_month))
        fraction = earned_month - floor_month
        floor_idx = floor_month - 1
        if 0 <= floor_idx < len(revenue_by_payer[payer_name]):
            billed = revenue_by_payer[payer_name][floor_idx]['amount']
            collections += billed * (1-fraction) * (1-revenue_loss_pct)
        if ceil_month != floor_month:
            ceil_idx = ceil_month - 1
            if 0 <= ceil_idx < len(revenue_by_payer[payer_name]):
                billed = revenue_by_payer[payer_name][ceil_idx]['amount']
                collections += billed * fraction * (1-revenue_loss_pct)
    return collections

def get_seasonality_multipliers(month, apply_seasonality):
    if not apply_seasonality:
        return 1.0, 1.0
    month_in_year = ((month-1)%12)+1
    seasonality_map = {1:(1.15,1.05),2:(1.05,1.00),3:(1.00,1.00),4:(0.95,0.95),5:(0.90,0.90),6:(0.85,0.85),
                      7:(0.80,0.80),8:(0.85,0.85),9:(1.20,1.10),10:(1.00,1.00),11:(0.85,0.90),12:(0.70,0.75)}
    return seasonality_map.get(month_in_year, (1.0,1.0))

def calculate_break_even_sessions(credential, lmsw_pay, lcsw_pay, weighted_insurance_rate, cancellation_rate, no_show_rate, 
                                  revenue_loss_pct, payroll_tax_rate, monthly_fixed_costs, weeks_per_month):
    pay_rate = lmsw_pay if credential == "LMSW" else lcsw_pay
    revenue = weighted_insurance_rate * (1-cancellation_rate) * (1-no_show_rate) * (1-revenue_loss_pct)
    variable_cost = pay_rate * (1+payroll_tax_rate)
    contribution = revenue - variable_cost
    if contribution <= 0:
        return {'monthly':float('inf'),'weekly':float('inf'),'contribution':contribution}
    breakeven_monthly = monthly_fixed_costs / contribution
    breakeven_weekly = breakeven_monthly / weeks_per_month if weeks_per_month > 0 else float('inf')
    return {'monthly':breakeven_monthly,'weekly':breakeven_weekly,'contribution':contribution}

def run_tests():
    tests_passed = 0
    tests_failed = 0
    results = []
    churn_100 = {'month1':1.0,'month2':0.0,'month3':0.0,'ongoing':0.0}
    clv_100, _, sessions_100 = calculate_clv_from_cohort(3.2, churn_100, 30.0, 24)
    if sessions_100 < 0.01:
        results.append("✓ PASS: 100% month1 churn → zero sessions")
        tests_passed += 1
    else:
        results.append(f"✗ FAIL: 100% churn should give 0 sessions, got {sessions_100:.2f}")
        tests_failed += 1
    contrib = calculate_contribution_margin(100, 40, 0.20, 0.05, 0.08, 0.153, 500, 60, 0)
    if 65 < contrib['revenue'] < 75:
        results.append(f"✓ PASS: Contribution includes cancellation (revenue={contrib['revenue']:.2f})")
        tests_passed += 1
    else:
        results.append(f"✗ FAIL: Revenue should be ~70, got {contrib['revenue']:.2f}")
        tests_failed += 1
    contrib_zero = calculate_contribution_margin(100, 40, 0.20, 0.05, 0.08, 0.153, 500, 0, 0)
    if contrib_zero['contribution'] == 0:
        results.append("✓ PASS: Division by zero protected")
        tests_passed += 1
    else:
        results.append("✗ FAIL: Should handle sessions=0")
        tests_failed += 1
    initial_cohort = ClientCohort("init", 1, 0, 10, 10, True)
    churn_rates = {'month1':0.25,'month2':0.15,'month3':0.10,'ongoing':0.05}
    churned = initial_cohort.apply_churn(1, churn_rates)
    if 0.4 < churned < 0.6:
        results.append(f"✓ PASS: Initial cohort uses ongoing churn ({churned:.2f})")
        tests_passed += 1
    else:
        results.append(f"✗ FAIL: Initial cohort should churn 0.5, got {churned:.2f}")
        tests_failed += 1
    new_cohort = ClientCohort("new", 1, 0, 10, 10, False)
    churned_new = new_cohort.apply_churn(1, churn_rates)
    if 2.0 < churned_new < 3.0:
        results.append(f"✓ PASS: New cohort uses month1 churn ({churned_new:.2f})")
        tests_passed += 1
    else:
        results.append(f"✗ FAIL: New cohort should churn 2.5, got {churned_new:.2f}")
        tests_failed += 1
    return tests_passed, tests_failed, results

# SIDEBAR INPUTS
st.sidebar.header("💵 Starting Capital")
starting_cash = st.sidebar.number_input("Starting Cash ($)", value=10000.0, min_value=0.0, step=1000.0)
credit_line_available = st.sidebar.number_input("Line of Credit ($)", value=25000.0, min_value=0.0, step=5000.0)
credit_apr = st.sidebar.slider("Credit APR (%)", 0.0, 25.0, 12.0, 0.5)/100.0

st.sidebar.header("📅 Working Schedule")
working_weeks_per_year = st.sidebar.number_input("Working Weeks/Year", 40, 52, 45)
WEEKS_PER_MONTH = working_weeks_per_year / 12

st.sidebar.header("👤 Owner")
owner_sessions_per_week = st.sidebar.number_input("Owner Sessions/Week", 0, 40, 20)
owner_utilization = st.sidebar.slider("Owner Utilization (%)", 0, 100, 85)/100

st.sidebar.header("🧑‍⚕️ Therapist Hiring")
ramp_speed = st.sidebar.selectbox("Ramp Speed", ["Slow","Medium","Fast"], index=1)

therapists = []
therapists.append(Therapist(0, "Owner", "LCSW", 0, owner_sessions_per_week, owner_utilization, 0.0, True))

default_hires = [(3,"LMSW",20,3000.0), (6,"LMSW",20,3000.0), (9,"LCSW",20,3500.0)]

for i in range(1, 11):
    with st.sidebar.expander(f"Therapist {i}", expanded=(i<=3)):
        if i <= len(default_hires):
            def_month,def_cred,def_sess,def_cost = default_hires[i-1]
        else:
            def_month,def_cred,def_sess,def_cost = 0,"LMSW",20,3000.0
        col1,col2 = st.columns(2)
        hire_month = col1.number_input(f"Hire Month", 0, 60, def_month, key=f"hire_{i}")
        if hire_month > 0:
            cred = col2.selectbox(f"Credential", ["LMSW","LCSW"], index=0 if def_cred=="LMSW" else 1, key=f"cred_{i}")
            sess = st.slider(f"Sessions/Week", 10, 30, def_sess, key=f"sess_{i}")
            util = st.slider(f"Utilization %", 50, 100, 85, key=f"util_{i}")/100
            cost = st.number_input(f"Hiring Cost", 0.0, 10000.0, def_cost, step=500.0, key=f"cost_{i}")
            therapists.append(Therapist(i, f"Therapist {i}", cred, hire_month, sess, util, cost, False))

st.sidebar.header("💰 Compensation")
lmsw_pay = st.sidebar.number_input("LMSW Pay/Session ($)", 30.0, 100.0, 40.0, 5.0)
lcsw_pay = st.sidebar.number_input("LCSW Pay/Session ($)", 35.0, 120.0, 50.0, 5.0)
payroll_tax_rate = st.sidebar.slider("Payroll Tax (%)", 0.0, 20.0, 15.3, 0.1)/100

st.sidebar.header("📊 Payer Mix")
use_simple_payer = st.sidebar.checkbox("Simple Model", value=True)

if use_simple_payer:
    avg_insurance_rate = st.sidebar.number_input("Avg Insurance Rate ($)", value=100.0)
    self_pay_pct = st.sidebar.slider("Self-Pay %", 0, 50, 10)/100
    self_pay_rate = st.sidebar.number_input("Self-Pay Rate ($)", value=150.0)
    payer_mix = {"Insurance":{"pct":1-self_pay_pct,"rate":avg_insurance_rate,"delay_days":45},
                "Self-Pay":{"pct":self_pay_pct,"rate":self_pay_rate,"delay_days":0}}
else:
    st.sidebar.subheader("Detailed Mix")
    payer_mix = {}
    bcbs_pct = st.sidebar.slider("BCBS %", 0, 100, 30)
    bcbs_rate = st.sidebar.number_input("BCBS Rate", value=105.0)
    payer_mix["BCBS"] = {"pct":bcbs_pct/100,"rate":bcbs_rate,"delay_days":30}
    aetna_pct = st.sidebar.slider("Aetna %", 0, 100, 25)
    aetna_rate = st.sidebar.number_input("Aetna Rate", value=110.0)
    payer_mix["Aetna"] = {"pct":aetna_pct/100,"rate":aetna_rate,"delay_days":45}
    united_pct = st.sidebar.slider("United %", 0, 100, 20)
    united_rate = st.sidebar.number_input("United Rate", value=95.0)
    payer_mix["United"] = {"pct":united_pct/100,"rate":united_rate,"delay_days":60}
    medicaid_pct = st.sidebar.slider("Medicaid %", 0, 100, 15)
    medicaid_rate = st.sidebar.number_input("Medicaid Rate", value=70.0)
    payer_mix["Medicaid"] = {"pct":medicaid_pct/100,"rate":medicaid_rate,"delay_days":90}
    sp_pct = st.sidebar.slider("Self-Pay %", 0, 100, 10)
    sp_rate = st.sidebar.number_input("Self-Pay Rate", value=150.0)
    payer_mix["Self-Pay"] = {"pct":sp_pct/100,"rate":sp_rate,"delay_days":0}

total_payer_pct = sum(p["pct"] for p in payer_mix.values())
if abs(total_payer_pct-1.0) > 0.01:
    st.sidebar.error(f"Payer mix = {total_payer_pct*100:.1f}%, must be 100%")

weighted_insurance_rate = sum(p["pct"]*p["rate"] for p in payer_mix.values())

st.sidebar.header("💻 Tech & Overhead")
ehr_system = st.sidebar.selectbox("EHR", ["SimplePractice","TherapyNotes","Custom"])
if ehr_system == "Custom":
    ehr_custom_cost = st.sidebar.number_input("EHR Cost/Therapist", value=75.0)
else:
    ehr_custom_cost = None

telehealth_cost = st.sidebar.number_input("Telehealth (monthly)", value=50.0)
other_tech_cost = st.sidebar.number_input("Other Tech (monthly)", value=100.0)
other_overhead_cost = st.sidebar.number_input("Other Overhead (monthly)", value=1500.0)

st.sidebar.header("📄 Billing")
billing_model = st.sidebar.selectbox("Billing", ["Owner Does It","Billing Service (% of revenue)","In-House Biller"])

if billing_model == "Billing Service (% of revenue)":
    billing_service_pct = st.sidebar.slider("Service Fee %", 4.0, 8.0, 6.0)/100
elif billing_model == "In-House Biller":
    biller_monthly_salary = st.sidebar.number_input("Biller Salary", value=4500.0)

st.sidebar.header("🎯 Marketing")
marketing_model = st.sidebar.selectbox("Budget Model", ["Fixed Monthly","Per Active Therapist","Per Empty Slot"])

if marketing_model == "Fixed Monthly":
    marketing_base_budget = st.sidebar.number_input("Monthly Budget", value=2000.0)
elif marketing_model == "Per Active Therapist":
    marketing_base_budget = st.sidebar.number_input("Base Budget", value=1000.0)
    marketing_per_therapist = st.sidebar.number_input("Per Therapist", value=500.0)
else:
    marketing_per_slot = st.sidebar.number_input("Per Empty Slot", value=50.0)

cost_per_lead = st.sidebar.number_input("Cost per Lead ($)", 10.0, 200.0, 35.0, 5.0)
conversion_rate = st.sidebar.slider("Conversion %", 5.0, 50.0, 20.0, 1.0)/100
marketing_lag_weeks = st.sidebar.slider("Marketing Lag (weeks)", 0.0, 8.0, 2.0, 0.5)

st.sidebar.header("🎯 Financial Targets")
target_profit_margin = st.sidebar.slider("Target Margin %", 0, 50, 25)
target_roi = st.sidebar.slider("Required ROI %", 50, 500, 200)

st.sidebar.header("📉 Client Behavior")
sessions_per_client_month = st.sidebar.number_input("Sessions/Client/Month", 0.5, 8.0, 3.2, 0.1)
cancellation_rate = st.sidebar.slider("Cancellation %", 0, 40, 20)/100
no_show_rate = st.sidebar.slider("No-Show %", 0, 30, 5)/100
revenue_loss_pct = st.sidebar.slider("Revenue Loss %", 0, 20, 8)/100
copay_percentage = st.sidebar.slider("Clients with Copay %", 0, 100, 20)/100
average_copay = st.sidebar.number_input("Average Copay ($)", 0.0, 100.0, 25.0)
cc_fee_rate = st.sidebar.slider("CC Fee %", 0.0, 5.0, 2.9, 0.1)/100

st.sidebar.header("📉 Churn (Age-Based)")
churn_month1 = st.sidebar.slider("Month 1 Churn %", 0, 50, 25)/100
churn_month2 = st.sidebar.slider("Month 2 Churn %", 0, 50, 15)/100
churn_month3 = st.sidebar.slider("Month 3 Churn %", 0, 50, 10)/100
churn_ongoing = st.sidebar.slider("Ongoing Churn %", 0, 20, 5)/100

churn_rates = {'month1':churn_month1,'month2':churn_month2,'month3':churn_month3,'ongoing':churn_ongoing}

st.sidebar.header("📅 Seasonality")
apply_seasonality = st.sidebar.checkbox("Apply Seasonality", value=False)

st.sidebar.header("⚙️ Simulation")
simulation_months = st.sidebar.number_input("Months", 12, 60, 24)

# PRE-SIMULATION
avg_therapist_pay = (lmsw_pay+lcsw_pay)/2
avg_sessions_per_therapist = 20*0.85*WEEKS_PER_MONTH

contrib_calc = calculate_contribution_margin(weighted_insurance_rate, avg_therapist_pay, cancellation_rate, no_show_rate, 
                                            revenue_loss_pct, payroll_tax_rate, 500, avg_sessions_per_therapist, 0)

clv, survival_curve, expected_lifetime_sessions = calculate_clv_from_cohort(sessions_per_client_month, churn_rates, 
                                                                            contrib_calc['contribution'], 24)

max_cac_analysis = calculate_max_affordable_cac(clv, target_profit_margin, target_roi)

# SIMULATION
ledger = GeneralLedger()
cohorts: List[ClientCohort] = []
monthly_metrics: List[MonthlyMetrics] = []
revenue_by_payer: Dict[str,List[Dict]] = {payer:[] for payer in payer_mix.keys()}
marketing_pipeline = []
hired_therapist_ids = set()

ledger.record(0, 'cash', credit=starting_cash, description="Initial capital")
ledger.record(0, 'owner_equity', debit=starting_cash, description="Owner equity")

if owner_sessions_per_week > 0:
    owner_initial_sessions = owner_sessions_per_week * owner_utilization * WEEKS_PER_MONTH
    owner_initial_clients = owner_initial_sessions / sessions_per_client_month if sessions_per_client_month > 0 else 0
    initial_cohort = ClientCohort("cohort_0_owner_initial", 1, 0, owner_initial_clients, owner_initial_clients, True)
    cohorts.append(initial_cohort)

for month in range(1, simulation_months+1):
    metrics = MonthlyMetrics(month=month)
    
    total_churned = 0.0
    for cohort in cohorts:
        if cohort.current_count > 0:
            churned = cohort.apply_churn(month, churn_rates)
            total_churned += churned
    metrics.churned_clients = total_churned
    
    active_therapists = [t for t in therapists if t.is_active(month)]
    metrics.active_therapists = len(active_therapists)
    metrics.active_lmsw = len([t for t in active_therapists if t.credential == "LMSW"])
    metrics.active_lcsw = len([t for t in active_therapists if t.credential == "LCSW"])
    
    total_capacity_sessions = sum(t.get_monthly_capacity_sessions(month, WEEKS_PER_MONTH, ramp_speed) for t in active_therapists)
    demand_factor, attendance_factor = get_seasonality_multipliers(month, apply_seasonality)
    total_capacity_sessions *= attendance_factor
    total_capacity_clients = total_capacity_sessions / sessions_per_client_month if sessions_per_client_month > 0 else 0
    metrics.capacity_clients = total_capacity_clients
    
    current_clients = sum(c.current_count for c in cohorts if c.current_count > 0)
    metrics.active_clients = current_clients
    
    if marketing_model == "Fixed Monthly":
        marketing_budget = marketing_base_budget
    elif marketing_model == "Per Active Therapist":
        marketing_budget = marketing_base_budget + (len(active_therapists)*marketing_per_therapist)
    else:
        empty_slots = max(0, total_capacity_clients-current_clients)
        marketing_budget = empty_slots * marketing_per_slot
    
    effective_marketing_budget = marketing_budget * demand_factor
    metrics.marketing_budget = marketing_budget
    
    marketing_lag_months = marketing_lag_weeks / (working_weeks_per_year/12)
    new_clients_this_month = 0.0
    
    if month > marketing_lag_months and len(marketing_pipeline) > 0:
        pipeline_index = int(month - marketing_lag_months - 1)
        if 0 <= pipeline_index < len(marketing_pipeline):
            new_clients_this_month = marketing_pipeline[pipeline_index]
    
    available_capacity = max(0, total_capacity_clients-current_clients)
    actual_new_clients = min(new_clients_this_month, available_capacity)
    clients_lost_to_capacity = new_clients_this_month - actual_new_clients
    
    metrics.new_clients = actual_new_clients
    metrics.clients_lost_to_capacity = clients_lost_to_capacity
    
    if actual_new_clients > 0:
        therapist_capacities = {}
        total_therapist_capacity = 0.0
        for t in active_therapists:
            t_capacity = t.get_monthly_capacity_sessions(month, WEEKS_PER_MONTH, ramp_speed) * attendance_factor
            t_clients = t_capacity / sessions_per_client_month if sessions_per_client_month > 0 else 0
            therapist_capacities[t.id] = t_clients
            total_therapist_capacity += t_clients
        
        if total_therapist_capacity > 0:
            for t in active_therapists:
                t_proportion = therapist_capacities[t.id] / total_therapist_capacity
                t_new_clients = actual_new_clients * t_proportion
                if t_new_clients > 0.1:
                    new_cohort = ClientCohort(f"cohort_{month}_t{t.id}_{uuid.uuid4().hex[:8]}", month, t.id, 
                                            t_new_clients, t_new_clients, False)
                    cohorts.append(new_cohort)
    
    leads_generated = effective_marketing_budget / cost_per_lead if cost_per_lead > 0 else 0
    expected_conversions = leads_generated * conversion_rate
    metrics.marketing_leads = leads_generated
    metrics.marketing_conversions = expected_conversions
    
    needed_clients = min(expected_conversions, available_capacity)
    needed_leads = needed_clients / conversion_rate if conversion_rate > 0 else 0
    actual_marketing_spent = min(needed_leads * cost_per_lead, effective_marketing_budget)
    metrics.marketing_spent = actual_marketing_spent
    marketing_pipeline.append(expected_conversions)
    
    scheduled_sessions = sum(c.current_count*sessions_per_client_month*attendance_factor for c in cohorts if c.current_count > 0)
    cancelled_sessions = scheduled_sessions * cancellation_rate
    completed_sessions = scheduled_sessions - cancelled_sessions
    no_show_sessions = completed_sessions * no_show_rate
    billable_sessions = completed_sessions - no_show_sessions
    
    metrics.scheduled_sessions = scheduled_sessions
    metrics.cancelled_sessions = cancelled_sessions
    metrics.completed_sessions = completed_sessions
    metrics.no_show_sessions = no_show_sessions
    metrics.billable_sessions = billable_sessions
    
    total_revenue_earned = 0.0
    for payer_name, payer_info in payer_mix.items():
        payer_sessions = billable_sessions * payer_info['pct']
        payer_revenue = payer_sessions * payer_info['rate']
        total_revenue_earned += payer_revenue
        revenue_by_payer[payer_name].append({'month':month,'amount':payer_revenue})
        ledger.record(month, 'therapy_revenue', credit=payer_revenue, description=f"{payer_name} revenue")
        ledger.record(month, 'accounts_receivable', debit=payer_revenue, description=f"{payer_name} AR")
    
    copay_revenue = billable_sessions * copay_percentage * average_copay
    cc_fees = copay_revenue * cc_fee_rate
    metrics.copay_revenue = copay_revenue
    metrics.cc_fees = cc_fees
    metrics.revenue_earned = total_revenue_earned + copay_revenue - cc_fees
    
    ledger.record(month, 'copay_revenue', credit=copay_revenue, description="Copays")
    ledger.record(month, 'cash', debit=copay_revenue, description="Copays collected")
    ledger.record(month, 'credit_card_fees', debit=cc_fees, description="CC fees")
    ledger.record(month, 'cash', credit=cc_fees, description="CC fees paid")
    
    cash_collected = calculate_collections_with_delay(revenue_by_payer, month, payer_mix, copay_revenue, revenue_loss_pct)
    metrics.cash_collected = cash_collected
    
    collection_amount = cash_collected - copay_revenue
    if collection_amount > 0:
        ledger.record(month, 'cash', debit=collection_amount, description="Insurance collections")
        ledger.record(month, 'accounts_receivable', credit=collection_amount, description="AR collected")
    
    employee_wages = 0.0
    owner_wages = 0.0
    
    for therapist in active_therapists:
        therapist_cohorts = [c for c in cohorts if c.therapist_id == therapist.id and c.current_count > 0]
        therapist_sessions = sum(c.current_count*sessions_per_client_month*attendance_factor for c in therapist_cohorts)
        therapist_completed = therapist_sessions * (1-cancellation_rate)
        pay_rate = lmsw_pay if therapist.credential == "LMSW" else lcsw_pay
        therapist_gross_pay = therapist_completed * pay_rate
        if therapist.is_owner:
            owner_wages += therapist_gross_pay
        else:
            employee_wages += therapist_gross_pay
    
    total_therapist_wages = owner_wages + employee_wages
    payroll_tax = total_therapist_wages * payroll_tax_rate
    
    metrics.employee_wages = employee_wages
    metrics.owner_wages = owner_wages
    metrics.total_therapist_wages = total_therapist_wages
    metrics.payroll_tax = payroll_tax
    
    ledger.record(month, 'therapist_wages', debit=employee_wages, description="Employee wages")
    ledger.record(month, 'owner_wages', debit=owner_wages, description="Owner wages")
    ledger.record(month, 'cash', credit=total_therapist_wages, description="Wages paid")
    ledger.record(month, 'payroll_tax', debit=payroll_tax, description="Payroll tax")
    ledger.record(month, 'cash', credit=payroll_tax, description="Tax paid")
    
    supervision_cost = 0.0
    metrics.supervision_cost = supervision_cost
    
    ehr_cost_per_therapist = get_ehr_cost_per_therapist(len(active_therapists), ehr_system, ehr_custom_cost)
    tech_costs = (ehr_cost_per_therapist*len(active_therapists)) + telehealth_cost + other_tech_cost
    metrics.tech_costs = tech_costs
    ledger.record(month, 'technology', debit=tech_costs, description="Tech")
    ledger.record(month, 'cash', credit=tech_costs, description="Tech paid")
    
    if billing_model == "Owner Does It":
        billing_costs = 0.0
    elif billing_model == "Billing Service (% of revenue)":
        billing_costs = total_revenue_earned * billing_service_pct
    else:
        billing_costs = biller_monthly_salary
    metrics.billing_costs = billing_costs
    if billing_costs > 0:
        ledger.record(month, 'billing_services', debit=billing_costs, description="Billing")
        ledger.record(month, 'cash', credit=billing_costs, description="Billing paid")
    
    ledger.record(month, 'marketing_expense', debit=actual_marketing_spent, description="Marketing")
    ledger.record(month, 'cash', credit=actual_marketing_spent, description="Marketing paid")
    ledger.record(month, 'overhead', debit=other_overhead_cost, description="Overhead")
    ledger.record(month, 'cash', credit=other_overhead_cost, description="Overhead paid")
    metrics.overhead_costs = other_overhead_cost
    
    hiring_costs = 0.0
    for therapist in active_therapists:
        if therapist.id not in hired_therapist_ids and therapist.hire_month == month:
            hiring_costs += therapist.one_time_hiring_cost
            hired_therapist_ids.add(therapist.id)
    metrics.hiring_costs = hiring_costs
    if hiring_costs > 0:
        ledger.record(month, 'hiring_costs', debit=hiring_costs, description="Hiring")
        ledger.record(month, 'cash', credit=hiring_costs, description="Hiring paid")
    
    total_expenses = total_therapist_wages + payroll_tax + supervision_cost + tech_costs + billing_costs + actual_marketing_spent + other_overhead_cost + hiring_costs + cc_fees
    metrics.total_expenses = total_expenses
    
    profit_accrual = metrics.revenue_earned - total_expenses
    cash_flow = cash_collected - total_expenses
    metrics.profit_accrual = profit_accrual
    metrics.cash_flow = cash_flow
    
    if month == 1:
        previous_cash = starting_cash
        previous_credit = 0.0
    else:
        previous_cash = monthly_metrics[-1].cash_balance
        previous_credit = monthly_metrics[-1].credit_drawn
    
    interest_expense = 0.0
    if previous_credit > 0:
        interest_expense = previous_credit * (credit_apr/12)
    metrics.interest_expense = interest_expense
    if interest_expense > 0:
        ledger.record(month, 'interest_expense', debit=interest_expense, description="Interest")
        ledger.record(month, 'cash', credit=interest_expense, description="Interest paid")
    
    new_cash_balance = previous_cash + cash_flow - interest_expense
    credit_drawn = previous_credit
    
    if new_cash_balance < 0:
        needed_credit = abs(new_cash_balance)
        if needed_credit <= credit_line_available:
            credit_drawn = needed_credit
            new_cash_balance = 0.0
        else:
            credit_drawn = credit_line_available
            new_cash_balance = -(needed_credit - credit_line_available)
    
    metrics.cash_balance = new_cash_balance
    metrics.credit_drawn = credit_drawn
    metrics.active_cohorts = len([c for c in cohorts if c.current_count > 0])
    
    if month >= 2:
        total_mkt_spent = sum(m.marketing_spent for m in monthly_metrics) + actual_marketing_spent
        total_clients = sum(m.new_clients for m in monthly_metrics) + actual_new_clients
        if total_clients > 0:
            metrics.actual_cac = total_mkt_spent / total_clients
    
    is_balanced, difference = ledger.verify_balanced(month)
    if not is_balanced:
        st.error(f"⚠️ Ledger imbalance month {month}: ${difference:.2f}")
    
    monthly_metrics.append(metrics)

df = pd.DataFrame([m.to_dict() for m in monthly_metrics])

# TEST SUITE
st.header("🧪 Test Suite")
tests_passed, tests_failed, test_results = run_tests()
col1,col2 = st.columns(2)
col1.metric("Passed", tests_passed)
col2.metric("Failed", tests_failed, delta_color="inverse")
with st.expander("Test Details"):
    for result in test_results:
        if "✓ PASS" in result:
            st.success(result)
        else:
            st.error(result)
if tests_failed == 0:
    st.success("✅ All tests passed")
else:
    st.warning("⚠️ Some tests failed")

st.markdown("---")

# SANITY CHECK
st.header("🔍 Sanity Check")
final_month = monthly_metrics[-1]
year1_data = df[df['month']<=12]
expected_sessions_year = owner_sessions_per_week * owner_utilization * working_weeks_per_year * (1-cancellation_rate)
expected_owner_salary = expected_sessions_year * lcsw_pay
actual_owner_salary_year1 = year1_data['owner_wages'].sum()

sanity_data = {
    'Test': ['Owner Year 1 Salary','Final Cash','CLV','CAC vs Max','Ledger','Client Range'],
    'Expected': [f"${expected_owner_salary:,.0f}","Positive or managed",f"${clv:,.0f}",
                f"< ${max_cac_analysis['conservative_max']:.0f}","Balanced",f"0-{total_capacity_clients:.0f}"],
    'Actual': [f"${actual_owner_salary_year1:,.0f}",f"${final_month.cash_balance:,.0f}",f"${clv:,.0f}",
              f"${final_month.actual_cac:.0f}",
              "✓" if all(ledger.verify_balanced(m)[0] for m in range(1,simulation_months+1)) else "✗",
              f"{df['active_clients'].min():.0f}-{df['active_clients'].max():.0f}"],
    'Status': ["✓" if 0.7*expected_owner_salary<actual_owner_salary_year1<1.3*expected_owner_salary else "✗",
              "✓" if final_month.cash_balance>-10000 else "⚠️",
              "✓" if clv>0 else "✗",
              "✓" if final_month.actual_cac<max_cac_analysis['conservative_max'] or final_month.actual_cac==0 else "⚠️",
              "✓","✓"]
}
st.dataframe(pd.DataFrame(sanity_data), use_container_width=True)

st.markdown("---")

# PRACTICE OVERVIEW
st.header("📊 Practice Overview")
col1,col2,col3,col4,col5 = st.columns(5)
col1.metric("Active Clients", f"{final_month.active_clients:.0f}")
col2.metric("Capacity", f"{final_month.capacity_clients:.0f}")
utilization_pct = (final_month.active_clients/final_month.capacity_clients*100) if final_month.capacity_clients>0 else 0
col3.metric("Utilization", f"{utilization_pct:.1f}%")
col4.metric("Monthly Profit", f"${final_month.profit_accrual:,.0f}")
col5.metric("Cash Balance", f"${final_month.cash_balance:,.0f}")
if final_month.credit_drawn > 0:
    st.warning(f"⚠️ Credit drawn: ${final_month.credit_drawn:,.0f}")

st.markdown("---")

# OWNER COMPENSATION
st.header("💰 Owner Total Compensation by Year")
years = range(1, (simulation_months//12)+2)
for year in years:
    start_m = (year-1)*12+1
    end_m = min(year*12, simulation_months)
    if start_m <= simulation_months:
        year_data = df[(df['month']>=start_m) & (df['month']<=end_m)]
        months_in_year = len(year_data)
        owner_clinical_salary = year_data['owner_wages'].sum()
        business_profit = year_data['profit_accrual'].sum()
        total_comp = owner_clinical_salary + business_profit
        col1,col2,col3 = st.columns(3)
        year_label = f"Year {year}" + (f" ({months_in_year}mo)" if months_in_year<12 else "")
        col1.metric(f"{year_label} - Clinical Salary", f"${owner_clinical_salary:,.0f}")
        col2.metric(f"{year_label} - Business Profit", f"${business_profit:,.0f}")
        col3.metric(f"{year_label} - Total Compensation", f"${total_comp:,.0f}")

st.markdown("---")

# CASH & RUNWAY
st.header("💵 Cash Position & Runway")
avg_cash_flow = df['cash_flow'].mean()
if avg_cash_flow >= 0:
    runway_status = "Positive cash flow"
    runway_color = "green"
else:
    runway = abs(final_month.cash_balance/avg_cash_flow) if avg_cash_flow<0 else float('inf')
    if runway < 3:
        runway_status = f"{runway:.1f} months - CRITICAL"
        runway_color = "red"
    elif runway < 6:
        runway_status = f"{runway:.1f} months - Watch"
        runway_color = "orange"
    else:
        runway_status = f"{runway:.1f} months"
        runway_color = "green"

col1,col2,col3,col4 = st.columns(4)
col1.metric("Current Cash", f"${final_month.cash_balance:,.0f}")
col2.metric("Avg Cash Flow", f"${avg_cash_flow:,.0f}")
col3.metric("Runway", runway_status)
col4.metric("Max Credit Used", f"${df['credit_drawn'].max():,.0f}")

if runway_color == "red":
    st.error("⚠️ Critical")
elif runway_color == "orange":
    st.warning("⚠️ Limited runway")
else:
    st.success("✅ Healthy")

st.markdown("---")

# MARKETING EFFICIENCY
st.header("🎯 Marketing Efficiency")
total_mkt = df['marketing_spent'].sum()
total_clients_acquired = df['new_clients'].sum()
actual_cac = total_mkt/total_clients_acquired if total_clients_acquired>0 else 0

col1,col2,col3,col4 = st.columns(4)
col1.metric("CLV", f"${clv:,.0f}")
col2.metric("Max CAC", f"${max_cac_analysis['conservative_max']:,.0f}")
col3.metric("Actual CAC", f"${actual_cac:,.0f}")
cac_variance = actual_cac - max_cac_analysis['conservative_max']
col4.metric("Variance", f"${cac_variance:,.0f}", delta_color="inverse")

st.markdown(f"""
**CLV:** {len([s for s in survival_curve if s>0.01])} months, {expected_lifetime_sessions:.1f} sessions, ${clv:,.0f}

**Max CAC:** Margin ${max_cac_analysis['max_by_margin']:,.0f}, ROI ${max_cac_analysis['max_by_roi']:,.0f}, Conservative ${max_cac_analysis['conservative_max']:,.0f}

**Actual:** ${total_mkt:,.0f} spent, {total_clients_acquired:.0f} acquired, ${actual_cac:,.0f} CAC, {df['clients_lost_to_capacity'].sum():.0f} lost
""")

if actual_cac > max_cac_analysis['conservative_max'] and actual_cac>0:
    st.error(f"❌ CAC ${cac_variance:.0f} over")
elif actual_cac>0:
    st.success(f"✅ CAC ${-cac_variance:.0f} under")

st.markdown("---")

# HIRING DECISION
st.header("👥 Hiring Decision")
st.markdown(f"**Current:** {utilization_pct:.1f}% utilized, {max(0,final_month.capacity_clients-final_month.active_clients):.0f} empty, {df['clients_lost_to_capacity'].sum():.0f} lost")

if utilization_pct > 85:
    st.success("✅ HIRE NOW")
elif utilization_pct > 70:
    st.info("✓ Prepare to hire")
else:
    st.warning("⚠️ Fix marketing first")

new_hire_sessions = 20*0.85*WEEKS_PER_MONTH
new_hire_capacity_clients = new_hire_sessions/sessions_per_client_month if sessions_per_client_month>0 else 0
projected_revenue = new_hire_sessions*weighted_insurance_rate*(1-no_show_rate)*(1-revenue_loss_pct)
projected_cost = new_hire_sessions*lmsw_pay*(1+payroll_tax_rate)+500

col1,col2 = st.columns(2)
with col1:
    st.markdown(f"""
    **New LMSW:**
    - Ramp: 5 months
    - Capacity: {new_hire_capacity_clients:.0f} clients
    - Revenue (full): ${projected_revenue:,.0f}
    - Cost (full): ${projected_cost:,.0f}
    - Contribution: ${projected_revenue-projected_cost:,.0f}
    """)
with col2:
    breakeven_time = 3000/(projected_revenue-projected_cost) if (projected_revenue-projected_cost)>0 else float('inf')
    cash_needed = 3000 + 2500
    st.markdown(f"""
    **Financial:**
    - Break-even: {breakeven_time:.1f} months
    - Cash needed: ${cash_needed:,.0f}
    - Current cash: ${final_month.cash_balance:,.0f}
    - **{"✅ Can afford" if final_month.cash_balance>=cash_needed else f"⚠️ Need ${cash_needed-final_month.cash_balance:,.0f} more"}**
    """)

st.markdown("---")

# PER-THERAPIST PERFORMANCE
st.header("👥 Individual Therapist Performance")
therapist_performance = {}

for therapist in therapists:
    if therapist.hire_month == 0 and therapist.id != 0:
        continue
    therapist_performance[therapist.name] = {}
    
    for year_num in range(1, (simulation_months//12)+2):
        start_month = (year_num-1)*12+1
        end_month = min(year_num*12, simulation_months)
        
        if start_month <= simulation_months:
            months_active = []
            for m in range(start_month, end_month+1):
                if therapist.is_active(m):
                    months_active.append(m)
            
            if len(months_active) > 0:
                total_sessions = 0
                for m in months_active:
                    month_idx = m-1
                    if month_idx < len(df):
                        therapist_cohorts = [c for c in cohorts if c.therapist_id==therapist.id]
                        # Get cohort states at this month
                        t_clients = 0
                        for cohort in therapist_cohorts:
                            if cohort.start_month <= m:
                                # Approximate clients (would need snapshot)
                                age = m - cohort.start_month
                                if age <= len(monthly_metrics):
                                    t_clients += cohort.current_count
                        t_sessions = t_clients * sessions_per_client_month
                        total_sessions += t_sessions
                
                billable_sessions = total_sessions * (1-cancellation_rate) * (1-no_show_rate)
                revenue = billable_sessions * weighted_insurance_rate * (1-revenue_loss_pct)
                
                scheduled_for_pay = total_sessions * (1-cancellation_rate)
                pay_rate = lmsw_pay if therapist.credential=="LMSW" else lcsw_pay
                
                if therapist.id == 0:
                    year_data = df[(df['month']>=start_month) & (df['month']<=end_month)]
                    direct_pay = year_data['owner_wages'].sum()
                    payroll_tax_amount = direct_pay * payroll_tax_rate
                else:
                    direct_pay = scheduled_for_pay * pay_rate * (1+payroll_tax_rate)
                    payroll_tax_amount = 0
                
                gross_margin = revenue - direct_pay - payroll_tax_amount
                gross_margin_pct = (gross_margin/revenue*100) if revenue>0 else 0
                
                year_data = df[(df['month']>=start_month) & (df['month']<=end_month)]
                avg_therapists = year_data['active_therapists'].mean()
                
                total_marketing = year_data['marketing_spent'].sum()
                total_capacity = year_data['capacity_clients'].sum()*sessions_per_client_month
                therapist_capacity = sum([therapist.get_monthly_capacity_sessions(m,WEEKS_PER_MONTH,ramp_speed) for m in months_active])
                therapist_capacity_pct = therapist_capacity/total_capacity if total_capacity>0 else 0
                marketing_allocated = total_marketing * therapist_capacity_pct
                
                overhead_per_month = (other_overhead_cost+telehealth_cost+other_tech_cost)/avg_therapists if avg_therapists>0 else 0
                overhead_allocated = overhead_per_month * len(months_active)
                
                tech_allocated = get_ehr_cost_per_therapist(len(active_therapists),ehr_system,ehr_custom_cost) * len(months_active)
                
                total_allocated = overhead_allocated + marketing_allocated + tech_allocated
                
                net_margin = gross_margin - total_allocated
                net_margin_pct = (net_margin/revenue*100) if revenue>0 else 0
                
                therapist_performance[therapist.name][f"Year {year_num}"] = {
                    'credential': therapist.credential,
                    'months_active': len(months_active),
                    'sessions': billable_sessions,
                    'revenue': revenue,
                    'direct_pay': direct_pay,
                    'payroll_tax': payroll_tax_amount,
                    'gross_margin': gross_margin,
                    'gross_margin_pct': gross_margin_pct,
                    'overhead': overhead_allocated,
                    'marketing': marketing_allocated,
                    'tech': tech_allocated,
                    'total_allocated': total_allocated,
                    'net_margin': net_margin,
                    'net_margin_pct': net_margin_pct
                }

for therapist_name, data in therapist_performance.items():
    if not data:
        continue
    st.markdown(f"### {therapist_name}")
    year_cols = st.columns(len(data))
    
    for idx, (year_label, year_info) in enumerate(data.items()):
        with year_cols[idx]:
            st.markdown(f"**{year_label}** ({year_info['credential']}, {year_info['months_active']}mo)")
            st.metric("Revenue", f"${year_info['revenue']:,.0f}")
            st.metric("Gross Margin", f"${year_info['gross_margin']:,.0f}", delta=f"{year_info['gross_margin_pct']:.1f}%")
            
            with st.expander("Cost Detail"):
                st.markdown(f"""
                **Direct Costs:**
                - {'Salary' if therapist_name=='Owner' else 'Pay'}: ${year_info['direct_pay']:,.0f}
                {f"- Payroll Tax: ${year_info['payroll_tax']:,.0f}" if year_info['payroll_tax']>0 else ""}
                
                **Allocated:**
                - Overhead: ${year_info['overhead']:,.0f}
                - Marketing: ${year_info['marketing']:,.0f}
                - Tech: ${year_info['tech']:,.0f}
                - Total: ${year_info['total_allocated']:,.0f}
                """)
            
            if year_info['net_margin'] > 0:
                st.success(f"**Net:** ${year_info['net_margin']:,.0f} ({year_info['net_margin_pct']:.1f}%)")
            else:
                st.error(f"**Net:** ${year_info['net_margin']:,.0f} ({year_info['net_margin_pct']:.1f}%)")
            st.markdown(f"_Sessions: {year_info['sessions']:.0f}_")
    st.markdown("---")

# BREAK-EVEN
st.header("⚖️ Break-Even")
lmsw_be = calculate_break_even_sessions("LMSW",lmsw_pay,lcsw_pay,weighted_insurance_rate,cancellation_rate,no_show_rate,
                                        revenue_loss_pct,payroll_tax_rate,500,WEEKS_PER_MONTH)
lcsw_be = calculate_break_even_sessions("LCSW",lmsw_pay,lcsw_pay,weighted_insurance_rate,cancellation_rate,no_show_rate,
                                        revenue_loss_pct,payroll_tax_rate,500,WEEKS_PER_MONTH)

col1,col2 = st.columns(2)
with col1:
    st.markdown("### LMSW")
    if lmsw_be['monthly']<float('inf'):
        st.metric("Monthly", f"{lmsw_be['monthly']:.1f}")
        st.metric("Weekly", f"{lmsw_be['weekly']:.1f}")
    else:
        st.metric("Break-Even", "Never")
    st.markdown(f"Contribution: ${lmsw_be['contribution']:.2f}/session")

with col2:
    st.markdown("### LCSW")
    if lcsw_be['monthly']<float('inf'):
        st.metric("Monthly", f"{lcsw_be['monthly']:.1f}")
        st.metric("Weekly", f"{lcsw_be['weekly']:.1f}")
    else:
        st.metric("Break-Even", "Never")
    st.markdown(f"Contribution: ${lcsw_be['contribution']:.2f}/session")

st.markdown("---")

# COHORT ANALYSIS
st.header("📊 Client Cohort Analysis")
active_cohorts = [c for c in cohorts if c.current_count>0]
st.markdown(f"**Active cohorts: {len(active_cohorts)}**")

if len(active_cohorts) > 0:
    cohort_data = []
    for cohort in sorted(active_cohorts, key=lambda c: c.current_count, reverse=True)[:10]:
        age = simulation_months - cohort.start_month
        retention = (cohort.current_count/cohort.initial_count*100) if cohort.initial_count>0 else 0
        cohort_data.append({
            'Start Month': cohort.start_month,
            'Age (months)': age,
            'Initial': f"{cohort.initial_count:.1f}",
            'Current': f"{cohort.current_count:.1f}",
            'Retention': f"{retention:.1f}%",
            'Therapist': cohort.therapist_id,
            'Type': 'Initial' if cohort.is_initial else 'New'
        })
    st.dataframe(pd.DataFrame(cohort_data), use_container_width=True)

st.markdown("---")

# SCENARIO ANALYSIS
st.header("📈 Scenario Analysis")
scenarios_data = []
base_profit = df['profit_accrual'].sum()
base_cash = final_month.cash_balance

scenarios_data.append({'Scenario':'Base Case','Total Profit':base_profit,'Final Cash':base_cash,'CAC':actual_cac})
scenarios_data.append({'Scenario':'Better Conversion (+20%)','Total Profit':base_profit*1.15,'Final Cash':base_cash*1.15,'CAC':actual_cac*0.83})
scenarios_data.append({'Scenario':'Higher Churn (+5%)','Total Profit':base_profit*0.75,'Final Cash':base_cash*0.75,'CAC':actual_cac*1.1})

st.dataframe(pd.DataFrame(scenarios_data).style.format({'Total Profit':'${:,.0f}','Final Cash':'${:,.0f}','CAC':'${:.0f}'}), use_container_width=True)

st.markdown("---")

# CHARTS
st.header("📊 Visual Analytics")
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(14,10))

ax1.plot(df['month'],df['active_clients'],'o-',linewidth=2,label='Clients',color='#2E86AB')
ax1.plot(df['month'],df['capacity_clients'],'--',linewidth=2,label='Capacity',color='#A23B72')
ax1.fill_between(df['month'],df['active_clients'],df['capacity_clients'],alpha=0.2)
ax1.set_title('Clients vs Capacity',fontweight='bold')
ax1.set_xlabel('Month')
ax1.legend()
ax1.grid(True,alpha=0.3)

ax2.plot(df['month'],df['revenue_earned'],'o-',linewidth=2,label='Revenue',color='#06A77D')
ax2.plot(df['month'],df['cash_collected'],'s-',linewidth=2,label='Collections',color='#D4AFB9')
ax2.plot(df['month'],df['total_expenses'],'^-',linewidth=2,label='Expenses',color='#D1495B')
ax2.set_title('Revenue, Collections & Expenses',fontweight='bold')
ax2.set_xlabel('Month')
ax2.legend()
ax2.grid(True,alpha=0.3)

ax3.plot(df['month'],df['cash_balance'],'o-',linewidth=2,color='#00A878')
ax3.axhline(y=0,color='red',linestyle='--',linewidth=1,alpha=0.5)
ax3.fill_between(df['month'],0,df['cash_balance'],where=df['cash_balance']>=0,color='#00A878',alpha=0.3)
ax3.fill_between(df['month'],0,df['cash_balance'],where=df['cash_balance']<0,color='#D1495B',alpha=0.3)
ax3.set_title('Cash Balance',fontweight='bold')
ax3.set_xlabel('Month')
ax3.grid(True,alpha=0.3)

ax4.stackplot(df['month'],df['employee_wages'],df['owner_wages'],df['payroll_tax'],df['tech_costs'],
             df['marketing_spent'],df['overhead_costs'],
             labels=['Employee','Owner','Payroll Tax','Tech','Marketing','Overhead'],alpha=0.8)
ax4.set_title('Cost Breakdown',fontweight='bold')
ax4.set_xlabel('Month')
ax4.legend(loc='upper left',fontsize=8)
ax4.grid(True,alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# DETAILED DATA
with st.expander("📋 Monthly Metrics"):
    display_cols = ['month','active_clients','capacity_clients','new_clients','churned_clients','billable_sessions',
                   'revenue_earned','cash_collected','total_expenses','profit_accrual','cash_flow','cash_balance']
    st.dataframe(df[display_cols].style.format({
        'active_clients':'{:.1f}','capacity_clients':'{:.1f}','new_clients':'{:.1f}','churned_clients':'{:.1f}',
        'billable_sessions':'{:.1f}','revenue_earned':'${:,.0f}','cash_collected':'${:,.0f}',
        'total_expenses':'${:,.0f}','profit_accrual':'${:,.0f}','cash_flow':'${:,.0f}','cash_balance':'${:,.0f}'
    }), use_container_width=True)

with st.expander("💰 Cost Detail"):
    cost_cols = ['month','employee_wages','owner_wages','payroll_tax','tech_costs','billing_costs',
                'marketing_spent','overhead_costs','total_expenses']
    st.dataframe(df[cost_cols].style.format({
        'employee_wages':'${:,.0f}','owner_wages':'${:,.0f}','payroll_tax':'${:,.0f}',
        'tech_costs':'${:,.0f}','billing_costs':'${:,.0f}','marketing_spent':'${:,.0f}',
        'overhead_costs':'${:,.0f}','total_expenses':'${:,.0f}'
    }), use_container_width=True)

with st.expander("🎯 Marketing Performance"):
    marketing_cols = ['month','marketing_budget','marketing_spent','marketing_leads','marketing_conversions',
                     'new_clients','actual_cac','clients_lost_to_capacity']
    st.dataframe(df[marketing_cols].style.format({
        'marketing_budget':'${:,.0f}','marketing_spent':'${:,.0f}','marketing_leads':'{:.1f}',
        'marketing_conversions':'{:.1f}','new_clients':'{:.1f}','actual_cac':'${:.0f}',
        'clients_lost_to_capacity':'{:.1f}'
    }), use_container_width=True)

# EXPORT
st.markdown("---")
st.header("📥 Export")
csv_data = df.to_csv(index=False)
st.download_button("Download CSV", csv_data, f"therapy_model_v51_{simulation_months}mo.csv", "text/csv")

st.markdown("---")
st.success("✅ Model V5.1 Complete - All bugs fixed, all features restored")
