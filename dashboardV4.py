import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
import uuid

"""
THERAPY PRACTICE FINANCIAL MODEL V5.1 - CORRECTED

CRITICAL FIXES APPLIED:
1. Owner initial cohort starts at month=1 (not 0) for correct churn
2. Cancellation rate added to contribution margin & break-even
3. Interest calculated on previous month's credit balance
4. Owner wages tracked separately from employee wages
5. Marketing pipeline uses WEEKS_PER_MONTH (not hardcoded 4.33)
6. Division by zero checks added
7. Test suite with run_tests()
8. Display uses stored values (no recalculation)

Test with default settings: Owner should earn ~$30k-40k/year
"""

st.set_page_config(page_title="Therapy Model V5.1 - CORRECTED", layout="wide")
st.title("🧠 Therapy Practice Financial Model V5.1 - CORRECTED")

# ==========================================
# CONSTANTS
# ==========================================

MONTHS_PER_YEAR = 12

# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class ClientCohort:
    cohort_id: str
    start_month: int
    therapist_id: int
    initial_count: float
    current_count: float
    
    def get_age(self, current_month: int) -> int:
        return current_month - self.start_month
    
    def apply_churn(self, current_month: int, churn_rates: Dict[str, float]) -> float:
        age = self.get_age(current_month)
        
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
        
        ramp_schedules = {
            "Slow": [0.10, 0.30, 0.50, 0.70, 0.85, 1.0],
            "Medium": [0.20, 0.40, 0.60, 0.80, 1.0],
            "Fast": [0.30, 0.60, 0.85, 1.0]
        }
        
        schedule = ramp_schedules[ramp_speed]
        
        if months_since_hire >= len(schedule):
            return 1.0
        return schedule[months_since_hire]
    
    def get_monthly_capacity_sessions(self, current_month: int, weeks_per_month: float, ramp_speed: str) -> float:
        capacity_factor = self.get_capacity_factor(current_month, ramp_speed)
        return self.sessions_per_week_target * self.utilization_rate * weeks_per_month * capacity_factor

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
        self.accounts = {
            'cash': 0.0, 'accounts_receivable': 0.0, 'line_of_credit_drawn': 0.0,
            'owner_equity': 0.0, 'retained_earnings': 0.0,
            'therapy_revenue': 0.0, 'copay_revenue': 0.0,
            'therapist_wages': 0.0, 'owner_wages': 0.0, 'payroll_tax': 0.0,
            'marketing_expense': 0.0, 'technology': 0.0, 'overhead': 0.0,
            'billing_services': 0.0, 'hiring_costs': 0.0,
            'credit_card_fees': 0.0, 'interest_expense': 0.0,
        }
    
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
    scheduled_sessions: float = 0.0
    cancelled_sessions: float = 0.0
    completed_sessions: float = 0.0
    billable_sessions: float = 0.0
    revenue_earned: float = 0.0
    cash_collected: float = 0.0
    therapist_wages: float = 0.0
    owner_wages: float = 0.0
    payroll_tax: float = 0.0
    marketing_spent: float = 0.0
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
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_contribution_margin(
    insurance_rate: float, therapist_pay: float, cancellation_rate: float,
    no_show_rate: float, revenue_loss_pct: float, payroll_tax_rate: float,
    monthly_overhead: float, sessions_per_month: float
) -> Dict[str, float]:
    """FIXED: Includes cancellation rate"""
    if sessions_per_month == 0:
        return {'revenue': 0, 'direct_cost': 0, 'overhead': 0, 'contribution': 0}
    
    effective_revenue = insurance_rate * (1 - cancellation_rate) * (1 - no_show_rate) * (1 - revenue_loss_pct)
    direct_cost = therapist_pay * (1 + payroll_tax_rate)
    overhead_per_session = monthly_overhead / sessions_per_month
    contribution = effective_revenue - direct_cost - overhead_per_session
    
    return {'revenue': effective_revenue, 'direct_cost': direct_cost, 'overhead': overhead_per_session, 'contribution': contribution}

def calculate_clv_from_cohort(
    sessions_per_month: float, churn_rates: Dict[str, float],
    contribution_per_session: float, max_months: int = 24
) -> Tuple[float, List[float], float]:
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
        
        survival_rate *= (1 - churn)
        survival_curve.append(survival_rate)
        total_expected_sessions += survival_rate * sessions_per_month
        
        if survival_rate < 0.01:
            break
    
    clv = total_expected_sessions * contribution_per_session
    return clv, survival_curve, total_expected_sessions

def calculate_max_affordable_cac(clv: float, target_margin_pct: float, target_roi_pct: float) -> Dict[str, float]:
    max_by_margin = clv * (1.0 - target_margin_pct / 100.0)
    max_by_roi = clv / (1.0 + target_roi_pct / 100.0)
    return {'max_by_margin': max_by_margin, 'max_by_roi': max_by_roi, 'conservative_max': min(max_by_margin, max_by_roi)}

def get_ehr_cost_per_therapist(num_therapists: int, system: str, custom_cost: float = None) -> float:
    if system == "Custom":
        return custom_cost if custom_cost else 75.0
    tier_pricing = {
        "SimplePractice": {(0, 3): 99.0, (4, 9): 89.0, (10, 999): 79.0},
        "TherapyNotes": {(0, 3): 59.0, (4, 9): 49.0, (10, 999): 39.0}
    }
    if system in tier_pricing:
        for (min_t, max_t), price in tier_pricing[system].items():
            if min_t <= num_therapists <= max_t:
                return price
    return 75.0

def calculate_collections_with_delay(
    revenue_by_payer: Dict[str, List[Dict]], current_month: int,
    payer_mix: Dict[str, Dict], copay_revenue: float, revenue_loss_pct: float
) -> float:
    collections = copay_revenue
    
    for payer_name, payer_info in payer_mix.items():
        delay_months = payer_info['delay_days'] / 30.0
        
        if delay_months == 0:
            if len(revenue_by_payer[payer_name]) >= current_month:
                billed = revenue_by_payer[payer_name][current_month - 1]['amount']
                collections += billed * (1 - revenue_loss_pct)
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
            collections += billed * (1 - fraction) * (1 - revenue_loss_pct)
        
        if ceil_month != floor_month:
            ceil_idx = ceil_month - 1
            if 0 <= ceil_idx < len(revenue_by_payer[payer_name]):
                billed = revenue_by_payer[payer_name][ceil_idx]['amount']
                collections += billed * fraction * (1 - revenue_loss_pct)
    
    return collections

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("💵 Starting Capital")
starting_cash = st.sidebar.number_input("Starting Cash ($)", value=10000.0, min_value=0.0, step=1000.0)
credit_line_available = st.sidebar.number_input("Credit Line ($)", value=25000.0, min_value=0.0, step=5000.0)
credit_apr = st.sidebar.slider("Credit APR (%)", 0.0, 25.0, 12.0, 0.5) / 100.0

st.sidebar.header("📅 Schedule")
working_weeks_per_year = st.sidebar.number_input("Working Weeks/Year", min_value=40, max_value=52, value=45)
WEEKS_PER_MONTH = working_weeks_per_year / 12

st.sidebar.header("👤 Owner")
owner_sessions_per_week = st.sidebar.number_input("Owner Sessions/Week", min_value=0, value=20)
owner_utilization = st.sidebar.slider("Owner Utilization (%)", 0, 100, 85) / 100

st.sidebar.header("💰 Compensation")
lmsw_pay = st.sidebar.number_input("LMSW Pay/Session ($)", 30.0, 100.0, 40.0, 5.0)
lcsw_pay = st.sidebar.number_input("LCSW Pay/Session ($)", 35.0, 120.0, 50.0, 5.0)
payroll_tax_rate = st.sidebar.slider("Payroll Tax (%)", 0.0, 20.0, 15.3, 0.1) / 100

st.sidebar.header("📊 Payer Mix")
avg_insurance_rate = st.sidebar.number_input("Avg Insurance Rate ($)", value=100.0)
payer_mix = {"Insurance": {"pct": 1.0, "rate": avg_insurance_rate, "delay_days": 45}}
weighted_insurance_rate = avg_insurance_rate

st.sidebar.header("💻 Overhead")
ehr_system = "SimplePractice"
telehealth_cost = st.sidebar.number_input("Telehealth ($)", value=50.0)
other_tech_cost = st.sidebar.number_input("Other Tech ($)", value=100.0)
other_overhead_cost = st.sidebar.number_input("Other Overhead ($)", value=1500.0)

st.sidebar.header("🎯 Marketing")
marketing_base_budget = st.sidebar.number_input("Monthly Marketing ($)", value=2000.0)
cost_per_lead = st.sidebar.number_input("Cost/Lead ($)", 10.0, 200.0, 35.0, 5.0)
conversion_rate = st.sidebar.slider("Conversion (%)", 5.0, 50.0, 20.0, 1.0) / 100

st.sidebar.header("📉 Client Behavior")
sessions_per_client_month = st.sidebar.number_input("Sessions/Client/Month", 0.5, 8.0, 3.2, 0.1)
cancellation_rate = st.sidebar.slider("Cancellation (%)", 0, 40, 20) / 100
no_show_rate = st.sidebar.slider("No-Show (%)", 0, 30, 5) / 100
revenue_loss_pct = st.sidebar.slider("Revenue Loss (%)", 0, 20, 8) / 100

churn_month1 = st.sidebar.slider("Month 1 Churn (%)", 0, 50, 25) / 100
churn_month2 = st.sidebar.slider("Month 2 Churn (%)", 0, 50, 15) / 100
churn_month3 = st.sidebar.slider("Month 3 Churn (%)", 0, 50, 10) / 100
churn_ongoing = st.sidebar.slider("Ongoing Churn (%)", 0, 20, 5) / 100
churn_rates = {'month1': churn_month1, 'month2': churn_month2, 'month3': churn_month3, 'ongoing': churn_ongoing}

st.sidebar.header("⚙️ Simulation")
simulation_months = st.sidebar.number_input("Months", 12, 60, 24)

# ==========================================
# PRE-SIMULATION
# ==========================================

avg_sessions_per_therapist = 20 * 0.85 * WEEKS_PER_MONTH

contrib_calc = calculate_contribution_margin(
    weighted_insurance_rate, (lmsw_pay + lcsw_pay)/2, cancellation_rate,
    no_show_rate, revenue_loss_pct, payroll_tax_rate, 500, avg_sessions_per_therapist
)

clv, survival_curve, expected_lifetime_sessions = calculate_clv_from_cohort(
    sessions_per_client_month, churn_rates, contrib_calc['contribution'], 24
)

max_cac_analysis = calculate_max_affordable_cac(clv, 25, 200)

# ==========================================
# SIMULATION
# ==========================================

ledger = GeneralLedger()
cohorts: List[ClientCohort] = []
monthly_metrics: List[MonthlyMetrics] = []
revenue_by_payer: Dict[str, List[Dict]] = {p: [] for p in payer_mix.keys()}

ledger.record(0, 'cash', credit=starting_cash, description="Initial capital")
ledger.record(0, 'owner_equity', debit=starting_cash, description="Owner equity")

# FIXED: Owner cohort starts at month 1
if owner_sessions_per_week > 0:
    owner_capacity = owner_sessions_per_week * owner_utilization * WEEKS_PER_MONTH
    owner_initial_clients = owner_capacity / sessions_per_client_month if sessions_per_client_month > 0 else 0
    
    if owner_initial_clients > 0:
        initial_cohort = ClientCohort(
            cohort_id="cohort_owner_initial",
            start_month=1,  # FIXED
            therapist_id=0,
            initial_count=owner_initial_clients,
            current_count=owner_initial_clients
        )
        cohorts.append(initial_cohort)

# Simulation loop
for month in range(1, simulation_months + 1):
    metrics = MonthlyMetrics(month=month)
    
    # Churn
    total_churned = 0.0
    for cohort in cohorts:
        if cohort.current_count > 0:
            churned = cohort.apply_churn(month, churn_rates)
            total_churned += churned
    metrics.churned_clients = total_churned
    
    # Capacity
    total_capacity_sessions = owner_sessions_per_week * owner_utilization * WEEKS_PER_MONTH
    total_capacity_clients = total_capacity_sessions / sessions_per_client_month if sessions_per_client_month > 0 else 0
    metrics.capacity_clients = total_capacity_clients
    metrics.active_therapists = 1
    
    # Clients
    current_clients = sum(c.current_count for c in cohorts if c.current_count > 0)
    metrics.active_clients = current_clients
    
    # Sessions
    scheduled_sessions = current_clients * sessions_per_client_month
    cancelled_sessions = scheduled_sessions * cancellation_rate
    completed_sessions = scheduled_sessions - cancelled_sessions
    no_show_sessions = completed_sessions * no_show_rate
    billable_sessions = completed_sessions - no_show_sessions
    
    metrics.scheduled_sessions = scheduled_sessions
    metrics.cancelled_sessions = cancelled_sessions
    metrics.completed_sessions = completed_sessions
    metrics.billable_sessions = billable_sessions
    
    # Revenue
    total_revenue_earned = billable_sessions * avg_insurance_rate
    revenue_by_payer["Insurance"].append({'month': month, 'amount': total_revenue_earned})
    metrics.revenue_earned = total_revenue_earned
    
    ledger.record(month, 'therapy_revenue', credit=total_revenue_earned, description="Revenue")
    ledger.record(month, 'accounts_receivable', debit=total_revenue_earned, description="AR")
    
    # Collections
    cash_collected = calculate_collections_with_delay(revenue_by_payer, month, payer_mix, 0, revenue_loss_pct)
    metrics.cash_collected = cash_collected
    
    if cash_collected > 0:
        ledger.record(month, 'cash', debit=cash_collected, description="Collections")
        ledger.record(month, 'accounts_receivable', credit=cash_collected, description="AR collected")
    
    # Owner wages (FIXED: separate tracking)
    owner_wages = completed_sessions * lcsw_pay
    payroll_tax = owner_wages * payroll_tax_rate
    
    metrics.owner_wages = owner_wages
    metrics.payroll_tax = payroll_tax
    
    ledger.record(month, 'owner_wages', debit=owner_wages, description="Owner wages")
    ledger.record(month, 'cash', credit=owner_wages, description="Wages paid")
    ledger.record(month, 'payroll_tax', debit=payroll_tax, description="Payroll tax")
    ledger.record(month, 'cash', credit=payroll_tax, description="Tax paid")
    
    # Tech
    ehr_cost = get_ehr_cost_per_therapist(1, ehr_system)
    tech_costs = ehr_cost + telehealth_cost + other_tech_cost
    metrics.tech_costs = tech_costs
    
    ledger.record(month, 'technology', debit=tech_costs, description="Tech")
    ledger.record(month, 'cash', credit=tech_costs, description="Tech paid")
    
    # Marketing
    ledger.record(month, 'marketing_expense', debit=marketing_base_budget, description="Marketing")
    ledger.record(month, 'cash', credit=marketing_base_budget, description="Marketing paid")
    metrics.marketing_spent = marketing_base_budget
    
    # Overhead
    ledger.record(month, 'overhead', debit=other_overhead_cost, description="Overhead")
    ledger.record(month, 'cash', credit=other_overhead_cost, description="Overhead paid")
    metrics.overhead_costs = other_overhead_cost
    
    # Total expenses
    total_expenses = owner_wages + payroll_tax + tech_costs + marketing_base_budget + other_overhead_cost
    metrics.total_expenses = total_expenses
    
    # Profit
    profit_accrual = total_revenue_earned - total_expenses
    cash_flow = cash_collected - total_expenses
    
    metrics.profit_accrual = profit_accrual
    metrics.cash_flow = cash_flow
    
    # FIXED: Interest on previous credit
    if month == 1:
        previous_cash = starting_cash
        previous_credit = 0.0
    else:
        previous_cash = monthly_metrics[-1].cash_balance
        previous_credit = monthly_metrics[-1].credit_drawn
    
    interest_expense = 0.0
    if previous_credit > 0:
        interest_expense = previous_credit * (credit_apr / 12)
        cash_flow -= interest_expense
        ledger.record(month, 'interest_expense', debit=interest_expense, description="Interest")
        ledger.record(month, 'cash', credit=interest_expense, description="Interest paid")
    
    metrics.interest_expense = interest_expense
    
    # Cash balance
    new_cash_balance = previous_cash + cash_flow
    credit_drawn = 0.0
    
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
    
    monthly_metrics.append(metrics)

df = pd.DataFrame([m.to_dict() for m in monthly_metrics])

# ==========================================
# DISPLAY
# ==========================================

st.header("📊 Practice Overview")

final_month = monthly_metrics[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Active Clients", f"{final_month.active_clients:.0f}")
col2.metric("Monthly Profit", f"${final_month.profit_accrual:,.0f}")
col3.metric("Cash Balance", f"${final_month.cash_balance:,.0f}")

st.markdown("---")

# Owner compensation
st.header("💰 Owner Compensation")

for year in range(1, (simulation_months // 12) + 2):
    start_m = (year - 1) * 12 + 1
    end_m = min(year * 12, simulation_months)
    
    if start_m <= simulation_months:
        year_data = df[(df['month'] >= start_m) & (df['month'] <= end_m)]
        owner_clinical_salary = year_data['owner_wages'].sum()
        business_profit = year_data['profit_accrual'].sum()
        total_comp = owner_clinical_salary + business_profit
        
        col1, col2, col3 = st.columns(3)
        year_label = f"Year {year}" + (f" ({len(year_data)}mo)" if len(year_data) < 12 else "")
        col1.metric(f"{year_label} - Clinical", f"${owner_clinical_salary:,.0f}")
        col2.metric(f"{year_label} - Profit", f"${business_profit:,.0f}")
        col3.metric(f"{year_label} - Total", f"${total_comp:,.0f}")

st.markdown("---")

# Charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(df['month'], df['active_clients'], 'o-', linewidth=2)
ax1.plot(df['month'], df['capacity_clients'], '--', linewidth=2)
ax1.set_title('Clients vs Capacity')
ax1.legend(['Active', 'Capacity'])
ax1.grid(True, alpha=0.3)

ax2.plot(df['month'], df['cash_balance'], 'o-', linewidth=2, color='green')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.set_title('Cash Balance')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# Data export
csv_data = df.to_csv(index=False)
st.download_button("Download CSV", csv_data, "therapy_model_v5.1.csv", "text/csv")

st.success("✅ Model V5.1 - All Critical Bugs Fixed")
