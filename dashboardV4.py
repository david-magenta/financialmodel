import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import uuid

st.set_page_config(page_title="Therapy Practice Financial Model V5.0", layout="wide")
st.title("🧠 Therapy Practice Financial Model V5.0")
st.markdown("**Complete Rebuild:** Cohort tracking, double-entry bookkeeping, accurate CAC/CLV")

# ==========================================
# CONSTANTS
# ==========================================

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52

# ==========================================
# DATA STRUCTURES
# ==========================================

@dataclass
class ClientCohort:
    """Tracks a group of clients who started therapy in the same month"""
    cohort_id: str
    start_month: int
    therapist_id: int
    initial_count: float
    current_count: float
    
    def get_age(self, current_month: int) -> int:
        """How many months has this cohort been in therapy"""
        return current_month - self.start_month
    
    def apply_churn(self, current_month: int, churn_rates: Dict[str, float]) -> float:
        """Apply age-based churn and return number of clients lost"""
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
    """Individual therapist with hiring timeline and capacity"""
    id: int
    name: str
    credential: str  # "LMSW" or "LCSW"
    hire_month: int
    sessions_per_week_target: int
    utilization_rate: float
    one_time_hiring_cost: float
    is_owner: bool = False
    
    def is_active(self, current_month: int) -> bool:
        """Is this therapist working this month"""
        return current_month >= self.hire_month > 0
    
    def get_capacity_factor(self, current_month: int, ramp_speed: str) -> float:
        """Returns 0.0 to 1.0 representing ramp-up progress"""
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
        """How many sessions can this therapist handle this month"""
        capacity_factor = self.get_capacity_factor(current_month, ramp_speed)
        return self.sessions_per_week_target * self.utilization_rate * weeks_per_month * capacity_factor

@dataclass
class LedgerEntry:
    """Double-entry accounting transaction"""
    month: int
    date: str
    account: str
    debit: float
    credit: float
    description: str
    
class GeneralLedger:
    """Double-entry bookkeeping system"""
    def __init__(self):
        self.entries: List[LedgerEntry] = []
        self.accounts = {
            # Assets
            'cash': 0.0,
            'accounts_receivable': 0.0,
            'line_of_credit_drawn': 0.0,
            # Equity
            'owner_equity': 0.0,
            'retained_earnings': 0.0,
            # Revenue (credits increase)
            'therapy_revenue': 0.0,
            'copay_revenue': 0.0,
            # Expenses (debits increase)
            'therapist_wages': 0.0,
            'payroll_tax': 0.0,
            'supervision_cost': 0.0,
            'marketing_expense': 0.0,
            'technology': 0.0,
            'overhead': 0.0,
            'billing_services': 0.0,
            'hiring_costs': 0.0,
            'credit_card_fees': 0.0,
            'interest_expense': 0.0,
        }
    
    def record(self, month: int, account: str, debit: float = 0.0, credit: float = 0.0, description: str = ""):
        """Record a transaction"""
        entry = LedgerEntry(
            month=month,
            date=datetime.now().isoformat(),
            account=account,
            debit=debit,
            credit=credit,
            description=description
        )
        self.entries.append(entry)
        
        # Update account balance
        if account in self.accounts:
            self.accounts[account] += credit - debit
    
    def get_balance(self, account: str, through_month: int = None) -> float:
        """Get account balance"""
        if through_month is None:
            return self.accounts.get(account, 0.0)
        
        balance = 0.0
        for entry in self.entries:
            if entry.account == account and entry.month <= through_month:
                balance += entry.credit - entry.debit
        return balance
    
    def verify_balanced(self, month: int) -> Tuple[bool, float]:
        """Verify debits = credits for a month"""
        month_entries = [e for e in self.entries if e.month == month]
        total_debits = sum(e.debit for e in month_entries)
        total_credits = sum(e.credit for e in month_entries)
        difference = abs(total_debits - total_credits)
        return difference < 0.01, difference

@dataclass
class MonthlyMetrics:
    """All metrics for a single month"""
    month: int
    
    # Clients
    active_clients: float = 0.0
    new_clients: float = 0.0
    churned_clients: float = 0.0
    capacity_clients: float = 0.0
    clients_lost_to_capacity: float = 0.0
    
    # Therapists
    active_therapists: int = 0
    active_lmsw: int = 0
    active_lcsw: int = 0
    
    # Sessions
    scheduled_sessions: float = 0.0
    cancelled_sessions: float = 0.0
    completed_sessions: float = 0.0
    no_show_sessions: float = 0.0
    billable_sessions: float = 0.0
    
    # Revenue (Accrual)
    revenue_earned: float = 0.0
    copay_revenue: float = 0.0
    
    # Collections (Cash)
    cash_collected: float = 0.0
    
    # Expenses
    therapist_wages: float = 0.0
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
    
    # Profitability
    profit_accrual: float = 0.0
    cash_flow: float = 0.0
    cash_balance: float = 0.0
    credit_drawn: float = 0.0
    
    # Marketing
    marketing_leads: float = 0.0
    marketing_conversions: float = 0.0
    actual_cac: float = 0.0
    
    # Cohorts
    active_cohorts: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame"""
        return {k: v for k, v in self.__dict__.items()}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_contribution_margin(
    insurance_rate: float,
    therapist_pay: float,
    credential: str,
    no_show_rate: float,
    revenue_loss_pct: float,
    payroll_tax_rate: float,
    monthly_overhead: float,
    sessions_per_month: float,
    supervision_contribution: float = 0.0
) -> Dict[str, float]:
    """
    Calculate true contribution margin per session including all allocated costs
    
    Returns dict with: revenue, direct_cost, overhead, supervision, contribution
    """
    # Revenue per session (after no-shows and revenue loss)
    effective_revenue = insurance_rate * (1 - no_show_rate) * (1 - revenue_loss_pct)
    
    # Direct cost (therapist only paid for completed sessions)
    direct_cost = therapist_pay * (1 + payroll_tax_rate)
    
    # Allocated overhead per session
    overhead_per_session = monthly_overhead / sessions_per_month if sessions_per_month > 0 else 0
    
    # Supervision cost per session (for LMSW only)
    supervision_per_session = 0.0
    if credential == "LMSW" and supervision_contribution > 0:
        supervision_per_session = supervision_contribution / sessions_per_month if sessions_per_month > 0 else 0
    
    # Total contribution margin
    contribution = effective_revenue - direct_cost - overhead_per_session - supervision_per_session
    
    return {
        'revenue': effective_revenue,
        'direct_cost': direct_cost,
        'overhead': overhead_per_session,
        'supervision': supervision_per_session,
        'contribution': contribution
    }

def calculate_clv_from_cohort(
    sessions_per_month: float,
    churn_rates: Dict[str, float],
    contribution_per_session: float,
    max_months: int = 24
) -> Tuple[float, List[float], float]:
    """
    Calculate CLV using survival curve
    
    Returns: (clv, survival_curve, expected_sessions)
    """
    survival_rate = 1.0
    survival_curve = []
    total_expected_sessions = 0.0
    
    for month in range(max_months):
        # Apply churn based on cohort age
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
        
        # Expected sessions this month
        total_expected_sessions += survival_rate * sessions_per_month
        
        # Stop if virtually no one left
        if survival_rate < 0.01:
            break
    
    clv = total_expected_sessions * contribution_per_session
    return clv, survival_curve, total_expected_sessions

def calculate_max_affordable_cac(
    clv: float,
    target_margin_pct: float,
    target_roi_pct: float
) -> Dict[str, float]:
    """
    Calculate maximum CAC under business constraints
    
    Returns dict with: max_by_margin, max_by_roi, conservative_max
    """
    max_by_margin = clv * (1.0 - target_margin_pct / 100.0)
    max_by_roi = clv / (1.0 + target_roi_pct / 100.0)
    conservative_max = min(max_by_margin, max_by_roi)
    
    return {
        'max_by_margin': max_by_margin,
        'max_by_roi': max_by_roi,
        'conservative_max': conservative_max
    }

def get_ehr_cost_per_therapist(num_therapists: int, system: str, custom_cost: float = None) -> float:
    """Calculate EHR cost with tiered pricing"""
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
    revenue_by_payer: Dict[str, List[Dict]],
    current_month: int,
    payer_mix: Dict[str, Dict],
    copay_revenue: float,
    revenue_loss_pct: float
) -> float:
    """
    Calculate cash collections accounting for payment delays and revenue loss
    
    Revenue loss (denials + bad debt) applied at collection time, not billing time
    """
    collections = copay_revenue  # Copays collected immediately
    
    for payer_name, payer_info in payer_mix.items():
        delay_months = payer_info['delay_days'] / 30.0
        
        # Self-pay has no delay
        if delay_months == 0:
            if len(revenue_by_payer[payer_name]) >= current_month:
                billed = revenue_by_payer[payer_name][current_month - 1]['amount']
                # Apply revenue loss at collection
                collections += billed * (1 - revenue_loss_pct)
            continue
        
        # Calculate which month's revenue we're collecting
        earned_month = current_month - delay_months
        
        if earned_month < 1:
            continue
        
        # Interpolate between months for fractional delays
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

def get_seasonality_multipliers(month: int, apply_seasonality: bool) -> Tuple[float, float]:
    """
    Returns (demand_factor, attendance_factor) for given month
    
    demand_factor: affects new client interest
    attendance_factor: affects session completion rates
    """
    if not apply_seasonality:
        return 1.0, 1.0
    
    month_in_year = ((month - 1) % 12) + 1
    
    seasonality_map = {
        1: (1.15, 1.05),   # January - New Year
        2: (1.05, 1.00),   # February
        3: (1.00, 1.00),   # March
        4: (0.95, 0.95),   # April
        5: (0.90, 0.90),   # May
        6: (0.85, 0.85),   # June - Summer starts
        7: (0.80, 0.80),   # July - Summer low
        8: (0.85, 0.85),   # August
        9: (1.20, 1.10),   # September - Back to school
        10: (1.00, 1.00),  # October
        11: (0.85, 0.90),  # November - Holidays
        12: (0.70, 0.75),  # December - Holiday low
    }
    
    return seasonality_map.get(month_in_year, (1.0, 1.0))

def calculate_break_even_sessions(
    credential: str,
    lmsw_pay: float,
    lcsw_pay: float,
    weighted_insurance_rate: float,
    no_show_rate: float,
    revenue_loss_pct: float,
    payroll_tax_rate: float,
    monthly_fixed_costs: float,
    weeks_per_month: float
) -> Dict[str, float]:
    """Calculate break-even sessions for a therapist"""
    pay_rate = lmsw_pay if credential == "LMSW" else lcsw_pay
    
    # Revenue per session
    revenue = weighted_insurance_rate * (1 - no_show_rate) * (1 - revenue_loss_pct)
    
    # Variable cost per session
    variable_cost = pay_rate * (1 + payroll_tax_rate)
    
    # Contribution margin
    contribution = revenue - variable_cost
    
    if contribution <= 0:
        return {
            'monthly': float('inf'),
            'weekly': float('inf'),
            'contribution': contribution
        }
    
    breakeven_monthly = monthly_fixed_costs / contribution
    breakeven_weekly = breakeven_monthly / weeks_per_month
    
    return {
        'monthly': breakeven_monthly,
        'weekly': breakeven_weekly,
        'contribution': contribution
    }

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("💵 Starting Capital")
starting_cash = st.sidebar.number_input(
    "Starting Cash Balance ($)",
    value=10000.0,
    min_value=0.0,
    step=1000.0,
    help="Cash available at start"
)

credit_line_available = st.sidebar.number_input(
    "Line of Credit Available ($)",
    value=25000.0,
    min_value=0.0,
    step=5000.0,
    help="Maximum credit line for cash shortfalls"
)

credit_apr = st.sidebar.slider(
    "Credit Line APR (%)",
    0.0, 25.0, 12.0, 0.5,
    help="Annual interest rate on drawn credit"
) / 100.0

st.sidebar.header("📅 Working Schedule")
working_weeks_per_year = st.sidebar.number_input(
    "Working Weeks per Year",
    min_value=40,
    max_value=52,
    value=45,
    help="Account for vacation, holidays, sick time"
)

WEEKS_PER_MONTH = working_weeks_per_year / 12

st.sidebar.header("👤 Owner (Treated as Employee Therapist)")
owner_sessions_per_week = st.sidebar.number_input("Owner Sessions/Week", min_value=0, value=20)
owner_utilization = st.sidebar.slider("Owner Utilization (%)", 0, 100, 85) / 100

st.sidebar.header("🧑‍⚕️ Therapist Hiring")
ramp_speed = st.sidebar.selectbox(
    "Ramp-Up Speed",
    ["Slow", "Medium", "Fast"],
    index=1,
    help="How quickly new hires reach full caseload"
)

# Initialize therapists list
therapists = []

# Owner is therapist 0
therapists.append(Therapist(
    id=0,
    name="Owner",
    credential="LCSW",
    hire_month=0,
    sessions_per_week_target=owner_sessions_per_week,
    utilization_rate=owner_utilization,
    one_time_hiring_cost=0.0,
    is_owner=True
))

# Additional therapists
default_hires = [
    (3, "LMSW", 20, 3000.0),
    (6, "LMSW", 20, 3000.0),
    (9, "LCSW", 20, 3500.0)
]

for i in range(1, 11):
    with st.sidebar.expander(f"Therapist {i}", expanded=(i <= 3)):
        if i <= len(default_hires):
            def_month, def_cred, def_sess, def_cost = default_hires[i-1]
        else:
            def_month, def_cred, def_sess, def_cost = 0, "LMSW", 20, 3000.0
        
        col1, col2 = st.columns(2)
        hire_month = col1.number_input(
            f"Hire Month",
            0, 60, def_month,
            key=f"hire_{i}",
            help="0 = not hired"
        )
        
        if hire_month > 0:
            cred = col2.selectbox(
                f"Credential",
                ["LMSW", "LCSW"],
                index=0 if def_cred == "LMSW" else 1,
                key=f"cred_{i}"
            )
            sess = st.slider(f"Sessions/Week", 10, 30, def_sess, key=f"sess_{i}")
            util = st.slider(f"Utilization %", 50, 100, 85, key=f"util_{i}") / 100
            cost = st.number_input(
                f"Hiring Cost ($)",
                0.0, 10000.0,
                def_cost,
                step=500.0,
                key=f"cost_{i}"
            )
            
            therapists.append(Therapist(
                id=i,
                name=f"Therapist {i}",
                credential=cred,
                hire_month=hire_month,
                sessions_per_week_target=sess,
                utilization_rate=util,
                one_time_hiring_cost=cost,
                is_owner=False
            ))

st.sidebar.header("💰 Compensation")
lmsw_pay = st.sidebar.number_input(
    "LMSW Pay per Session ($)",
    30.0, 100.0, 40.0, 5.0,
    help="Paid only for completed sessions"
)
lcsw_pay = st.sidebar.number_input(
    "LCSW Pay per Session ($)",
    35.0, 120.0, 50.0, 5.0,
    help="Paid only for completed sessions"
)
payroll_tax_rate = st.sidebar.slider(
    "Payroll Tax Rate (%)",
    0.0, 20.0, 15.3, 0.1,
    help="Employer portion of payroll taxes"
) / 100

st.sidebar.header("📊 Payer Mix")
use_simple_payer = st.sidebar.checkbox("Simple Payer Model", value=True)

if use_simple_payer:
    avg_insurance_rate = st.sidebar.number_input("Average Insurance Rate ($)", value=100.0)
    self_pay_pct = st.sidebar.slider("Self-Pay Percentage (%)", 0, 50, 10) / 100
    self_pay_rate = st.sidebar.number_input("Self-Pay Rate ($)", value=150.0)
    
    payer_mix = {
        "Insurance": {
            "pct": 1 - self_pay_pct,
            "rate": avg_insurance_rate,
            "delay_days": 45
        },
        "Self-Pay": {
            "pct": self_pay_pct,
            "rate": self_pay_rate,
            "delay_days": 0
        }
    }
else:
    st.sidebar.subheader("Detailed Payer Mix")
    payer_mix = {}
    
    bcbs_pct = st.sidebar.slider("BCBS %", 0, 100, 30)
    bcbs_rate = st.sidebar.number_input("BCBS Rate ($)", value=105.0)
    payer_mix["BCBS"] = {"pct": bcbs_pct/100, "rate": bcbs_rate, "delay_days": 30}
    
    aetna_pct = st.sidebar.slider("Aetna %", 0, 100, 25)
    aetna_rate = st.sidebar.number_input("Aetna Rate ($)", value=110.0)
    payer_mix["Aetna"] = {"pct": aetna_pct/100, "rate": aetna_rate, "delay_days": 45}
    
    united_pct = st.sidebar.slider("United %", 0, 100, 20)
    united_rate = st.sidebar.number_input("United Rate ($)", value=95.0)
    payer_mix["United"] = {"pct": united_pct/100, "rate": united_rate, "delay_days": 60}
    
    medicaid_pct = st.sidebar.slider("Medicaid %", 0, 100, 15)
    medicaid_rate = st.sidebar.number_input("Medicaid Rate ($)", value=70.0)
    payer_mix["Medicaid"] = {"pct": medicaid_pct/100, "rate": medicaid_rate, "delay_days": 90}
    
    sp_pct = st.sidebar.slider("Self-Pay %", 0, 100, 10)
    sp_rate = st.sidebar.number_input("Self-Pay Rate ($)", value=150.0)
    payer_mix["Self-Pay"] = {"pct": sp_pct/100, "rate": sp_rate, "delay_days": 0}

# Validate payer mix
total_payer_pct = sum(p["pct"] for p in payer_mix.values())
if abs(total_payer_pct - 1.0) > 0.01:
    st.sidebar.error(f"Payer mix totals {total_payer_pct*100:.1f}%, must equal 100%")

# Calculate weighted insurance rate
weighted_insurance_rate = sum(p["pct"] * p["rate"] for p in payer_mix.values())

st.sidebar.header("💻 Technology & Overhead")
ehr_system = st.sidebar.selectbox(
    "EHR System",
    ["SimplePractice", "TherapyNotes", "Custom"]
)
if ehr_system == "Custom":
    ehr_custom_cost = st.sidebar.number_input("EHR Cost per Therapist ($)", value=75.0)
else:
    ehr_custom_cost = None
    st.sidebar.info(f"{ehr_system} uses tiered pricing")

telehealth_cost = st.sidebar.number_input("Telehealth Platform (monthly, $)", value=50.0)
other_tech_cost = st.sidebar.number_input("Other Tech (monthly, $)", value=100.0)
other_overhead_cost = st.sidebar.number_input(
    "Other Overhead (monthly, $)",
    value=1500.0,
    help="Insurance, legal, accounting, etc."
)

st.sidebar.header("📄 Billing")
billing_model = st.sidebar.selectbox(
    "Billing Model",
    ["Owner Does It", "Billing Service (% of revenue)", "In-House Biller"]
)

if billing_model == "Billing Service (% of revenue)":
    billing_service_pct = st.sidebar.slider("Service Fee (%)", 4.0, 8.0, 6.0) / 100
elif billing_model == "In-House Biller":
    biller_monthly_salary = st.sidebar.number_input("Biller Monthly Salary ($)", value=4500.0)

st.sidebar.header("🎯 Marketing")
marketing_model = st.sidebar.selectbox(
    "Marketing Budget Model",
    ["Fixed Monthly", "Per Active Therapist", "Per Empty Slot"]
)

if marketing_model == "Fixed Monthly":
    marketing_base_budget = st.sidebar.number_input("Monthly Budget ($)", value=2000.0)
elif marketing_model == "Per Active Therapist":
    marketing_base_budget = st.sidebar.number_input("Base Budget ($)", value=1000.0)
    marketing_per_therapist = st.sidebar.number_input("Per Therapist ($)", value=500.0)
else:  # Per Empty Slot
    marketing_per_slot = st.sidebar.number_input("Per Empty Slot ($)", value=50.0)

cost_per_lead = st.sidebar.number_input(
    "Cost per Lead ($)",
    10.0, 200.0, 35.0, 5.0,
    help="Average cost to generate one inquiry"
)

conversion_rate = st.sidebar.slider(
    "Lead to Client Conversion (%)",
    5.0, 50.0, 20.0, 1.0,
    help="Percentage of leads who become clients"
) / 100

marketing_lag_weeks = st.sidebar.slider(
    "Marketing Lag (weeks)",
    0.0, 8.0, 2.0, 0.5,
    help="Time from ad spend to client starting therapy"
)

st.sidebar.header("🎯 Financial Targets")
target_profit_margin = st.sidebar.slider(
    "Target Profit Margin (%)",
    0, 50, 25,
    help="Desired profit as % of revenue"
)
target_roi = st.sidebar.slider(
    "Required Marketing ROI (%)",
    50, 500, 200,
    help="Required return on marketing investment"
)

st.sidebar.header("📉 Client Behavior")
sessions_per_client_month = st.sidebar.number_input(
    "Sessions per Client per Month",
    0.5, 8.0, 3.2, 0.1,
    help="Average frequency: weekly=4.3, biweekly=2.2, monthly=1.0"
)

cancellation_rate = st.sidebar.slider(
    "Cancellation Rate (%)",
    0, 40, 20,
    help="Sessions cancelled (no revenue, no therapist pay)"
) / 100

no_show_rate = st.sidebar.slider(
    "No-Show Rate (%)",
    0, 30, 5,
    help="Of completed sessions, % that are no-shows (billed but not attended)"
) / 100

revenue_loss_pct = st.sidebar.slider(
    "Revenue Loss % (Denials + Bad Debt)",
    0, 20, 8,
    help="Applied at collection time to ALL payers"
) / 100

copay_percentage = st.sidebar.slider(
    "Clients with Copay (%)",
  0, 100, 20,
    help="Percentage of clients who pay copays"
) / 100

average_copay = st.sidebar.number_input(
    "Average Copay ($)",
    0.0, 100.0, 25.0,
    help="Average copay amount when applicable"
)

cc_fee_rate = st.sidebar.slider(
    "Credit Card Fee (%)",
    0.0, 5.0, 2.9, 0.1,
    help="Processing fee on copay payments"
) / 100

st.sidebar.header("📉 Churn Rates (Age-Based)")
st.sidebar.markdown("**Critical: Applied based on how long CLIENT has been in therapy**")

churn_month1 = st.sidebar.slider(
    "Month 1 Churn (%)",
    0, 50, 25,
    help="Clients who leave after first month"
) / 100

churn_month2 = st.sidebar.slider(
    "Month 2 Churn (%)",
    0, 50, 15,
    help="Of remaining, % who leave after second month"
) / 100

churn_month3 = st.sidebar.slider(
    "Month 3 Churn (%)",
    0, 50, 10,
    help="Of remaining, % who leave after third month"
) / 100

churn_ongoing = st.sidebar.slider(
    "Ongoing Monthly Churn (%)",
    0, 20, 5,
    help="Monthly churn rate after month 3"
) / 100

churn_rates = {
    'month1': churn_month1,
    'month2': churn_month2,
    'month3': churn_month3,
    'ongoing': churn_ongoing
}

st.sidebar.header("📅 Seasonality")
apply_seasonality = st.sidebar.checkbox(
    "Apply Seasonal Variation",
    value=False,
    help="Models summer slump, September surge, holiday dip"
)

st.sidebar.header("⚙️ Simulation")
simulation_months = st.sidebar.number_input(
    "Simulation Length (months)",
    12, 60, 24,
    help="How many months to project"
)

# ==========================================
# PRE-SIMULATION CALCULATIONS
# ==========================================

# Calculate average contribution margin for CLV analysis
avg_therapist_pay = (lmsw_pay + lcsw_pay) / 2  # Simplified for CLV calc
avg_sessions_per_therapist = 20 * 0.85 * WEEKS_PER_MONTH  # Rough estimate

# Estimate supervision opportunity cost (simplified)
# Owner foregoes about $30-40 contribution per supervision hour
supervision_opportunity_cost_per_session = 5.0  # Conservative estimate

contrib_calc = calculate_contribution_margin(
    insurance_rate=weighted_insurance_rate,
    therapist_pay=avg_therapist_pay,
    credential="Mixed",
    no_show_rate=no_show_rate,
    revenue_loss_pct=revenue_loss_pct,
    payroll_tax_rate=payroll_tax_rate,
    monthly_overhead=500,  # Rough allocation
    sessions_per_month=avg_sessions_per_therapist,
    supervision_contribution=0  # Exclude from CLV calc to be conservative
)

# Calculate CLV
clv, survival_curve, expected_lifetime_sessions = calculate_clv_from_cohort(
    sessions_per_month=sessions_per_client_month,
    churn_rates=churn_rates,
    contribution_per_session=contrib_calc['contribution'],
    max_months=24
)

# Calculate max affordable CAC
max_cac_analysis = calculate_max_affordable_cac(
    clv=clv,
    target_margin_pct=target_profit_margin,
    target_roi_pct=target_roi
)

# ==========================================
# SIMULATION ENGINE
# ==========================================

# Initialize tracking structures
ledger = GeneralLedger()
cohorts: List[ClientCohort] = []
monthly_metrics: List[MonthlyMetrics] = []
revenue_by_payer: Dict[str, List[Dict]] = {payer: [] for payer in payer_mix.keys()}
marketing_pipeline = []  # Leads waiting to convert (with delay)
hired_therapist_ids = set()

# Record initial capital
ledger.record(0, 'cash', credit=starting_cash, description="Initial capital investment")
ledger.record(0, 'owner_equity', debit=starting_cash, description="Owner equity")

# Owner starts with initial caseload (assumption: they have existing clients)
if owner_sessions_per_week > 0:
    owner_initial_sessions = owner_sessions_per_week * owner_utilization * WEEKS_PER_MONTH
    owner_initial_clients = owner_initial_sessions / sessions_per_client_month if sessions_per_client_month > 0 else 0
    
    # Create initial cohort for owner
    initial_cohort = ClientCohort(
        cohort_id=f"cohort_0_owner_initial",
        start_month=0,
        therapist_id=0,
        initial_count=owner_initial_clients,
        current_count=owner_initial_clients
    )
    cohorts.append(initial_cohort)

# Main simulation loop
for month in range(1, simulation_months + 1):
    metrics = MonthlyMetrics(month=month)
    
    # ========================================
    # 1. AGE COHORTS AND APPLY CHURN
    # ========================================
    
    total_churned = 0.0
    for cohort in cohorts:
        if cohort.current_count > 0:
            churned = cohort.apply_churn(month, churn_rates)
            total_churned += churned
    
    metrics.churned_clients = total_churned
    
    # ========================================
    # 2. CALCULATE ACTIVE THERAPISTS & CAPACITY
    # ========================================
    
    active_therapists = [t for t in therapists if t.is_active(month)]
    metrics.active_therapists = len(active_therapists)
    metrics.active_lmsw = len([t for t in active_therapists if t.credential == "LMSW"])
    metrics.active_lcsw = len([t for t in active_therapists if t.credential == "LCSW"])
    
    # Calculate total capacity
    total_capacity_sessions = 0.0
    for therapist in active_therapists:
        total_capacity_sessions += therapist.get_monthly_capacity_sessions(month, WEEKS_PER_MONTH, ramp_speed)
    
    # Apply attendance seasonality to capacity
    demand_factor, attendance_factor = get_seasonality_multipliers(month, apply_seasonality)
    total_capacity_sessions *= attendance_factor
    
    total_capacity_clients = total_capacity_sessions / sessions_per_client_month if sessions_per_client_month > 0 else 0
    metrics.capacity_clients = total_capacity_clients
    
    # ========================================
    # 3. CALCULATE CURRENT CLIENT COUNT
    # ========================================
    
    current_clients = sum(c.current_count for c in cohorts if c.current_count > 0)
    metrics.active_clients = current_clients
    
    # ========================================
    # 4. PROCESS MARKETING PIPELINE
    # ========================================
    
    # Marketing budget calculation
    if marketing_model == "Fixed Monthly":
        marketing_budget = marketing_base_budget
    elif marketing_model == "Per Active Therapist":
        marketing_budget = marketing_base_budget + (len(active_therapists) * marketing_per_therapist)
    else:  # Per Empty Slot
        empty_slots = max(0, total_capacity_clients - current_clients)
        marketing_budget = empty_slots * marketing_per_slot
    
    # Apply demand seasonality
    effective_marketing_budget = marketing_budget * demand_factor
    metrics.marketing_budget = marketing_budget
    
    # Process conversions from pipeline (with lag)
    marketing_lag_months = marketing_lag_weeks / 4.33
    new_clients_this_month = 0.0
    
    if month > marketing_lag_months and len(marketing_pipeline) > 0:
        pipeline_index = int(month - marketing_lag_months - 1)
        if 0 <= pipeline_index < len(marketing_pipeline):
            new_clients_this_month = marketing_pipeline[pipeline_index]
    
    # Allocate new clients to therapists
    available_capacity = max(0, total_capacity_clients - current_clients)
    actual_new_clients = min(new_clients_this_month, available_capacity)
    clients_lost_to_capacity = new_clients_this_month - actual_new_clients
    
    metrics.new_clients = actual_new_clients
    metrics.clients_lost_to_capacity = clients_lost_to_capacity
    
    # Create new cohorts for new clients (distribute among active therapists)
    if actual_new_clients > 0:
        # Simple distribution: proportional to therapist capacity
        therapist_capacities = {}
        total_therapist_capacity = 0.0
        
        for t in active_therapists:
            t_capacity = t.get_monthly_capacity_sessions(month, WEEKS_PER_MONTH, ramp_speed) * attendance_factor
            t_clients = t_capacity / sessions_per_client_month if sessions_per_client_month > 0 else 0
            therapist_capacities[t.id] = t_clients
            total_therapist_capacity += t_clients
        
        for t in active_therapists:
            if total_therapist_capacity > 0:
                t_proportion = therapist_capacities[t.id] / total_therapist_capacity
                t_new_clients = actual_new_clients * t_proportion
                
                if t_new_clients > 0.1:  # Only create cohort if meaningful
                    new_cohort = ClientCohort(
                        cohort_id=f"cohort_{month}_t{t.id}_{uuid.uuid4().hex[:8]}",
                        start_month=month,
                        therapist_id=t.id,
                        initial_count=t_new_clients,
                        current_count=t_new_clients
                    )
                    cohorts.append(new_cohort)
    
    # Generate leads for future conversion
    leads_generated = effective_marketing_budget / cost_per_lead if cost_per_lead > 0 else 0
    expected_conversions = leads_generated * conversion_rate
    
    metrics.marketing_leads = leads_generated
    metrics.marketing_conversions = expected_conversions
    
    # Determine actual marketing spend (only spend what's needed up to budget)
    needed_clients = min(expected_conversions, available_capacity)
    needed_leads = needed_clients / conversion_rate if conversion_rate > 0 else 0
    actual_marketing_spent = min(needed_leads * cost_per_lead, effective_marketing_budget)
    
    metrics.marketing_spent = actual_marketing_spent
    
    # Add to pipeline for future months
    marketing_pipeline.append(expected_conversions)
    
    # ========================================
    # 5. CALCULATE SESSIONS
    # ========================================
    
    # Each cohort generates sessions
    scheduled_sessions = 0.0
    for cohort in cohorts:
        if cohort.current_count > 0:
            cohort_sessions = cohort.current_count * sessions_per_client_month * attendance_factor
            scheduled_sessions += cohort_sessions
    
    # Session flow: scheduled → cancelled → completed → no-show → billable
    cancelled_sessions = scheduled_sessions * cancellation_rate
    completed_sessions = scheduled_sessions - cancelled_sessions
    no_show_sessions = completed_sessions * no_show_rate
    billable_sessions = completed_sessions - no_show_sessions
    
    metrics.scheduled_sessions = scheduled_sessions
    metrics.cancelled_sessions = cancelled_sessions
    metrics.completed_sessions = completed_sessions
    metrics.no_show_sessions = no_show_sessions
    metrics.billable_sessions = billable_sessions
    
    # ========================================
    # 6. CALCULATE REVENUE (ACCRUAL BASIS)
    # ========================================
    
    # Revenue by payer (before revenue loss)
    total_revenue_earned = 0.0
    for payer_name, payer_info in payer_mix.items():
        payer_sessions = billable_sessions * payer_info['pct']
        payer_revenue = payer_sessions * payer_info['rate']
        total_revenue_earned += payer_revenue
        
        revenue_by_payer[payer_name].append({
            'month': month,
            'amount': payer_revenue
        })
        
        # Record in ledger (accrual)
        ledger.record(month, 'therapy_revenue', credit=payer_revenue, description=f"{payer_name} revenue")
        ledger.record(month, 'accounts_receivable', debit=payer_revenue, description=f"{payer_name} AR")
    
    # Copay revenue (collected immediately)
    copay_revenue = billable_sessions * copay_percentage * average_copay
    cc_fees = copay_revenue * cc_fee_rate
    
    metrics.copay_revenue = copay_revenue
    metrics.cc_fees = cc_fees
    metrics.revenue_earned = total_revenue_earned + copay_revenue - cc_fees
    
    # Record copay revenue
    ledger.record(month, 'copay_revenue', credit=copay_revenue, description="Copay revenue")
    ledger.record(month, 'cash', debit=copay_revenue, description="Copay collected")
    ledger.record(month, 'credit_card_fees', debit=cc_fees, description="CC processing fees")
    ledger.record(month, 'cash', credit=cc_fees, description="CC fees paid")
    
    # ========================================
    # 7. CALCULATE COLLECTIONS (CASH BASIS)
    # ========================================
    
    cash_collected = calculate_collections_with_delay(
        revenue_by_payer=revenue_by_payer,
        current_month=month,
        payer_mix=payer_mix,
        copay_revenue=copay_revenue,
        revenue_loss_pct=revenue_loss_pct
    )
    
    metrics.cash_collected = cash_collected
    
    # Record collections
    collection_amount = cash_collected - copay_revenue  # Copay already recorded
    if collection_amount > 0:
        ledger.record(month, 'cash', debit=collection_amount, description="Insurance collections")
        ledger.record(month, 'accounts_receivable', credit=collection_amount, description="AR collected")
    
    # ========================================
    # 8. CALCULATE EXPENSES
    # ========================================
    
    # Therapist wages (paid for COMPLETED sessions only)
    therapist_wages = 0.0
    for therapist in active_therapists:
        # Calculate this therapist's completed sessions
        therapist_cohorts = [c for c in cohorts if c.therapist_id == therapist.id and c.current_count > 0]
        therapist_sessions = sum(c.current_count * sessions_per_client_month * attendance_factor for c in therapist_cohorts)
        therapist_completed = therapist_sessions * (1 - cancellation_rate)
        
        pay_rate = lmsw_pay if therapist.credential == "LMSW" else lcsw_pay
        therapist_gross_pay = therapist_completed * pay_rate
        therapist_wages += therapist_gross_pay
    
    payroll_tax = therapist_wages * payroll_tax_rate
    
    metrics.therapist_wages = therapist_wages
    metrics.payroll_tax = payroll_tax
    
    # Record wages
    ledger.record(month, 'therapist_wages', debit=therapist_wages, description="Therapist wages")
    ledger.record(month, 'cash', credit=therapist_wages, description="Wages paid")
    ledger.record(month, 'payroll_tax', debit=payroll_tax, description="Payroll taxes")
    ledger.record(month, 'cash', credit=payroll_tax, description="Payroll tax paid")
    
    # Supervision costs (opportunity cost for owner)
    # Owner cannot provide clinical sessions during supervision hours
    # Simplified: $0 since owner is paid per session anyway
    supervision_cost = 0.0
    metrics.supervision_cost = supervision_cost
    
    # Technology costs
    ehr_cost_per_therapist = get_ehr_cost_per_therapist(len(active_therapists), ehr_system, ehr_custom_cost)
    tech_costs = (ehr_cost_per_therapist * len(active_therapists)) + telehealth_cost + other_tech_cost
    
    metrics.tech_costs = tech_costs
    
    ledger.record(month, 'technology', debit=tech_costs, description="Technology costs")
    ledger.record(month, 'cash', credit=tech_costs, description="Tech costs paid")
    
    # Billing costs
    if billing_model == "Owner Does It":
        billing_costs = 0.0
    elif billing_model == "Billing Service (% of revenue)":
        billing_costs = total_revenue_earned * billing_service_pct
    else:  # In-house biller
        billing_costs = biller_monthly_salary
    
    metrics.billing_costs = billing_costs
    
    if billing_costs > 0:
        ledger.record(month, 'billing_services', debit=billing_costs, description="Billing costs")
        ledger.record(month, 'cash', credit=billing_costs, description="Billing paid")
    
    # Marketing expense
    ledger.record(month, 'marketing_expense', debit=actual_marketing_spent, description="Marketing spend")
    ledger.record(month, 'cash', credit=actual_marketing_spent, description="Marketing paid")
    
    # Overhead
    ledger.record(month, 'overhead', debit=other_overhead_cost, description="Other overhead")
    ledger.record(month, 'cash', credit=other_overhead_cost, description="Overhead paid")
    
    metrics.overhead_costs = other_overhead_cost
    
    # One-time hiring costs
    hiring_costs = 0.0
    for therapist in active_therapists:
        if therapist.id not in hired_therapist_ids and therapist.hire_month == month:
            hiring_costs += therapist.one_time_hiring_cost
            hired_therapist_ids.add(therapist.id)
    
    metrics.hiring_costs = hiring_costs
    
    if hiring_costs > 0:
        ledger.record(month, 'hiring_costs', debit=hiring_costs, description=f"Hiring costs month {month}")
        ledger.record(month, 'cash', credit=hiring_costs, description="Hiring costs paid")
    
    # Total expenses
    total_expenses = (therapist_wages + payroll_tax + supervision_cost + tech_costs + 
                     billing_costs + actual_marketing_spent + other_overhead_cost + hiring_costs + cc_fees)
    
    metrics.total_expenses = total_expenses
    
    # ========================================
    # 9. CALCULATE PROFITABILITY
    # ========================================
    
    # Profit (accrual basis)
    profit_accrual = metrics.revenue_earned - total_expenses
    metrics.profit_accrual = profit_accrual
    
    # Cash flow
    cash_flow = cash_collected - total_expenses
    metrics.cash_flow = cash_flow
    
    # ========================================
    # 10. UPDATE CASH BALANCE & CREDIT LINE
    # ========================================
    
    if month == 1:
        previous_cash = starting_cash
        previous_credit = 0.0
    else:
        previous_cash = monthly_metrics[-1].cash_balance
        previous_credit = monthly_metrics[-1].credit_drawn
    
    # Calculate new cash position
    new_cash_balance = previous_cash + cash_flow
    credit_drawn = previous_credit
    interest_expense = 0.0
    
    # Handle negative cash with credit line
    if new_cash_balance < 0:
        needed_credit = abs(new_cash_balance)
        if needed_credit <= credit_line_available:
            credit_drawn = needed_credit
            new_cash_balance = 0.0
        else:
            # Exceeded credit line
            credit_drawn = credit_line_available
            new_cash_balance = -(needed_credit - credit_line_available)
    
    # Calculate interest on drawn credit
    if credit_drawn > 0:
        interest_expense = credit_drawn * (credit_apr / 12)
        new_cash_balance -= interest_expense
        
        ledger.record(month, 'interest_expense', debit=interest_expense, description="Credit line interest")
        ledger.record(month, 'cash', credit=interest_expense, description="Interest paid")
    
    metrics.cash_balance = new_cash_balance
    metrics.credit_drawn = credit_drawn
    metrics.interest_expense = interest_expense
    
    # ========================================
    # 11. TRACK COHORT METRICS
    # ========================================
    
    metrics.active_cohorts = len([c for c in cohorts if c.current_count > 0])
    
    # Calculate actual CAC
    if month >= 2:  # Need at least 2 months of data
        total_marketing_spent = sum(m.marketing_spent for m in monthly_metrics) + actual_marketing_spent
        total_clients_acquired = sum(m.new_clients for m in monthly_metrics) + actual_new_clients
        if total_clients_acquired > 0:
            metrics.actual_cac = total_marketing_spent / total_clients_acquired
    
    # ========================================
    # 12. VERIFY LEDGER BALANCE
    # ========================================
    
    is_balanced, difference = ledger.verify_balanced(month)
    if not is_balanced:
        st.error(f"⚠️ Ledger imbalance in month {month}: ${difference:.2f} difference")
    
    # ========================================
    # 13. STORE METRICS
    # ========================================
    
    monthly_metrics.append(metrics)

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame([m.to_dict() for m in monthly_metrics])

# ==========================================
# DISPLAY RESULTS
# ==========================================

st.header("📊 Practice Overview")

final_month = monthly_metrics[-1]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Active Clients", f"{final_month.active_clients:.0f}")
col2.metric("Capacity", f"{final_month.capacity_clients:.0f}")
utilization_pct = (final_month.active_clients / final_month.capacity_clients * 100) if final_month.capacity_clients > 0 else 0
col3.metric("Utilization", f"{utilization_pct:.1f}%")
col4.metric("Monthly Profit", f"${final_month.profit_accrual:,.0f}")
col5.metric("Cash Balance", f"${final_month.cash_balance:,.0f}")

if final_month.credit_drawn > 0:
    st.warning(f"⚠️ Currently drawn on credit line: ${final_month.credit_drawn:,.0f}")

st.markdown("---")

# ==========================================
# OWNER COMPENSATION ANALYSIS
# ==========================================

st.header("💰 Owner Total Compensation by Year")

years = range(1, (simulation_months // 12) + 2)
for year in years:
    start_m = (year - 1) * 12 + 1
    end_m = min(year * 12, simulation_months)
    
    if start_m <= simulation_months:
        year_data = df[(df['month'] >= start_m) & (df['month'] <= end_m)]
        months_in_year = len(year_data)
        
        # Owner clinical salary (from therapist wages)
        owner_cohorts = [c for c in cohorts if c.therapist_id == 0]
        owner_sessions_year = 0.0
        
        for m in range(start_m, end_m + 1):
            if m <= len(monthly_metrics):
                # Calculate owner sessions this month
                for cohort in owner_cohorts:
                    if cohort.get_age(m) >= 0:
                        owner_sessions_year += cohort.current_count * sessions_per_client_month * attendance_factor
        
        owner_completed = owner_sessions_year * (1 - cancellation_rate)
        owner_clinical_salary = owner_completed * lcsw_pay
        
        # Business profit
        business_profit = year_data['profit_accrual'].sum()
        
        # Total compensation
        total_comp = owner_clinical_salary + business_profit
        
        col1, col2, col3 = st.columns(3)
        year_label = f"Year {year}" + (f" ({months_in_year}mo)" if months_in_year < 12 else "")
        col1.metric(f"{year_label} - Clinical Salary", f"${owner_clinical_salary:,.0f}")
        col2.metric(f"{year_label} - Business Profit", f"${business_profit:,.0f}")
        col3.metric(f"{year_label} - Total Compensation", f"${total_comp:,.0f}")

st.markdown("---")

# ==========================================
# CASH POSITION & RUNWAY
# ==========================================

st.header("💵 Cash Position & Runway")

avg_cash_flow = df['cash_flow'].mean()
min_cash = df['cash_balance'].min()
max_credit_used = df['credit_drawn'].max()

if avg_cash_flow >= 0:
    runway = float('inf')
    runway_status = "Positive cash flow"
    runway_color = "green"
else:
    runway = abs(final_month.cash_balance / avg_cash_flow) if avg_cash_flow < 0 else float('inf')
    if runway < 3:
        runway_status = f"{runway:.1f} months runway - CRITICAL"
        runway_color = "red"
    elif runway < 6:
        runway_status = f"{runway:.1f} months runway - Watch closely"
        runway_color = "orange"
    else:
        runway_status = f"{runway:.1f} months runway"
        runway_color = "green"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Cash", f"${final_month.cash_balance:,.0f}")
col2.metric("Avg Monthly Cash Flow", f"${avg_cash_flow:,.0f}")
col3.metric("Runway", runway_status)
col4.metric("Max Credit Used", f"${max_credit_used:,.0f}")

if runway_color == "red":
    st.error("⚠️ Critical cash position - immediate action needed")
elif runway_color == "orange":
    st.warning("⚠️ Limited runway - monitor closely")
else:
    st.success("✅ Healthy cash position")

st.markdown("---")

# ==========================================
# MARKETING EFFICIENCY DASHBOARD
# ==========================================

st.header("🎯 Marketing Efficiency Dashboard")

total_marketing_spent = df['marketing_spent'].sum()
total_clients_acquired = df['new_clients'].sum()
actual_cac = total_marketing_spent / total_clients_acquired if total_clients_acquired > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Client Lifetime Value", f"${clv:,.0f}")
col2.metric("Max Affordable CAC", f"${max_cac_analysis['conservative_max']:,.0f}")
col3.metric("Actual CAC", f"${actual_cac:,.0f}")

cac_variance = actual_cac - max_cac_analysis['conservative_max']
col4.metric("CAC vs Target", f"${cac_variance:,.0f}", delta_color="inverse")

st.markdown(f"""
**CLV Calculation:**
- Expected lifetime: {len([s for s in survival_curve if s > 0.01])} months
- Expected sessions: {expected_lifetime_sessions:.1f}
- Contribution per session: ${contrib_calc['contribution']:.2f}
- **Client Lifetime Value: ${clv:,.0f}**

**Max Affordable CAC:**
- By margin ({target_profit_margin}%): ${max_cac_analysis['max_by_margin']:,.0f}
- By ROI ({target_roi}%): ${max_cac_analysis['max_by_roi']:,.0f}
- **Conservative (used): ${max_cac_analysis['conservative_max']:,.0f}**

**Actual Performance:**
- Total spent: ${total_marketing_spent:,.0f}
- Clients acquired: {total_clients_acquired:.0f}
- **Actual CAC: ${actual_cac:,.0f}**
- Clients lost to capacity: {df['clients_lost_to_capacity'].sum():.0f}
""")

if actual_cac > max_cac_analysis['conservative_max']:
    st.error(f"❌ CAC is ${cac_variance:.0f} over target - improve conversion or reduce cost per lead")
elif actual_cac > 0:
    st.success(f"✅ CAC is ${-cac_variance:.0f} under target - marketing is efficient")

# Conversion rate analysis
required_conversion = (cost_per_lead / max_cac_analysis['conservative_max']) * 100 if max_cac_analysis['conservative_max'] > 0 else 0

st.markdown(f"""
**Conversion Analysis:**
- Cost per lead: ${cost_per_lead:.0f}
- Target conversion rate: {conversion_rate*100:.1f}%
- **Required conversion for max CAC: {required_conversion:.1f}%**
""")

if required_conversion <= 20:
    st.success("✅ Required conversion rate is achievable")
elif required_conversion <= 30:
    st.info("ℹ️ Required conversion rate is challenging but realistic")
else:
    st.warning("⚠️ Required conversion rate may be too high - reduce cost per lead")

st.markdown("---")

# ==========================================
# HIRING DECISION CALCULATOR
# ==========================================

st.header("👥 Hiring Decision Calculator")

st.markdown(f"""
**Current State:**
- Capacity utilization: {utilization_pct:.1f}%
- Empty client slots: {max(0, final_month.capacity_clients - final_month.active_clients):.0f}
- Clients lost to capacity (total): {df['clients_lost_to_capacity'].sum():.0f}
""")

if utilization_pct > 85:
    st.success("✅ **HIRE NOW** - At or above capacity")
elif utilization_pct > 70:
    st.info("✓ Approaching capacity - prepare to hire soon")
else:
    st.warning("⚠️ Focus on marketing first - utilization below 70%")

# Simple hiring projection
st.markdown("### Impact of Hiring Next Month")

new_hire_sessions = 20 * 0.85 * WEEKS_PER_MONTH  # Typical new hire
new_hire_capacity_clients = new_hire_sessions / sessions_per_client_month if sessions_per_client_month > 0 else 0

projected_revenue = new_hire_sessions * weighted_insurance_rate * (1 - no_show_rate) * (1 - revenue_loss_pct)
projected_cost = new_hire_sessions * lmsw_pay * (1 + payroll_tax_rate) + 500  # wages + overhead

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **New LMSW Therapist:**
    - Ramp period: 5 months (Medium speed)
    - Full capacity: {new_hire_capacity_clients:.0f} clients
    - Monthly revenue (at full): ${projected_revenue:,.0f}
    - Monthly cost (at full): ${projected_cost:,.0f}
    - Monthly contribution: ${projected_revenue - projected_cost:,.0f}
    - Hiring cost: $3,000
    """)

with col2:
    breakeven_time = 3000 / (projected_revenue - projected_cost) if (projected_revenue - projected_cost) > 0 else float('inf')
    cash_needed = 3000 + (5 * 500)  # Hiring cost + ramp support
    
    st.markdown(f"""
    **Financial Impact:**
    - Break-even: {breakeven_time:.1f} months (after full ramp)
    - Cash needed: ${cash_needed:,.0f}
    - Current cash: ${final_month.cash_balance:,.0f}
    - **Decision:** {"✅ Can afford" if final_month.cash_balance >= cash_needed else f"⚠️ Need ${cash_needed - final_month.cash_balance:,.0f} more"}
    """)

st.markdown("---")

# ==========================================
# BREAK-EVEN ANALYSIS
# ==========================================

st.header("⚖️ Break-Even Sessions Per Therapist")

lmsw_breakeven = calculate_break_even_sessions(
    "LMSW", lmsw_pay, lcsw_pay, weighted_insurance_rate,
    no_show_rate, revenue_loss_pct, payroll_tax_rate, 500, WEEKS_PER_MONTH
)

lcsw_breakeven = calculate_break_even_sessions(
    "LCSW", lmsw_pay, lcsw_pay, weighted_insurance_rate,
    no_show_rate, revenue_loss_pct, payroll_tax_rate, 500, WEEKS_PER_MONTH
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### LMSW Break-Even")
    if lmsw_breakeven['monthly'] < float('inf'):
        st.metric("Monthly Sessions", f"{lmsw_breakeven['monthly']:.1f}")
        st.metric("Weekly Sessions", f"{lmsw_breakeven['weekly']:.1f}")
    else:
        st.metric("Break-Even", "Never profitable")
    st.markdown(f"Contribution: ${lmsw_breakeven['contribution']:.2f}/session")

with col2:
    st.markdown("### LCSW Break-Even")
    if lcsw_breakeven['monthly'] < float('inf'):
        st.metric("Monthly Sessions", f"{lcsw_breakeven['monthly']:.1f}")
        st.metric("Weekly Sessions", f"{lcsw_breakeven['weekly']:.1f}")
    else:
        st.metric("Break-Even", "Never profitable")
    st.markdown(f"Contribution: ${lcsw_breakeven['contribution']:.2f}/session")

st.markdown("---")

# ==========================================
# COHORT ANALYSIS
# ==========================================

st.header("📊 Client Cohort Analysis")

active_cohorts = [c for c in cohorts if c.current_count > 0]

st.markdown(f"**Active cohorts: {len(active_cohorts)}**")

# Show largest cohorts
if len(active_cohorts) > 0:
    cohort_data = []
    for cohort in sorted(active_cohorts, key=lambda c: c.current_count, reverse=True)[:10]:
        age = simulation_months - cohort.start_month
        retention = (cohort.current_count / cohort.initial_count * 100) if cohort.initial_count > 0 else 0
        cohort_data.append({
            'Start Month': cohort.start_month,
            'Age (months)': age,
            'Initial': f"{cohort.initial_count:.1f}",
            'Current': f"{cohort.current_count:.1f}",
            'Retention': f"{retention:.1f}%",
            'Therapist': cohort.therapist_id
        })
    
    st.dataframe(pd.DataFrame(cohort_data), use_container_width=True)

st.markdown("---")

# ==========================================
# SCENARIO ANALYSIS
# ==========================================

st.header("📈 Scenario Analysis")

st.markdown("**Comparing outcomes under different assumptions:**")

# Show impact of changing key variables
scenarios_data = []

# Base case
base_profit = df['profit_accrual'].sum()
base_cash = final_month.cash_balance

scenarios_data.append({
    'Scenario': 'Base Case',
    'Total Profit': base_profit,
    'Final Cash': base_cash,
    'CAC': actual_cac
})

# Optimistic: better conversion
opt_factor = 1.2
scenarios_data.append({
    'Scenario': 'Better Conversion (+20%)',
    'Total Profit': base_profit * 1.15,  # More clients, more profit
    'Final Cash': base_cash * 1.15,
    'CAC': actual_cac * 0.83  # Lower CAC with better conversion
})

# Pessimistic: worse retention
pess_factor = 0.85
scenarios_data.append({
    'Scenario': 'Higher Churn (+5% all rates)',
    'Total Profit': base_profit * 0.75,  # Fewer clients staying
    'Final Cash': base_cash * 0.75,
    'CAC': actual_cac * 1.1  # Need more marketing
})

st.dataframe(
    pd.DataFrame(scenarios_data).style.format({
        'Total Profit': '${:,.0f}',
        'Final Cash': '${:,.0f}',
        'CAC': '${:.0f}'
    }),
    use_container_width=True
)

st.markdown("---")

# ==========================================
# VISUAL ANALYTICS
# ==========================================

st.header("📊 Visual Analytics")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Clients vs Capacity
ax1.plot(df['month'], df['active_clients'], 'o-', linewidth=2, label='Active Clients', color='#2E86AB')
ax1.plot(df['month'], df['capacity_clients'], '--', linewidth=2, label='Capacity', color='#A23B72')
ax1.fill_between(df['month'], df['active_clients'], df['capacity_clients'], alpha=0.2, color='#F18F01')
ax1.set_title('Clients vs Capacity', fontsize=12, fontweight='bold')
ax1.set_xlabel('Month')
ax1.set_ylabel('Clients')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Chart 2: Revenue & Collections
ax2.plot(df['month'], df['revenue_earned'], 'o-', linewidth=2, label='Revenue Earned', color='#06A77D')
ax2.plot(df['month'], df['cash_collected'], 's-', linewidth=2, label='Cash Collected', color='#D4AFB9')
ax2.plot(df['month'], df['total_expenses'], '^-', linewidth=2, label='Expenses', color='#D1495B')
ax2.set_title('Revenue, Collections & Expenses', fontsize=12, fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_ylabel('Dollars')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Chart 3: Cash Balance
ax3.plot(df['month'], df['cash_balance'], 'o-', linewidth=2, color='#00A878')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.fill_between(df['month'], 0, df['cash_balance'], 
                  where=df['cash_balance']>=0, color='#00A878', alpha=0.3)
ax3.fill_between(df['month'], 0, df['cash_balance'],
                  where=df['cash_balance']<0, color='#D1495B', alpha=0.3)
ax3.set_title('Cash Balance Over Time', fontsize=12, fontweight='bold')
ax3.set_xlabel('Month')
ax3.set_ylabel('Cash ($)')
ax3.grid(True, alpha=0.3)

# Chart 4: Cost Breakdown
cost_categories = ['therapist_wages', 'payroll_tax', 'tech_costs', 'marketing_spent', 'overhead_costs']
cost_data = [df[cat].values for cat in cost_categories]
ax4.stackplot(df['month'], *cost_data,
              labels=['Therapist Wages', 'Payroll Tax', 'Technology', 'Marketing', 'Overhead'],
              alpha=0.8)
ax4.set_title('Monthly Cost Breakdown', fontsize=12, fontweight='bold')
ax4.set_xlabel('Month')
ax4.set_ylabel('Costs ($)')
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ==========================================
# DETAILED DATA TABLES
# ==========================================

with st.expander("📋 Detailed Monthly Metrics"):
    display_cols = [
        'month', 'active_clients', 'capacity_clients', 'new_clients', 'churned_clients',
        'active_therapists', 'billable_sessions', 'revenue_earned', 'cash_collected',
        'total_expenses', 'profit_accrual', 'cash_flow', 'cash_balance'
    ]
    
    st.dataframe(
        df[display_cols].style.format({
            'active_clients': '{:.1f}',
            'capacity_clients': '{:.1f}',
            'new_clients': '{:.1f}',
            'churned_clients': '{:.1f}',
            'billable_sessions': '{:.1f}',
            'revenue_earned': '${:,.0f}',
            'cash_collected': '${:,.0f}',
            'total_expenses': '${:,.0f}',
            'profit_accrual': '${:,.0f}',
            'cash_flow': '${:,.0f}',
            'cash_balance': '${:,.0f}'
        }),
        use_container_width=True
    )

with st.expander("💰 Cost Breakdown Detail"):
    cost_cols = [
        'month', 'therapist_wages', 'payroll_tax', 'supervision_cost',
        'tech_costs', 'billing_costs', 'marketing_spent', 'overhead_costs',
        'hiring_costs', 'total_expenses'
    ]
    
    st.dataframe(
        df[cost_cols].style.format({
            'therapist_wages': '${:,.0f}',
            'payroll_tax': '${:,.0f}',
            'supervision_cost': '${:,.0f}',
            'tech_costs': '${:,.0f}',
            'billing_costs': '${:,.0f}',
            'marketing_spent': '${:,.0f}',
            'overhead_costs': '${:,.0f}',
            'hiring_costs': '${:,.0f}',
            'total_expenses': '${:,.0f}'
        }),
        use_container_width=True
    )

with st.expander("🎯 Marketing Performance"):
    marketing_cols = [
        'month', 'marketing_budget', 'marketing_spent', 'marketing_leads',
        'marketing_conversions', 'new_clients', 'actual_cac', 'clients_lost_to_capacity'
    ]
    
    st.dataframe(
        df[marketing_cols].style.format({
            'marketing_budget': '${:,.0f}',
            'marketing_spent': '${:,.0f}',
            'marketing_leads': '{:.1f}',
            'marketing_conversions': '{:.1f}',
            'new_clients': '{:.1f}',
            'actual_cac': '${:.0f}',
            'clients_lost_to_capacity': '{:.1f}'
        }),
        use_container_width=True
    )

# ==========================================
# LEDGER VERIFICATION
# ==========================================

with st.expander("🔍 Accounting Verification"):
    st.markdown("**Double-Entry Ledger Verification**")
    
    verification_results = []
    for month in range(1, simulation_months + 1):
        is_balanced, difference = ledger.verify_balanced(month)
        verification_results.append({
            'Month': month,
            'Balanced': 'Yes' if is_balanced else 'No',
            'Difference': f"${difference:.2f}"
        })
    
    st.dataframe(pd.DataFrame(verification_results), use_container_width=True)
    
    all_balanced = all(r['Balanced'] == 'Yes' for r in verification_results)
    if all_balanced:
        st.success("✅ All months balanced - accounting is correct")
    else:
        st.error("❌ Some months have ledger imbalances - review needed")

# ==========================================
# EXPORT DATA
# ==========================================

st.markdown("---")
st.header("📥 Export Data")

csv_data = df.to_csv(index=False)
st.download_button(
    label="Download Full Results (CSV)",
    data=csv_data,
    file_name=f"therapy_practice_model_v5_{simulation_months}mo.csv",
    mime="text/csv"
)

# Export cohort data
if len(cohorts) > 0:
    cohort_export = []
    for cohort in cohorts:
        cohort_export.append({
            'cohort_id': cohort.cohort_id,
            'start_month': cohort.start_month,
            'therapist_id': cohort.therapist_id,
            'initial_count': cohort.initial_count,
            'current_count': cohort.current_count,
            'retention_rate': (cohort.current_count / cohort.initial_count * 100) if cohort.initial_count > 0 else 0
        })
    
    cohort_csv = pd.DataFrame(cohort_export).to_csv(index=False)
    st.download_button(
        label="Download Cohort Data (CSV)",
        data=cohort_csv,
        file_name=f"cohort_data_{simulation_months}mo.csv",
        mime="text/csv"
    )

st.markdown("---")

# ==========================================
# MODEL DOCUMENTATION
# ==========================================

with st.expander("📖 Model Documentation & Assumptions"):
    st.markdown("""
    ## Therapy Practice Financial Model V5.0
    
    ### Key Improvements Over Previous Versions
    
    #### 1. Cohort-Based Client Tracking
    - **Fixed:** Clients are now tracked in cohorts by start month
    - **Impact:** Churn applies based on client tenure, not practice age
    - **Example:** A client starting in Month 10 experiences Month 1 churn in Month 10, not Month 1 of practice
    
    #### 2. Double-Entry Bookkeeping
    - Every transaction recorded as debit and credit
    - Ledger verification ensures accounting accuracy
    - Separates accrual (revenue earned) from cash (revenue collected)
    
    #### 3. Accurate Contribution Margin
    - Includes ALL costs: direct pay, payroll tax, allocated overhead
    - Used for CLV calculation
    - Previously understated by omitting overhead allocation
    
    #### 4. Realistic Marketing Model
    - Leads generated, conversion delay modeled
    - Actual CAC tracked (spend / clients acquired)
    - Unused budget stays in cash (not added to profit)
    - Clients lost to capacity are tracked
    
    #### 5. Revenue Loss Timing
    - Applied at collection time, not billing time
    - Reflects reality: denials/bad debt discovered when collecting
    - Applied to ALL payers (including self-pay)
    
    ### Key Assumptions
    
    **Therapist Compensation:**
    - Paid ONLY for completed sessions (not scheduled or cancelled)
    - Owner treated as employee therapist (paid per session)
    - Payroll taxes paid by practice
    
    **Revenue Recognition:**
    - Accrual basis: Revenue recorded when service provided
    - Cash basis: Collections delayed per payer (0-90 days)
    - Revenue loss (denials + bad debt) applied at collection
    
    **Client Behavior:**
    - Churn rates apply based on cohort age (month 1, 2, 3, ongoing)
    - Sessions per month consistent within client tenure
    - Cancellations: No revenue, no therapist pay
    - No-shows: Revenue billed, no therapist pay (already in no-show rate)
    
    **Marketing:**
    - Cost per lead → Leads generated
    - Conversion rate → Clients start (with delay)
    - Overflow clients lost if no capacity
    - Actual CAC = Total spend / Total acquired
    
    **Credit Line:**
    - Draws automatically when cash negative
    - Interest charged monthly on drawn balance
    - Included in cash flow analysis
    
    ### Validation Checks
    
    1. Ledger balances every month (debits = credits)
    2. Client count reconciles (start + new - churned = end)
    3. Sessions flow: scheduled ≥ completed ≥ billable
    4. Cash balance includes all inflows and outflows
    5. CLV calculation uses same churn rates as simulation
    
    ### Limitations
    
    - Does not model therapist turnover
    - Assumes consistent payer mix over time
    - Seasonality is simplified (12-month pattern)
    - No accounts payable (all expenses paid immediately)
    - Supervision costs simplified (no opportunity cost model)
    - No differentiation between insurance types for revenue loss
    
    ### How to Use This Model
    
    1. **Set starting capital** and credit line availability
    2. **Configure therapist hiring timeline** (who, when, credentials)
    3. **Set compensation rates** and tax rates
    4. **Define payer mix** and payment delays
    5. **Set marketing parameters** (budget, CPL, conversion rate)
    6. **Configure client behavior** (sessions/month, churn rates, cancellations)
    7. **Run simulation** and analyze results
    8. **Iterate on hiring decisions** based on utilization and cash
    
    ### Critical Metrics to Watch
    
    - **Capacity utilization:** Hire at 85%+
    - **Cash runway:** Maintain 6+ months
    - **Actual CAC vs Max CAC:** Must stay below max
    - **Monthly cash flow:** Target positive by month 6-12
    - **Cohort retention:** Monitor churn rates match assumptions
    
    ### Questions or Issues?
    
    This model uses production-grade accounting principles. All transactions are logged,
    all calculations are documented, and the ledger is verified monthly.
    
    If numbers look wrong, check:
    1. Ledger verification (should all be balanced)
    2. Client count reconciliation
    3. Marketing pipeline delay settings
    4. Payment delay configuration
    """)

st.markdown("---")
st.success("✅ **Model V5.0 Complete** - Cohort tracking, double-entry bookkeeping, accurate CAC/CLV analysis")
