import pandas as pd
import numpy as np

np.random.seed(42)

states = [
    'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
    'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
    'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT'
]

state_cost_multiplier = {
    'CA': 1.35, 'NY': 1.40, 'MA': 1.30, 'NJ': 1.28, 'CT': 1.25,
    'TX': 1.10, 'FL': 1.08, 'IL': 1.12, 'WA': 1.20, 'CO': 1.15,
    'PA': 1.05, 'OH': 1.00, 'MI': 1.02, 'GA': 0.95, 'NC': 0.93,
    'VA': 1.05, 'AZ': 1.03, 'TN': 0.90, 'IN': 0.92, 'MO': 0.91,
    'MD': 1.18, 'WI': 0.98, 'MN': 1.08, 'SC': 0.88, 'AL': 0.85,
    'LA': 0.87, 'KY': 0.89, 'OR': 1.10, 'OK': 0.86, 'UT': 1.00
}

drg_groups = {
    'Cardiovascular': [
        'HEART FAILURE & SHOCK', 'CARDIAC ARRHYTHMIA', 'CORONARY BYPASS',
        'PERCUTANEOUS CARDIOVASCULAR PROC', 'CARDIAC CATHETERIZATION'
    ],
    'Respiratory': [
        'SIMPLE PNEUMONIA & PLEURISY', 'CHRONIC OBSTRUCTIVE PULMONARY',
        'RESPIRATORY INFECTIONS', 'PULMONARY EDEMA'
    ],
    'Orthopedic': [
        'MAJOR JOINT REPLACEMENT - LOWER EXTREMITY', 'SPINAL FUSION',
        'HIP & FEMUR PROCEDURES', 'KNEE PROCEDURES'
    ],
    'Neurological': [
        'STROKE', 'SEIZURES', 'INTRACRANIAL HEMORRHAGE',
        'TRANSIENT ISCHEMIA'
    ],
    'Digestive': [
        'DIGESTIVE MALIGNANCY', 'GI HEMORRHAGE', 'PANCREATIC DISORDERS',
        'INFLAMMATORY BOWEL DISEASE'
    ],
    'Renal': [
        'KIDNEY & URINARY TRACT INFECTIONS', 'RENAL FAILURE',
        'KIDNEY TRANSPLANT'
    ],
    'Oncology': [
        'LYMPHOMA & LEUKEMIA', 'CHEMOTHERAPY', 'RADIATION THERAPY',
        'BREAST PROCEDURES'
    ]
}

drg_base_cost = {
    'Cardiovascular': 18000, 'Respiratory': 9000, 'Orthopedic': 28000,
    'Neurological': 15000, 'Digestive': 12000, 'Renal': 11000, 'Oncology': 22000
}

n = 3000
rows = []

provider_names = [f"Provider_{str(i).zfill(4)}" for i in range(1, 301)]

for _ in range(n):
    state = np.random.choice(states)
    mult  = state_cost_multiplier[state]
    group = np.random.choice(list(drg_groups.keys()),
                              p=[0.25, 0.15, 0.18, 0.12, 0.12, 0.10, 0.08])
    drg   = np.random.choice(drg_groups[group])
    base  = drg_base_cost[group]

    discharges = int(np.random.exponential(120)) + 10

    covered    = base * mult * np.random.uniform(0.85, 1.20)
    total_pay  = covered * np.random.uniform(0.28, 0.42)
    medicare   = total_pay * np.random.uniform(0.80, 0.95)

    rows.append({
        'provider_id':               np.random.choice(provider_names),
        'provider_state':            state,
        'drg_definition':            drg,
        'drg_category':              group,
        'total_discharges':          discharges,
        'average_covered_charges':   round(covered, 2),
        'average_total_payments':    round(total_pay, 2),
        'average_medicare_payments': round(medicare, 2),
        'cost_to_charge_ratio':      round(total_pay / covered, 4)
    })

df = pd.DataFrame(rows)
df.to_csv('cms_inpatient.csv', index=False)
print(f"Dataset generated: {len(df)} records")
print(df.head())
print(f"\nCategory distribution:\n{df['drg_category'].value_counts()}")
print(f"\nAvg total payment: ${df['average_total_payments'].mean():,.0f}")
