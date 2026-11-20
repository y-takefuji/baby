import re
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ------------------------------------------------------------------
# 1. Load and prepare the data
# ------------------------------------------------------------------
df = pd.read_csv('data.csv')
df.columns = df.columns.str.strip()          # 'Estimate ' -> 'Estimate'

# Parse timeframe, e.g. "09/24/2023-10/28/2023"
df[['Start', 'End']] = df['Timeframe'].str.split('-', expand=True)
df['Start'] = pd.to_datetime(df['Start'].str.strip(), format='%m/%d/%Y')
df['End'] = pd.to_datetime(df['End'].str.strip(), format='%m/%d/%Y')

# Time-frame order (1 = earliest, 6 = latest)
frames = sorted(df['Start'].unique())
order_map = {start: i + 1 for i, start in enumerate(frames)}
df['Period'] = df['Start'].map(order_map)

# Split the CI string "16.1 - 27.4" into lower and upper limits
ci = df['95% Confidence Interval'].str.extract(r'([\d.]+)\s*-\s*([\d.]+)').astype(float)
df['CI_low'] = ci[0]
df['CI_high'] = ci[1]

# Standard error from the 95% CI and inverse-variance weight
df['SE'] = (df['CI_high'] - df['CI_low']) / (2 * 1.96)
df['Weight'] = 1 / df['SE'] ** 2

# ------------------------------------------------------------------
# 2. Fixed display order
# ------------------------------------------------------------------
cohort_order = ['Those with a baby <8 months', 'Currently pregnant', 'Trying to get pregnant']
category_order = [
    'Baby got nirsevimab',
    'Definitely will get nirsevimab for baby',
    'Probably will get nirsevimab for baby or unsure',
    'Probably or definitely will not get nirsevimab for baby',
]

# ------------------------------------------------------------------
# 3. Weighted linear trend for each cohort and category
# ------------------------------------------------------------------
rows = []
for cohort in cohort_order:
    for category in category_order:
        sub = df[(df['Indicator Variable'] == cohort) &
                 (df['Indicator Category'] == category)].sort_values('Period')
        if len(sub) < 3:                      # not enough points (or category absent)
            continue

        X = sm.add_constant(sub['Period'].astype(float))
        model = sm.WLS(sub['Estimate'], X, weights=sub['Weight']).fit()

        slope = model.params['Period']
        low, high = model.conf_int(alpha=0.05).loc['Period']
        p_value = model.pvalues['Period']

        rows.append({
            'Cohort': cohort,
            'Category': category,
            'Number of time frames': len(sub),
            'First estimate (%)': round(sub['Estimate'].iloc[0], 1),
            'Last estimate (%)': round(sub['Estimate'].iloc[-1], 1),
            'Slope (percentage points per time frame)': round(slope, 2),
            '95% CI lower': round(low, 2),
            '95% CI upper': round(high, 2),
            'p-value': round(p_value, 3),
        })

table1 = pd.DataFrame(rows)

# Combined CI column for the manuscript
table1['95% CI'] = (table1['95% CI lower'].map('{:.2f}'.format) + ' to ' +
                    table1['95% CI upper'].map('{:.2f}'.format))

manuscript_table = table1[['Cohort', 'Category',
                           'Slope (percentage points per time frame)',
                           '95% CI', 'p-value']]

# ------------------------------------------------------------------
# 4. Show and save
# ------------------------------------------------------------------
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 60)
print(manuscript_table.to_string(index=False))

manuscript_table.to_csv('Table1_trend_analysis.csv', index=False)
table1.to_csv('Table1_trend_analysis_full.csv', index=False)

# Optional: Excel output
# manuscript_table.to_excel('Table1_trend_analysis.xlsx', index=False)

# ------------------------------------------------------------------
# 5. Optional: sample size range per cohort (do NOT sum across categories)
# ------------------------------------------------------------------
n_per_frame = (df.groupby(['Indicator Variable', 'Timeframe'])['Sample Size']
                 .first().reset_index())
print("\nSample size range per cohort (one value per time frame):")
print(n_per_frame.groupby('Indicator Variable')['Sample Size']
                 .agg(['min', 'max']))