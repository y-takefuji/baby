import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------------------------------------------------
# Load the data
# ------------------------------------------------------------------
df = pd.read_csv('data.csv')

# The CSV header contains 'Estimate ' with a trailing space; remove stray spaces
df.columns = df.columns.str.strip()

# ------------------------------------------------------------------
# Parse 'Timeframe' (e.g., "09/24/2023-10/28/2023") into start and end dates
# ------------------------------------------------------------------
df[['Start', 'End']] = df['Timeframe'].str.split('-', expand=True)
df['Start'] = pd.to_datetime(df['Start'].str.strip(), format='%m/%d/%Y')
df['End'] = pd.to_datetime(df['End'].str.strip(), format='%m/%d/%Y')

# Midpoint of each time frame (X-axis position)
df['Midpoint'] = df['Start'] + (df['End'] - df['Start']) / 2

# Sort by time order
df = df.sort_values('Midpoint')

# ------------------------------------------------------------------
# Let the user select an 'Indicator Variable'
# ------------------------------------------------------------------
indicator_variables = df['Indicator Variable'].unique()
print("Unique Indicator Variables:")
for i, indicator in enumerate(indicator_variables):
    print(f"{i + 1}. {indicator}")

while True:
    try:
        selected_indicator = int(input("Select an Indicator Variable by number: ")) - 1
        if 0 <= selected_indicator < len(indicator_variables):
            break
    except ValueError:
        pass
    print(f"Please enter a number between 1 and {len(indicator_variables)}.")

selected_name = indicator_variables[selected_indicator]
filtered_df = df[df['Indicator Variable'] == selected_name]

# ------------------------------------------------------------------
# Fixed category order so that each category always has the same style
# ------------------------------------------------------------------
category_order = [
    'Baby got nirsevimab',
    'Definitely will get nirsevimab for baby',
    'Probably will get nirsevimab for baby or unsure',
    'Probably or definitely will not get nirsevimab for baby',
]

# Fixed style per category (independent of the cohort or row order)
line_styles = ['-', '--', '-.', ':']   # 4 different line styles
line_widths = [1, 2, 1, 2]             # 2 different widths (1, 2)
style_map = {
    cat: (line_styles[i], line_widths[i]) for i, cat in enumerate(category_order)
}

# Categories present in the selected cohort, in the fixed order
present = filtered_df['Indicator Category'].unique()
categories = [c for c in category_order if c in present]

# Any unexpected category is appended after the fixed ones
extra = [c for c in present if c not in category_order]
for j, c in enumerate(extra):
    style_map[c] = (line_styles[(len(category_order) + j) % 4], 1)
categories += extra

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

for category in categories:
    plot_df = filtered_df[filtered_df['Indicator Category'] == category]
    linestyle, linewidth = style_map[category]
    ax.plot(
        plot_df['Midpoint'],
        plot_df['Estimate'],
        label=category,
        color='black',
        linestyle=linestyle,
        linewidth=linewidth,
    )

# Axis formatting
ax.set_ylabel('Estimate (%)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
fig.autofmt_xdate()

# Shrink the axis height by 10% on the top to make room for the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])

# Legend above the plot, one column (multiple rows)
ax.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=1)

# Label under the X-axis
ax.set_xlabel(f"Graph for '{selected_name}'")

# Save the figure
filename = re.sub(r'\W+', '_', selected_name) + '.png'
fig.savefig(filename, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()