import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(ab_data_path=r'C:\Users\rouna\Downloads\MiniProject- AB test\ab_data.csv', countries_path=r'C:\Users\rouna\Downloads\MiniProject- AB test\countries.csv'):
    """Loads, merges, and cleans the A/B test datasets."""
    df_ab = pd.read_csv(ab_data_path)
    df_countries = pd.read_csv(countries_path)
    df = df_ab.merge(df_countries, on='user_id', how='inner')
    valid_sessions = ((df['group'] == 'control') & (df['landing_page'] == 'old_page')) | \
                     ((df['group'] == 'treatment') & (df['landing_page'] == 'new_page'))
    df = df[valid_sessions]
    df = df.drop_duplicates(subset='user_id', keep='first')
    return df
def perform_descriptive_stats(df):
    """Calculates sample sizes and conversion rates."""
    summary = df.groupby('group')['converted'].agg(['count', 'sum', 'mean']).reset_index()
    summary.rename(columns={'count': 'Users (n)', 'sum': 'Conversions', 'mean': 'Conversion Rate'}, inplace=True)
    summary['Conversion Rate'] = summary['Conversion Rate'] * 100
    return summary
def run_ttest(df):
    """Runs Welch's independent t-test."""
    control = df[df['group'] == 'control']['converted']
    treatment = df[df['group'] == 'treatment']['converted']
    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
    diff = treatment.mean() - control.mean()
    return t_stat, p_val, diff * 100
def run_chisquare(df):
    """Runs Chi-square test of independence."""
    contingency = pd.crosstab(df['group'], df['converted'])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    return chi2, p_val, contingency
def run_anova(df):
    """Runs One-way ANOVA for countries and country-strategy combinations."""
    # ANOVA by Country
    countries = df['country'].unique()
    country_groups = [df[df['country'] == c]['converted'] for c in countries]
    f_stat_country, p_val_country = stats.f_oneway(*country_groups)
    df['country_strategy'] = df['country'] + "_" + df['group']
    combos = df['country_strategy'].unique()
    combo_groups = [df[df['country_strategy'] == c]['converted'] for c in combos]
    f_stat_combo, p_val_combo = stats.f_oneway(*combo_groups)
    return f_stat_country, p_val_country, f_stat_combo, p_val_combo
def plot_visualizations(df):
    """Generates visualizations for the report."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='group', data=df, palette='muted')
    plt.title('Sample Size by Group (after cleaning)')
    plt.ylabel('Number of Users')
    for p in ax.patches:
        ax.annotate(format(p.get_height(), ','), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', xytext = (0, 5), 
                    textcoords = 'offset points', weight='bold')
    plt.show()
    plt.figure(figsize=(8, 5))
    ax2 = sns.barplot(x='group', y='converted', data=df, errorbar=('ci', 95), capsize=.1, palette='muted')
    plt.title('Conversion Rate by Pricing Strategy (95% CI)')
    plt.ylabel('Conversion Rate (%)')
    # Multiply ticks by 100 for percentage
    ticks = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.0f}'.format(x*100) for x in ticks])
    plt.show()
    plt.figure(figsize=(10, 6))
    ax3 = sns.barplot(x='country', y='converted', hue='group', data=df, errorbar=None, palette='muted')
    plt.title('Conversion Rate by Country and Strategy')
    plt.ylabel('Conversion Rate (%)')
    ticks = ax3.get_yticks()
    ax3.set_yticklabels(['{:,.0f}'.format(x*100) for x in ticks])
    plt.show()
if __name__ == "__main__":
    df_clean = load_and_clean_data()
    print("--- Descriptive Statistics ---")
    print(perform_descriptive_stats(df_clean))
    print("\n--- Welch's T-Test ---")
    t, p_t, diff = run_ttest(df_clean)
    print(f"T-statistic: {t:.3f}, P-value: {p_t:.3f}, Difference (pp): {diff:.3f}%")
    print("\n--- Chi-Square Test ---")
    chi, p_chi, xtab = run_chisquare(df_clean)
    print(f"Chi2: {chi:.3f}, P-value: {p_chi:.3f}")
    print("Contingency Table:\n", xtab)
    print("\n--- ANOVA ---")
    f_c, p_c, f_cs, p_cs = run_anova(df_clean)
    print(f"Country -> F-statistic: {f_c:.3f}, P-value: {p_c:.3f}")
    print(f"Country+Strategy -> F-statistic: {f_cs:.3f}, P-value: {p_cs:.3f}")
    plot_visualizations(df_clean)