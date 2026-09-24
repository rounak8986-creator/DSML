import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean visual theme for all charts
sns.set_theme(style="whitegrid")

print("Loading and cleaning data...")

cases = pd.read_csv(
    "https://raw.githubusercontent.com/imdevskp/covid-19-india-data/master/complete.csv"
)
vax = pd.read_csv(
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/India.csv"
)

cases.columns = [
    "date", "state", "lat", "long", "confirmed",
    "deaths", "cured", "new_cases", "new_deaths", "new_recovered",
]
cases["date"] = pd.to_datetime(cases["date"])
cases["deaths"] = pd.to_numeric(cases["deaths"], errors="coerce").fillna(0).astype(int)

name_fix = {
    "Telangana***": "Telangana",
    "Telengana": "Telangana",
    "Union Territory of Jammu and Kashmir": "Jammu and Kashmir",
    "Union Territory of Ladakh": "Ladakh",
    "Union Territory of Chandigarh": "Chandigarh",
}
cases["state"] = cases["state"].replace(name_fix)
cases = cases.drop_duplicates(subset=["date", "state"])
cases["active"] = cases["confirmed"] - cases["deaths"] - cases["cured"]


national = cases.groupby("date", as_index=False)[["confirmed", "deaths", "cured", "new_cases"]].sum()
national["recovery_rate"] = (national["cured"] / national["confirmed"] * 100).round(2)
national["death_rate"] = (national["deaths"] / national["confirmed"] * 100).round(2)

latest = cases[cases["date"] == cases["date"].max()].copy()
latest = latest.sort_values("confirmed", ascending=False)
latest["recovery_rate"] = (latest["cured"] / latest["confirmed"] * 100)
latest["death_rate"] = (latest["deaths"] / latest["confirmed"] * 100)

cases["month"] = cases["date"].dt.strftime("%Y-%m")

print("Data ready! Generating charts...")

plt.figure(figsize=(10, 6))
plt.plot(national["date"], national["confirmed"], label="Confirmed", color="#2c5c9c")
plt.plot(national["date"], national["cured"], label="Recovered", color="#2c9c5c")
plt.plot(national["date"], national["deaths"], label="Deaths", color="#c03c3c")
plt.title("India: Cumulative COVID-19 Cases Over Time")
plt.ylabel("People")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(national["date"], national["new_cases"], color="#4c7cbc")
plt.title("India: Daily New Confirmed Cases")
plt.ylabel("New cases")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

top_10 = latest.head(10)
plt.figure(figsize=(10, 6))
# Fixed: Added hue="state" and legend=False
sns.barplot(data=top_10, x="confirmed", y="state", hue="state", palette="Blues_r", legend=False)
plt.title(f"Top 10 States by Total Confirmed Cases (as of {latest['date'].max().date()})")
plt.xlabel("Confirmed cases")
plt.ylabel("")
plt.tight_layout()
plt.show()

top_states = top_10["state"].tolist()
latest["state_grouped"] = latest["state"].apply(lambda x: x if x in top_states else "Rest of India")
pie_data = latest.groupby("state_grouped")["confirmed"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 8))

plt.pie(
    pie_data, labels=pie_data.index, autopct='%1.1f%%', 
    colors=sns.color_palette("Blues_r", len(pie_data)), startangle=90
)
plt.title("Share of Total Confirmed Cases by State")
plt.tight_layout()
plt.show()

top_5_states = latest.head(5)["state"].tolist()
top_5_data = cases[cases["state"].isin(top_5_states)]

plt.figure(figsize=(10, 6))
sns.lineplot(data=top_5_data, x="date", y="confirmed", hue="state", linewidth=2)
plt.title("Confirmed Cases Over Time — Top 5 States")
plt.xlabel("Date")
plt.ylabel("Confirmed cases")
plt.legend()
plt.tight_layout()
plt.show()

top_10_sorted = top_10.sort_values("recovery_rate", ascending=True)
plt.figure(figsize=(10, 6))
# Fixed: Added hue="state" and legend=False
sns.barplot(data=top_10_sorted, x="recovery_rate", y="state", hue="state", palette="Greens", legend=False)
plt.title("Recovery Rate (%) — Top 10 Worst-Hit States")
plt.xlabel("Recovery rate (%)")
plt.ylabel("")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(national["date"], national["recovery_rate"], color="#2c9c5c", linewidth=2)
plt.title("India: National Recovery Rate Over Time")
plt.ylabel("Recovery rate (%)")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

bubble_data = latest[latest["confirmed"] > 1000] 
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=bubble_data, x="recovery_rate", y="death_rate",
    size="confirmed", sizes=(50, 1500), alpha=0.7, palette="Blues_r", hue="confirmed", legend=False
)
for i in range(5):
    plt.text(
        bubble_data["recovery_rate"].iloc[i] + 1, 
        bubble_data["death_rate"].iloc[i], 
        bubble_data["state"].iloc[i], 
        fontsize=9
    )
plt.title("Recovery Rate vs Death Rate by State (bubble = case count)")
plt.xlabel("Recovery rate (%)")
plt.ylabel("Death rate (%)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
corr_matrix = national[["confirmed", "deaths", "cured", "new_cases", "recovery_rate", "death_rate"]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Reds", vmin=-0.2, vmax=1)
plt.title("Correlation Between National COVID Metrics")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=cases, x="month", y="new_cases", color="#3b75af")
plt.title("Spread of Daily New Cases by Month")
plt.xlabel("Month")
plt.ylabel("Daily new cases")
plt.tight_layout()
plt.show()

bottom_10 = latest.sort_values("confirmed", ascending=True).head(10)
plt.figure(figsize=(10, 6))
# Fixed: Added hue="state" and legend=False
sns.barplot(data=bottom_10, x="confirmed", y="state", hue="state", palette="Oranges", legend=False)
plt.title(f"10 Least-Affected States/UTs by Confirmed Cases (as of {latest['date'].max().date()})")
plt.xlabel("Confirmed cases")
plt.ylabel("")
plt.tight_layout()
plt.show()

vax["date"] = pd.to_datetime(vax["date"])
plt.figure(figsize=(10, 6))
plt.plot(vax["date"], vax["total_vaccinations"] / 1e9, label="Total doses (Bn)", color="#6c3483", linewidth=2)
plt.plot(vax["date"], vax["people_fully_vaccinated"] / 1e9, label="Fully vaccinated (Bn)", color="#117a65", linewidth=2)
plt.title("India: COVID-19 Vaccination Progress (2021-2024)")
plt.ylabel("People / doses (in billions)")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.show()