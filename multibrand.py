import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Multi-Brand Sales Dashboard", layout="wide")
st.title("📈 Multi-Brand Sales Trend Analyzer")
st.markdown("---")

# --- Load All Brand CSVs Automatically ---
brand_files = {
    "loreal_sales.csv": "L'Oréal",
    "dove.csv": "Dove",
    "lakme.csv": "Lakmé",
    "ponds.csv": "Pond's",
    "nivea.csv": "Nivea",
    "maybelline.csv": "Maybelline",
    "garnier.csv": "Garnier"
}

dataframes = []
base_path = os.path.dirname(__file__)
products_path = os.path.join(base_path, "PRODUCTS")

for file, brand in brand_files.items():
    file_path = os.path.join(products_path, file)
    try:
        df = pd.read_csv(file_path)
        df["Brand"] = brand
        dataframes.append(df)
    except Exception as e:
        st.error(f"Could not load {file}: {e}")

if not dataframes:
    st.error("No data files loaded. Please check your PRODUCTS folder.")
    st.stop()

# Combine all brand DataFrames
df_full = pd.concat(dataframes, ignore_index=True)

# --- Data Preprocessing ---
month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
             'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
df_full["Month_Num"] = df_full["Month"].map(month_map)

overall_avg_sales = df_full.groupby("Product")["Sales"].mean()
overall_low_products = overall_avg_sales[overall_avg_sales < 1000].index.tolist()

df_full["Boosted_Sales"] = df_full.apply(
    lambda row: row["Sales"] * 1.15 if row["Product"] in overall_low_products else row["Sales"],
    axis=1
)
df_full["Growth_Rate"] = df_full.groupby("Product")["Sales"].pct_change().round(2) * 100

# --- Sidebar Filters ---
st.sidebar.header("📊 Filter Options")
unique_brands = sorted(df_full["Brand"].unique())
selected_brand = st.sidebar.selectbox("Select Brand", ["All"] + unique_brands)

if selected_brand != "All":
    df = df_full[df_full["Brand"] == selected_brand].copy()
    st.header(f"✨ Analysis for {selected_brand}")
else:
    df = df_full.copy()
    st.header("✨ Analysis for All Brands")

unique_products_for_brand = sorted(df["Product"].unique())
selected_product = st.sidebar.selectbox("Select Product", ["All"] + unique_products_for_brand)

if selected_product != "All":
    df = df[df["Product"] == selected_product]
    st.subheader(f"Detailed View for {selected_product}")

st.markdown("---")
st.subheader("📊 Monthly Sales Data")
st.dataframe(df)

st.subheader("🔻 Worst-Selling Month per Product")
if not df.empty:
    for p in df["Product"].unique():
        product_data = df[df["Product"] == p]
        if not product_data.empty:
            worst_row = product_data.loc[product_data["Sales"].idxmin()]
            st.write(f"- **{p}**: {worst_row['Month']} (₹{worst_row['Sales']:.2f})")
else:
    st.info("No data available for the selected filters.")

st.subheader("🤝 Bundle Suggestions")
if not df.empty:
    low_products_in_view = [p for p in overall_low_products if p in df["Product"].unique()]
    if low_products_in_view:
        for p in low_products_in_view:
            st.write(f"- **{p}**: Consider bundling with related popular items like Hair Shampoo or Face Cream to boost sales.")
    else:
        st.info("No low-performing products identified in this view for bundle suggestions.")
else:
    st.info("No data available for bundle suggestions.")

st.subheader("📈 Sales Trend – Before Boost")
fig1, ax1 = plt.subplots(figsize=(10, 5))
if not df.empty:
    sns.lineplot(data=df, x="Month_Num", y="Sales", hue="Product", marker="o", ax=ax1)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_map.keys())
    ax1.set_ylabel("Sales (₹)")
    ax1.set_xlabel("Month")
    ax1.set_title(f"Sales Trend Before Boost for {selected_brand if selected_brand != 'All' else 'All Brands'}")
    st.pyplot(fig1)
else:
    st.info("No data to display sales trend before boost.")

st.subheader("🚀 Sales Trend – After Boost")
fig2, ax2 = plt.subplots(figsize=(10, 5))
if not df.empty:
    sns.lineplot(data=df, x="Month_Num", y="Boosted_Sales", hue="Product", marker="o", ax=ax2)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_map.keys())
    ax2.set_ylabel("Boosted Sales (₹)")
    ax2.set_xlabel("Month")
    ax2.set_title(f"Sales Trend After Boost for {selected_brand if selected_brand != 'All' else 'All Brands'}")
    st.pyplot(fig2)
else:
    st.info("No data to display sales trend after boost.")

st.subheader("📉 Sales Comparison: Before vs After Boost")
fig3, ax3 = plt.subplots(figsize=(12, 6))
if not df.empty:
    for p in df["Product"].unique():
        temp = df[df["Product"] == p]
        if not temp.empty:
            ax3.plot(temp["Month_Num"], temp["Sales"], linestyle='--', label=f"{p} - Before")
            ax3.plot(temp["Month_Num"], temp["Boosted_Sales"], marker='o', label=f"{p} - After")
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(month_map.keys())
    ax3.set_ylabel("Sales (₹)")
    ax3.set_xlabel("Month")
    ax3.set_title(f"Sales Comparison for {selected_brand if selected_brand != 'All' else 'All Brands'}")
    ax3.legend()
    st.pyplot(fig3)
else:
    st.info("No data to display sales comparison.")

st.subheader("📈 Sales Stats Summary")
if not df.empty:
    total_before = df["Sales"].sum()
    total_after = df["Boosted_Sales"].sum()
    percent_increase = ((total_after - total_before) / total_before) * 100 if total_before != 0 else 0

    st.write(f"- **Total Before Boost**: ₹{total_before:.2f}")
    st.write(f"- **Total After Boost**: ₹{total_after:.2f}")
    st.write(f"- **Total Increase**: ₹{total_after - total_before:.2f} ({percent_increase:.2f}%)")
else:
    st.info("No data for sales summary.")

st.subheader("📊 Average Sales Per Product")
if not df.empty:
    avg_compare = df.groupby("Product")[["Sales", "Boosted_Sales"]].mean().round(2)
    st.dataframe(avg_compare)
else:
    st.info("No data for average sales per product.")

st.subheader("⚠️ Months with Sales < ₹1000")
if not df.empty:
    for p in df["Product"].unique():
        low_count = df[(df["Product"] == p) & (df["Sales"] < 1000)].shape[0]
        if low_count > 0:
            st.write(f"- {p}: {low_count} month(s) with sales below ₹1000")
    if df[(df["Sales"] < 1000)].empty:
        st.info("No months with sales below ₹1000 for the selected view.")
else:
    st.info("No data to check for low sales months.")

st.subheader("🔮 Forecast for Next 3 Months (Boosted Sales)")
if not df.empty:
    for p in df["Product"].unique():
        pdata = df[df["Product"] == p].sort_values(by="Month_Num")
        if len(pdata) > 1:
            X = pdata[["Month_Num"]]
            y = pdata["Boosted_Sales"]
            model = LinearRegression()
            model.fit(X, y)
            last_month_num = X["Month_Num"].max()
            future_months = np.array([[last_month_num + 1], [last_month_num + 2], [last_month_num + 3]])
            predictions = model.predict(future_months)

            st.markdown(f"**{p}** (Forecast from month {int(last_month_num + 1)} onwards):")
            forecast_month_labels = [
                list(month_map.keys())[(last_month_num % 12)] + "+",
                list(month_map.keys())[((last_month_num + 1) % 12)] + "+",
                list(month_map.keys())[((last_month_num + 2) % 12)] + "+"
            ]
            for m_label, val in zip(forecast_month_labels, predictions):
                st.write(f"   {m_label}: ₹{val:.2f}")
        else:
            st.info(f"Not enough data to forecast for {p}. Need at least 2 months of data.")
else:
    st.info("No data available for sales forecasting.")

st.markdown("---")
st.subheader("📥 Download Report")
try:
    excel_filename = f"{selected_brand.lower().replace(' ', '_')}_sales_report.xlsx" if selected_brand != "All" else "all_brands_sales_report.xlsx"
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Sales Data", index=False)
        if not df.empty:
            avg_compare = df.groupby("Product")[["Sales", "Boosted_Sales"]].mean().round(2)
            avg_compare.to_excel(writer, sheet_name="Average Sales", index=True)
            df[df["Sales"] < 1000].to_excel(writer, sheet_name="Low Sales Months", index=False)
    output.seek(0)
    st.download_button(
        label="Download Current View as Excel Report",
        data=output,
        file_name=excel_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except Exception as e:
    st.error(f"An error occurred while generating the Excel report: {e}")
