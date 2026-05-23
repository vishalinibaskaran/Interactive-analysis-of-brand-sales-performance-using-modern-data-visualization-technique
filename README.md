# Multi-Brand Sales Trend Analyzer

## Overview

The **Multi-Brand Sales Trend Analyzer** is an interactive data analytics dashboard developed using **Python** and **Streamlit**.

This project analyzes sales performance across multiple cosmetic and skincare brands such as **L'Oréal, Dove, Lakmé, Pond’s, Nivea, Maybelline, and Garnier**.

The dashboard provides visual insights, sales forecasting, low-performing product analysis, and downloadable reports to support better business decisions.

## Features

- Multi-brand sales analysis
- Product-wise sales tracking
- Interactive data filtering
- Monthly sales trend visualization
- Sales comparison before and after boost
- Low-performing product identification
- Bundle recommendation suggestions
- Future sales forecasting using Linear Regression
- Downloadable Excel reports
- Interactive charts and dashboards


## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset Information

The project uses CSV datasets for multiple brands:

- loreal_sales.csv
- dove.csv
- lakme.csv
- ponds.csv
- nivea.csv
- maybelline.csv
- garnier.csv

Each dataset contains:

- Product Name
- Month
- Sales Data


## Project Structure

```plaintext
Multi-Brand-Sales-Analyzer/
│
├── PRODUCTS/
│   ├── loreal_sales.csv
│   ├── dove.csv
│   ├── lakme.csv
│   ├── ponds.csv
│   ├── nivea.csv
│   ├── maybelline.csv
│   └── garnier.csv
│
├── multibrand.py
├── README.md
```

## Dashboard Functionalities

### Sales Trend Analysis
Visualizes monthly sales performance for all products and brands.

### Boosted Sales Analysis
Compares original sales with boosted sales for low-performing products.

### Forecasting
Predicts future sales trends for the next 3 months using Linear Regression.

### Report Generation
Allows users to download filtered sales reports in Excel format.


## Sample Insights

- Identifies worst-selling months for products
- Detects products with sales below ₹1000
- Suggests bundle recommendations
- Shows percentage increase after sales boosting


## Future Improvements

- Add real-time database integration
- Deploy dashboard online
- Add authentication system
- Improve forecasting accuracy using advanced ML models
- Add more interactive visualizations


## Conclusion

This project demonstrates practical implementation of **data analytics, visualization, and machine learning concepts** through an interactive dashboard.

It helps businesses analyze sales trends, identify weak-performing products, and make better data-driven decisions.
