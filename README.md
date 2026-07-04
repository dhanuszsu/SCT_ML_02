# Customer Segmentation using K-Means Clustering

## 📌 Project Overview
This project applies **K-Means Clustering**, an unsupervised machine learning
algorithm, to segment mall customers into distinct groups based on their
**Annual Income** and **Spending Score**. These segments can help a business
target marketing campaigns more effectively.

**Task:** SCT_ML_02
**Type:** Unsupervised Learning / Clustering

## 🎯 Objective
- Explore and understand the Mall Customers dataset.
- Use the Elbow Method to determine the optimal number of clusters.
- Train a K-Means model to group customers into segments.
- Visualize and interpret the resulting customer segments.
- Translate clusters into actionable business insights.

## 📊 Dataset
- **Name:** Mall_Customers.csv
- **Source:** [Mall Customer Segmentation Data](https://github.com/SteffiPeTaffy/machineLearningAZ/blob/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv)
- **Columns:**
  - `CustomerID` – unique identifier
  - `Gender` – Male / Female
  - `Age` – customer age
  - `Annual Income (k$)` – yearly income in thousands of dollars
  - `Spending Score (1-100)` – score assigned based on spending behavior

## 🛠️ Libraries Used
| Library | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib | Plotting charts |
| seaborn | Statistical visualizations |
| scikit-learn | K-Means clustering algorithm |

## 🗂️ Project Structure
```
SCT_ML_02/
│
├── data/
│      Mall_Customers.csv
│
├── images/              # all generated charts
├── notebooks/           # optional exploratory Jupyter notebook
├── src/                 # optional helper modules
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

## ⚙️ Methodology / Workflow
1. **Data Loading** – Read the CSV file with pandas.
2. **Exploratory Data Analysis (EDA)** – Shape, info, describe, missing
   values, duplicates.
3. **Data Visualization** – Histograms, count plots, scatter plots,
   pairplot, and correlation heatmap.
4. **Elbow Method** – Determine the optimal number of clusters (k) by
   plotting WCSS (inertia) against k.
5. **Model Training** – Fit a `KMeans` model with the chosen k.
6. **Cluster Visualization** – Plot customer segments with centroids.
7. **Business Insights** – Interpret each cluster's income/spending profile.

## 📈 Results
Five customer segments were identified based on Annual Income and
Spending Score:

| Cluster | Profile | Suggested Strategy |
|---|---|---|
| 1 | Low income, low spending | Budget-friendly offers |
| 2 | Low income, high spending | Loyalty rewards, financing options |
| 3 | Average income, average spending | Standard marketing |
| 4 | High income, low spending | Targeted premium promotions to increase engagement |
| 5 | High income, high spending | VIP treatment, exclusive/premium products |

## 🖼️ Visualizations
All charts are saved automatically to the `images/` folder when you run
`main.py`, including:
- `age_distribution.png`
- `gender_count.png`
- `income_vs_spending.png`
- `pairplot.png`
- `correlation_heatmap.png`
- `elbow_method.png`
- `customer_clusters.png`

*(Add screenshots here after running the project.)*

## ▶️ How to Run
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/SCT_ML_02.git
cd SCT_ML_02

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place Mall_Customers.csv inside the data/ folder

# 5. Run the script
python main.py
```

## 🔮 Future Improvements
- Add more features (e.g., Age) to the clustering for 3D segmentation.
- Compare K-Means with other algorithms (DBSCAN, Hierarchical Clustering).
- Build an interactive dashboard (Streamlit/Plotly) for cluster exploration.
- Deploy the model as a small web app for live customer scoring.

## 👤 Author
**\<Your Name\>**
GitHub: [@your-username](https://github.com/your-username)

## 📄 License
This project is licensed under the MIT License.
