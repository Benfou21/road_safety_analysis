# Road Safety Analysis in France

## Project Overview
This project analyzes road accident data in France (2023) to explore patterns, cluster similar accidents, and predict mortality risk for involved users. Leveraging data from French national traffic records, the pipeline includes data cleaning, exploratory visualization, unsupervised clustering, and supervised learning for mortality prediction.

## Key Features & Methods
- **Data Preprocessing**:  
  - Load raw CSV datasets: `caract`, `lieux`, `vehicules`, `usagers`.  
  - Clean and categorize fields (time-of-day, day-type, season, department region).  
  - Impute missing or aberrant values and drop irrelevant columns.  
- **Exploratory Data Analysis**:  
  - Accident counts by month.  
  - Severity distribution over time.  
  - Geographic breakdown by department category.  
  - Vehicle type distributions and user demographics.  
- **Clustering**:  
  - Merge datasets and one-hot encode categorical variables.  
  - Standardize numeric features.  
  - Perform K-Means clustering, select optimal **k** via silhouette score.  
  - Visualize cluster distributions and top discriminative variables with heatmaps.  
- **Mortality Prediction**:  
  - Build decision tree classifier on user-level data.  
  - Handle class imbalance with SMOTE oversampling.  
  - Evaluate with accuracy, precision, recall, F1-score, and confusion matrix.  
  - Display decision tree structure and feature importances.

## Repository Structure

ROAD_SAFETY_ANALYSIS/

├── data/

│ ├── raw/ # Original CSV files from traffic records

│ └── processed/ # Cleaned CSV outputs

├── notebooks/ # Jupyter notebooks

│ ├── 1_preprocessing.ipynb # Data loading & cleaning steps

│ ├── 2_visualization.ipynb # Exploratory plots & charts

│ ├── 3_clustering.ipynb # K-Means clustering experiments

│ └── 4_predict_death.ipynb # Mortality prediction pipeline

├── src/ # Python modules

│ ├── preprocessing.py # Cleaning functions & categorization

│ ├── features.py # Clustering preparation & plotting

│ └── utils.py # Helper utilities

├── requirements.txt # Python package dependencies

└── README.md # This file: overview and instructions


## Installation & Setup
1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/road_safety_analysis.git
   cd road_safety_analysis

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt


## Usage

1. **Preprocess data**
Run the preprocessing notebook or script:
jupyter notebook notebooks/1_preprocessing.ipynb

2. **Explore data**
Open notebooks/2_visualization.ipynb and execute cells to generate EDA charts.

3. **Run clustering**
Execute notebooks/3_clustering.ipynb to determine optimal clusters and visualize results.

4. **Train mortality model**
Execute notebooks/4_predict_death.ipynb for SMOTE oversampling, training, and evaluation.

## Results

### Clustering
- **Optimal number of clusters (k)**: silhouette analysis indicated **k = 2**, with clusters of meaningful size and separation.
- **Cluster 0**:  
  - High‐speed, wide roads outside urban centers (high vma, nbv, larrout)  
  - Low intersection rate → predominantly rural/interurban accidents  
- **Cluster 1**:  
  - Urban environments (high agg, complex intersections, lateral collisions)  
  - Strong representation in Paris and other dense agglomérations  
- **Additional segmentation (k = 4)** revealed further profiles:  
  1. Urban narrow residential roads  
  2. Standard urban roads at moderate speed  
  3. Fast interurban roads, few intersections  
  4. Structured urban thoroughfares (boulevards, ring roads)  

### Mortality Prediction
- **Model & Evaluation**: Decision tree with SMOTE oversampling achieved  
  - **Overall accuracy**: ~66%  
  
- **Key predictors**:  
  1. **Seatbelt usage (secu1)** – unbelted users far more at risk  
  2. **Sex** – male users exhibited slightly higher mortality  
  3. **User category (catu)** – pedestrians and passengers showed distinct risk patterns  


