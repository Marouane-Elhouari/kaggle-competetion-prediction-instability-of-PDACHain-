# 🧬 Novozymes Enzyme Stability Prediction

## 🎯 Objective
The primary goal of this project is to predict the thermal stability of enzyme variants (mutations) based on their amino acid sequences and 3D structures. By developing a robust Machine Learning / Deep Learning model, we aim to accurately rank mutated proteins according to their change in melting temperature ($\Delta T_m$) compared to the wild-type (original) protein.

## 🧠 Context
Enzymes are essential proteins used in various industries, including laundry detergents, baking, and biofuels. However, industrial environments often require enzymes to function at high temperatures. 
Finding thermally stable mutations in the lab is extremely expensive and time-consuming. This Kaggle competition (hosted by Novozymes) challenges data scientists to computationally predict which mutations will increase or decrease an enzyme's stability, thereby accelerating the discovery of robust industrial enzymes.

## 📊 Dataset
The project utilizes data sourced from Kaggle and public scientific databases (such as Pucci et al. / S1626 and ThermoMutDB):
* **Wild-Type Structure:** The 3D PDB (Protein Data Bank) file of the original enzyme.
* **Mutations:** Thousands of rows detailing specific single or multiple point mutations (e.g., `M1A` meaning Methionine at position 1 mutated to Alanine).
* **Target Variable:** `dTm` ($\Delta T_m$) or `ddG` ($\Delta\Delta G$), representing the change in thermal stability.
* **Features:** 3D Voxel features, sequence-based embeddings, and physical/chemical properties of the amino acids.

## 📈 Evaluation
Models are evaluated using the **Spearman Correlation Coefficient** between the predicted stability rankings and the actual experimental measurements. The goal is to get the ranking order right, rather than predicting the exact physical value perfectly.

## 🛠️ Tech Stack & Methodology
* **Language:** Python
* **Data Processing:** Pandas, NumPy, Biopython, HTMD (for PDB parsing and Voxel generation).
* **Machine Learning:** 
  * Ensemble Models: XGBoost, LightGBM (trained on aggregated sequence and physical features).
  * Deep Learning: PyTorch (3D CNN architectures like ThermoNet to process 3D spatial voxel grids).
* **Environment:** Kaggle Notebooks / Local Jupyter / VS Code.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd <your-repo-name>
