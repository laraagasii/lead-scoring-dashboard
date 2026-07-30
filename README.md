# Lead Scoring Dashboard - Big Data & Machine Learning 📊🎯

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://lead-scoring-dashboard-btakb7phraysvqjn8mfx9b.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)

A data-driven web application built with **Streamlit** to predict and score potential leads. By leveraging Machine Learning, this project helps businesses prioritize high-quality prospects, optimizing the sales funnel and improving conversion rates.

## ✨ Key Features
*   **Interactive Dashboard:** A user-friendly interface built with Streamlit for analyzing lead data.
*   **Predictive Lead Scoring:** Utilizes a pre-trained Machine Learning model (`.joblib`) to calculate the probability of a lead converting.
*   **Real-time Inference:** Users can input lead characteristics and get instant predictions.
*   **Data Visualization:** (Optional: Add if your dashboard includes charts/graphs of the data).

## 🛠️ Tech Stack
*   **Frontend/UI:** Streamlit
*   **Backend & ML:** Python, Scikit-Learn, Pandas, NumPy
*   **Deployment:** Streamlit Community Cloud

## 📂 Project Structure
```text
Proyek Big Data - Lead Scoring/
├── app.py                            # Main Streamlit application script
├── bikin_model.py                    # Script used for data preprocessing and model training
├── best_lead_scoring_model.joblib    # The saved pre-trained machine learning model
└── Lead Scoring.csv                  # The dataset used for training and testing
