
# Iris Species Classification Dashboard

This interactive dashboard was developed in Streamlit. The project was first built in Google Colab, later organized into a GitHub repository, and finally deployed in Streamlit in a interactive dashboard

# Description
This project is part of the Machine Learning Classification activity.  
The objective is to analyze the Iris dataset, train a classification model, visualize the data, and allow the user to predict the species of an Iris flower based on four measurements.

# Workflow / Methodology
The project followed a simple and clear end-to-end process:

1. Data Understanding:
   Initial exploration of the Iris dataset, checking sample size, class balance, and relationships between features using basic visualizations.

2. Preprocessing:  
   Removal of the *Id* column and division of the dataset into training and testing sets.  
   A `StandardScaler` was applied to normalize the numerical features.

3. Model Training:  
   A Random Forest Classifier was selected because it performs well on small datasets, handles non-linear patterns, and works reliably without complex tuning.

4. Evaluation:  
   Model performance was measured using Accuracy, Precision, Recall, and F1-Score.

5. Dashboard Development:  
   A Streamlit dashboard was built to display dataset visualizations, model metrics, a prediction panel, and a 3D scatter plot showing the position of the predicted sample.

# Features
- Load of the Iris.csv dataset  
- Automatic training of a Random Forest model  
- Performance metrics: Accuracy, Precision, Recall, F1-Score  
- Visualizations:
  - 3D scatter plot of the dataset  
  - Highlighted position of the predicted sample  
- Manual prediction panel for sepal/petal measurements  

# Technologies Used
- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Plotly  
- Streamlit  

# Running the app:
pip install -r requirements.txt
streamlit run app.py


## Team
Members: Juan Carrascal (10032) - Karl Luckert Heinz (10919) - Karolaith Orozco (10895) - Jhonatan Monterroza (10895)

## Links
GitHub Repository: https://github.com/jcarrasc7/REPO-DE-Prueba.git  
Streamlit Dashboard: https://repo-de-prueba-l923sekxmy6ur7mfr9zmys.streamlit.app/

