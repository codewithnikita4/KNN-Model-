# KNN-Model-
🌸 KNN Iris Classification Model
📖 Introduction
This project uses the K-Nearest Neighbors (KNN) algorithm to classify flowers from the famous Iris dataset. It is one of the most popular beginner-friendly machine learning projects.

🎯 Objective
To build a machine learning model that can correctly classify iris flowers into different species based on their features.

📊 Dataset Information
The Iris dataset contains information about 3 types of flowers:

Setosa
Versicolor
Virginica
Features:
Sepal Length

Sepal Width

Petal Length

Petal Width

Total Samples: 150

Target: Flower Species

🧠 Algorithm Used
K-Nearest Neighbors (KNN):

Finds the nearest neighbors of a data point
Uses majority voting for classification
Value of K = 5 (can be changed)
⚙️ Technologies Used
Python
NumPy
Pandas
Scikit-learn
Matplotlib
🧹 Data Preprocessing
Loaded dataset using sklearn
Checked for missing values
Applied feature scaling using StandardScaler
✂️ Train-Test Split
Training Data: 80%
Testing Data: 20%
💻 Model Implementation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
🔮 Prediction
y_pred = model.predict(X_test)
📈 Model Evaluation
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
📊 Results
The model achieves high accuracy (usually above 90%) in classifying iris flower species.

▶️ How to Run
Clone the repository

Install required libraries:

pip install numpy pandas scikit-learn matplotlib
Run the notebook or Python file

📂 Project Structure
KNN-Iris-Project/
│── knn_iris.ipynb
│── README.md
💡 Future Improvements
Try different values of K
Use cross-validation
Visualize decision boundaries
🙌 Conclusion
KNN works very well for the Iris dataset and is a great starting point for learning machine learning.

👤 Author
Nikita
