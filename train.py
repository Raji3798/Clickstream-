from preprocess import load_data, preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
import os 

os.environ["LOKY_MAX_CPU_COUNT"] = "4"   

# Load Data
df = load_data("C:/Users/DS/Documents/Clickstream/eshop.csv")
print(df.columns.tolist())


X, y_class, y_reg = preprocess(df)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2)

# ------------------ Classification ------------------
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Accuracy:", accuracy_score(y_test, y_pred))

pickle.dump(clf, open("models/classifier.pkl", "wb"))

# ------------------ Regression ------------------
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2)

reg = LinearRegression()
reg.fit(X_train_r, y_train_r)

y_pred_r = reg.predict(X_test_r)
print("Regression MSE:", mean_squared_error(y_test_r, y_pred_r))

pickle.dump(reg, open("models/regressor.pkl", "wb"))

# ------------------ Clustering ------------------
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

pickle.dump(kmeans, open("models/cluster.pkl", "wb"))

print("All models trained successfully!")