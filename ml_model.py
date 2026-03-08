import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

input_file = r'C:\Users\karan\PycharmProjects\solubility_prediction_ml\cal_descriptor_file.csv'
df = pd.read_csv(input_file)


X = df.drop(['SMILES', 'LogS'], axis=1).apply(pd.to_numeric, errors='coerce')
y = df['LogS']

#remove rows with infinity and too big for float32
float32_max = np.finfo(np.float32).max
mask = np.all(np.isfinite(X), axis=1) & np.all(np.abs(X) <= float32_max, axis=1)

X = X[mask].fillna(X.mean())
y = y[mask]

print(f"remaining molecules for training: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("spliting the data was done")

# training the model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train.astype(np.float64), y_train)

# Training evaluation
train_prediction = model.predict(X_train.astype(np.float64))
train_r2 = r2_score(y_train, train_prediction)
print(f"Training R2 score: {train_r2:.4f}")

# evaluation of the test data
ml_prediction = model.predict(X_test.astype(np.float64))
r2_evaluation = r2_score(y_test, ml_prediction)
mse_evaluation = mean_squared_error(y_test, ml_prediction)

print(f"r2 score for the model is {r2_evaluation:.4f}")
print(f"mean squared error of model is {mse_evaluation:.4f}")

# save
output_dir = r'C:\Users\karan\PycharmProjects\solubility_prediction_ml'
model_filename = 'solubility_model.pkl'
model_save_path = os.path.join(output_dir, model_filename)

joblib.dump(model, model_save_path)
print("trained model saved to ", model_save_path)

# checking the important features
importance = model.feature_importances_
features_name = X.columns
feature_imp_df = pd.DataFrame({'Feature': features_name, 'Importance': importance})
top_10_features = feature_imp_df.sort_values(by="Importance", ascending=False).head(10)
print("top 10 important features ", top_10_features)

# visualization
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=top_10_features, palette='viridis')
plt.title('Top 10 features contribute for solubility')
plt.xlabel('Importance')
plt.ylabel('Features')

plot_save_path = os.path.join(output_dir, 'feature_importance.png')
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"feature importance plot saved to: {plot_save_path}")
plt.show()