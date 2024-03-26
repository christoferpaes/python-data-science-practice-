import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Features (all columns except the last)
    y = data.iloc[:, -1].values   # Target variable (last column)
    return X, y

def preprocess_data(X, y, test_size=0.2, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train, degree=1):
    # Create polynomial features
    polynomial_features = PolynomialFeatures(degree=degree)
    X_train_poly = polynomial_features.fit_transform(X_train)
    
    # Fit the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    # Create polynomial features for test data
    polynomial_features = PolynomialFeatures(degree=len(model.coef_) - 1)
    X_test_poly = polynomial_features.fit_transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_poly)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return mse, rmse, r2, y_pred

def plot_results(X_train, y_train, X_test, y_test, y_pred):
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X_train[:, 0], y_train, color='blue', label='Training Data')
    
    # Plot testing data
    plt.scatter(X_test[:, 0], y_test, color='red', label='Testing Data')
    
    # Sort test data for smooth curve
    sort_indices = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sort_indices]
    y_pred_sorted = y_pred[sort_indices]
    
    # Plot predicted values
    plt.plot(X_test_sorted[:, 0], y_pred_sorted, color='green', linewidth=2, label='Predictions')
    
    plt.title('Polynomial Regression')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load data
    X, y = load_data('your_data_file.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train polynomial regression model
    model = train_model(X_train, y_train, degree=3)
    
    # Evaluate model
    mse, rmse, r2, y_pred = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')
    
    # Plot results
    plot_results(X_train, y_train, X_test, y_test, y_pred)

if __name__ == "__main__":
    main()
