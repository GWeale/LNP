import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import signal
from contextlib import contextmanager
import time
import shap

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def analyze_data(df):
    """Analyze and print summary statistics of the dataset"""
    # Filter out control samples
    df = df[~((df['PEI Ratio'] == 0) & 
              (df['NP Ratio'] == 0) & 
              (df['PBA Ratio'] == 0))]
    
    print("\nData Summary (excluding controls):")
    for col in ['PEI Ratio', 'NP Ratio', 'PBA Ratio', 'Comp-Pacific Blue-A subset', 'After Mean']:
        print(f"\n{col}:")
        print(f"  Range: {df[col].min():.2f} to {df[col].max():.2f}")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Std: {df[col].std():.2f}")

def process_data(df):
    # Filter out control samples (where all ratios are 0)
    df = df[~((df['PEI Ratio'] == 0) & 
              (df['NP Ratio'] == 0) & 
              (df['PBA Ratio'] == 0))]
    
    # Group by unique combination of parameters to handle multiple trials
    grouped = df.groupby(['PEI Ratio', 'NP Ratio', 'PBA Ratio']).agg({
        'Comp-Pacific Blue-A subset': ['mean', 'std'],
        'After Mean': ['mean', 'std']
    }).reset_index()
    
    # Flatten multi-level columns
    grouped.columns = ['PEI Ratio', 'NP Ratio', 'PBA Ratio', 
                      'Comp_Blue_mean', 'Comp_Blue_std',
                      'After_Mean_mean', 'After_Mean_std']
    
    print(f"\nNumber of samples after removing controls: {len(grouped)}")
    return grouped

def train_models_with_kfold(X, y_comp_blue, y_after_mean, n_splits=3):
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    comp_blue_rmse_scores = []
    after_mean_rmse_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_comp_blue_train, y_comp_blue_test = y_comp_blue[train_index], y_comp_blue[test_index]
        y_after_mean_train, y_after_mean_test = y_after_mean[train_index], y_after_mean[test_index]
        
        param_grid = {
            'max_depth': [2, 3],           # Reduced options
            'learning_rate': [0.1],        # Single value
            'n_estimators': [50],          # Single value
            'min_child_weight': [1],       # Single value
            'gamma': [0],                  # Single value
            'subsample': [0.8]             # Single value
        }
        
        model_comp_blue = GridSearchCV(
            xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            param_grid,
            cv=2,                          # Reduced from 3 to 2
            scoring='neg_mean_squared_error',
            verbose=1
        )
        model_comp_blue.fit(X_train, y_comp_blue_train)
        
        model_after_mean = GridSearchCV(
            xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            param_grid,
            cv=2,                          # Reduced from 3 to 2
            scoring='neg_mean_squared_error',
            verbose=1
        )
        model_after_mean.fit(X_train, y_after_mean_train)
        
        y_comp_blue_pred = model_comp_blue.predict(X_test)
        y_after_mean_pred = model_after_mean.predict(X_test)
        
        comp_blue_rmse = np.sqrt(mean_squared_error(y_comp_blue_test, y_comp_blue_pred))
        after_mean_rmse = np.sqrt(mean_squared_error(y_after_mean_test, y_after_mean_pred))
        
        comp_blue_rmse_scores.append(comp_blue_rmse)
        after_mean_rmse_scores.append(after_mean_rmse)
    
    print("\nCross-Validation Performance:")
    print(f"Comp-Pacific Blue-A subset RMSE: {np.mean(comp_blue_rmse_scores):.2f} ± {np.std(comp_blue_rmse_scores):.2f}")
    print(f"After Mean RMSE: {np.mean(after_mean_rmse_scores):.2f} ± {np.std(after_mean_rmse_scores):.2f}")
    
    return model_comp_blue, model_after_mean

def find_optimal_parameters(model_comp_blue, model_after_mean, data_ranges):
    """Find optimal parameters using grid search"""
    pei_range = np.linspace(50.0, 80.0, 20)
    np_range = np.linspace(0.0, 10.0, 20)
    pba_range = np.linspace(10.0, 77.0, 20)
    
    best_score = float('-inf')
    best_params = None
    best_predictions = None
    
    # Grid search
    for pei in pei_range:
        for np_val in np_range:
            for pba in pba_range:
                params = np.array([[pei, np_val, pba]])
                
                comp_blue_pred = model_comp_blue.predict(params)[0]
                after_mean_pred = model_after_mean.predict(params)[0]
                
                # Score based on objectives:
                # High Comp-Pacific Blue-A subset and low After Mean
                # With constraints based on observed data ranges
                if (comp_blue_pred <= data_ranges['Comp-Pacific Blue-A subset']['max'] * 1.1 and
                    comp_blue_pred >= data_ranges['Comp-Pacific Blue-A subset']['min'] * 0.9 and
                    after_mean_pred <= data_ranges['After Mean']['max'] * 1.1 and
                    after_mean_pred >= data_ranges['After Mean']['min'] * 0.9):
                    
                    # Score function: maximize Comp Blue while minimizing After Mean
                    score = comp_blue_pred - 2 * after_mean_pred
                    
                    if score > best_score:
                        best_score = score
                        best_params = [pei, np_val, pba]
                        best_predictions = [comp_blue_pred, after_mean_pred]
    
    return best_params, best_predictions

def plot_feature_importance(model_comp_blue, model_after_mean):
    """Plot feature importance for both models"""
    features = ['PEI Ratio', 'NP Ratio', 'PBA Ratio']
    
    # Get feature importance scores
    importance_comp_blue = model_comp_blue.best_estimator_.feature_importances_
    importance_after_mean = model_after_mean.best_estimator_.feature_importances_
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot for Comp-Pacific Blue-A subset
    ax1.bar(features, importance_comp_blue)
    ax1.set_title('Feature Importance\nComp-Pacific Blue-A subset')
    ax1.set_ylabel('Importance Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot for After Mean
    ax2.bar(features, importance_after_mean)
    ax2.set_title('Feature Importance\nAfter Mean')
    ax2.set_ylabel('Importance Score')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_all_shap_analysis(model_comp_blue, model_after_mean, X):
    """
    Create all SHAP plots in a single figure
    """
    plt.style.use('seaborn')
    features = ['PEI Ratio', 'NP Ratio', 'PBA Ratio']
    
    # Create a large figure with subplots
    fig = plt.figure(figsize=(20, 25))
    
    # Calculate SHAP values
    explainer_comp = shap.TreeExplainer(model_comp_blue.best_estimator_)
    explainer_after = shap.TreeExplainer(model_after_mean.best_estimator_)
    shap_values_comp = explainer_comp.shap_values(X)
    shap_values_after = explainer_after.shap_values(X)
    
    # Summary dot plots (top row)
    plt.subplot(4, 2, 1)
    shap.summary_plot(shap_values_comp, X, 
                     feature_names=features,
                     plot_type="dot",
                     show=False,
                     alpha=0.5,
                     max_display=3)
    plt.title("SHAP Values Impact on Comp-Pacific Blue-A", pad=20, fontsize=12)
    
    plt.subplot(4, 2, 2)
    shap.summary_plot(shap_values_after, X, 
                     feature_names=features,
                     plot_type="dot",
                     show=False,
                     alpha=0.5,
                     max_display=3)
    plt.title("SHAP Values Impact on After Mean", pad=20, fontsize=12)
    
    # Dependence plots for each feature (remaining rows)
    for i, feature in enumerate(features):
        # Comp-Pacific Blue-A
        plt.subplot(4, 2, 2*i + 3)
        shap.dependence_plot(
            ind=i, 
            shap_values=shap_values_comp, 
            features=X,
            feature_names=features,
            show=False,
            alpha=0.5
        )
        plt.title(f'Impact of {feature} on Comp-Pacific Blue-A', pad=20, fontsize=12)
        
        # After Mean
        plt.subplot(4, 2, 2*i + 4)
        shap.dependence_plot(
            ind=i, 
            shap_values=shap_values_after, 
            features=X,
            feature_names=features,
            show=False,
            alpha=0.5
        )
        plt.title(f'Impact of {feature} on After Mean', pad=20, fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout(pad=3.0)
    plt.savefig('all_shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    try:
        with time_limit(300):  # 5 minutes timeout
            # Load and process data
            df = pd.read_csv('flow_cytometry_summary.csv')
            
            # Analyze raw data
            analyze_data(df)
            
            # Process data
            processed_data = process_data(df)
            
            # Prepare features and targets
            X = processed_data[['PEI Ratio', 'NP Ratio', 'PBA Ratio']].values
            y_comp_blue = processed_data['Comp_Blue_mean'].values
            y_after_mean = processed_data['After_Mean_mean'].values
            
            # Calculate data ranges for constraints
            data_ranges = {
                'Comp-Pacific Blue-A subset': {
                    'min': df['Comp-Pacific Blue-A subset'].min(),
                    'max': df['Comp-Pacific Blue-A subset'].max()
                },
                'After Mean': {
                    'min': df['After Mean'].min(),
                    'max': df['After Mean'].max()
                }
            }
            
            # Train models
            model_comp_blue, model_after_mean = train_models_with_kfold(X, y_comp_blue, y_after_mean)
            
            # Find optimal parameters
            optimal_params, optimal_predictions = find_optimal_parameters(model_comp_blue, model_after_mean, data_ranges)
            
            print("\nOptimal Parameters:")
            print(f"PEI Ratio: {optimal_params[0]:.2f}")
            print(f"NP Ratio: {optimal_params[1]:.2f}")
            print(f"PBA Ratio: {optimal_params[2]:.2f}")
            print("\nPredicted Outcomes:")
            print(f"Comp-Pacific Blue-A subset: {optimal_predictions[0]:.2f}")
            print(f"After Mean: {optimal_predictions[1]:.2f}")
            
            # Return the models and data for plotting outside the timeout
            return model_comp_blue, model_after_mean, X

    except TimeoutException as e:
        print("Script timed out after 5 minutes!")
        return None, None, None

if __name__ == "__main__":
    # Run main analysis
    model_comp_blue, model_after_mean, X = main()
    
    # If models were successfully trained, create plots
    if model_comp_blue is not None and model_after_mean is not None:
        # Plot feature importance
        plot_feature_importance(model_comp_blue, model_after_mean)
        
        # Plot all SHAP analysis in one figure
        plot_all_shap_analysis(model_comp_blue, model_after_mean, X)
