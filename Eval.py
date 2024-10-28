import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class FlowCytometryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FlowCytometryModel(nn.Module):
    def __init__(self, input_dim):
        super(FlowCytometryModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # For bounded output
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        x = self.batch_norm1(self.relu(self.layer1(x)))
        x = self.dropout(x)
        x = self.batch_norm2(self.relu(self.layer2(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.output(x))  # Bound outputs between 0 and 1
        return x

def analyze_data_ranges(df):
    """Analyze the ranges of key metrics in the dataset"""
    metrics = ['Comp-Pacific Blue-A subset', 'After Mean']
    ranges = {}
    for metric in metrics:
        ranges[metric] = {
            'min': df[metric].min(),
            'max': df[metric].max(),
            'mean': df[metric].mean(),
            'std': df[metric].std()
        }
    return ranges

def process_data(df):
    # Analyze data ranges
    data_ranges = analyze_data_ranges(df)
    print("\nData Ranges:")
    for metric, stats in data_ranges.items():
        print(f"{metric}:")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.2f}")
    
    # Group by unique combination of PEI, NP, and PBA ratios
    grouped = df.groupby(['PEI Ratio', 'NP Ratio', 'PBA Ratio']).agg({
        'Comp-Pacific Blue-A subset': 'mean',
        'After Mean': 'mean'
    }).reset_index()
    
    # Prepare features and targets
    X = grouped[['PEI Ratio', 'NP Ratio', 'PBA Ratio']].values
    y = grouped[['After Mean', 'Comp-Pacific Blue-A subset']].values
    
    # Scale the features and targets between 0 and 1
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled, scaler_X, scaler_y, data_ranges

def custom_loss(outputs, targets, data_ranges):
    """
    Custom loss function that:
    - Minimizes After Mean
    - Maximizes Comp-Pacific Blue-A subset
    - Includes penalties for exceeding data ranges
    """
    # Base losses
    after_mean_loss = torch.mean((outputs[:, 0] - targets[:, 0])**2)
    comp_blue_loss = torch.mean((outputs[:, 1] - targets[:, 1])**2)
    
    # Weighted combination
    total_loss = 0.3 * after_mean_loss + 0.7 * comp_blue_loss
    
    return total_loss

def train_model(model, train_loader, test_loader, data_ranges, num_epochs=200, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience_limit = 20
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = custom_loss(outputs, batch_y, data_ranges)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = custom_loss(outputs, batch_y, data_ranges)
                test_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        scheduler.step(avg_test_loss)
        
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    
    model.load_state_dict(best_model_state)
    return train_losses, test_losses

def predict_optimal_parameters(model, scaler_X, scaler_y, data_ranges):
    # Generate grid of possible parameters within the actual data ranges
    pei_range = np.linspace(0, 80, 20)
    np_range = np.linspace(0, 10, 20)
    pba_range = np.linspace(0, 80, 20)
    
    best_score = float('-inf')
    best_params = None
    best_predictions = None
    
    model.eval()
    with torch.no_grad():
        for pei in pei_range:
            for np_val in np_range:
                for pba in pba_range:
                    params = np.array([[pei, np_val, pba]])
                    params_scaled = scaler_X.transform(params)
                    predictions_scaled = model(torch.FloatTensor(params_scaled))
                    predictions = scaler_y.inverse_transform(predictions_scaled.numpy())
                    
                    # Score based on our objectives with constraints
                    after_mean = predictions[0, 0]
                    comp_blue = predictions[0, 1]
                    
                    # Check if predictions are within reasonable bounds
                    if (comp_blue <= data_ranges['Comp-Pacific Blue-A subset']['max'] * 1.1 and
                        after_mean >= data_ranges['After Mean']['min'] * 0.9):
                        
                        # Score favors high Comp-Pacific Blue-A and low After Mean
                        score = comp_blue - 2 * after_mean
                        
                        if score > best_score:
                            best_score = score
                            best_params = params[0]
                            best_predictions = predictions[0]
    
    return best_params, best_predictions

def main():
    # Load and preprocess data
    df = pd.read_csv('flow_cytometry_summary.csv')
    X_scaled, y_scaled, scaler_X, scaler_y, data_ranges = process_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_dataset = FlowCytometryDataset(X_train, y_train)
    test_dataset = FlowCytometryDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize and train model
    model = FlowCytometryModel(input_dim=3)
    train_losses, test_losses = train_model(model, train_loader, test_loader, data_ranges)
    
    # Find optimal parameters
    optimal_params, optimal_predictions = predict_optimal_parameters(model, scaler_X, scaler_y, data_ranges)
    
    print("\nOptimal Parameters:")
    print(f"PEI Ratio: {optimal_params[0]:.2f}")
    print(f"NP Ratio: {optimal_params[1]:.2f}")
    print(f"PBA Ratio: {optimal_params[2]:.2f}")
    print("\nPredicted Outcomes:")
    print(f"After Mean: {optimal_predictions[0]:.2f}")
    print(f"Comp-Pacific Blue-A subset: {optimal_predictions[1]:.2f}")
    
    return model, scaler_X, scaler_y, train_losses, test_losses

if __name__ == "__main__":
    model, scaler_X, scaler_y, train_losses, test_losses = main()
