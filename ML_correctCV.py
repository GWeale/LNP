# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load data
data = 'flow_cytometry_summary.csv'
df = pd.read_csv(data)

# Rename columns
df.rename(columns={
    'Comp-Pacific Blue-A subset': 'cell viability',
    'After Mean': 'Knockdown'
}, inplace=True)

# Separate controls
controls = df[(df['PEI Ratio'] == 0) & (df['NP Ratio'] == 0) & (df['PBA Ratio'] == 0)]
controls.reset_index(drop=True, inplace=True)

# Remove controls from main dataframe
df = df[~((df['PEI Ratio'] == 0) & (df['NP Ratio'] == 0) & (df['PBA Ratio'] == 0))]
df.reset_index(drop=True, inplace=True)

# Select input features (excluding 'Mean') and target ratios
X = df[['cell viability', 'Knockdown']].values  # Updated to exclude 'Mean'
y = df[['PEI Ratio', 'NP Ratio', 'PBA Ratio']].values

# Scale inputs and outputs
input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_scaled = input_scaler.fit_transform(X)
y_scaled = output_scaler.fit_transform(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Define the neural network
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# Initialize the model with updated input size
model = Net(input_size=2, output_size=3)  # input_size changed from 3 to 2

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate grid for prediction using only 'cell viability' and 'Knockdown'
cell_viability_values = np.linspace(df['cell viability'].min(), df['cell viability'].max(), num=50)
knockdown_values = np.linspace(df['Knockdown'].min(), df['Knockdown'].max(), num=50)

grid = np.meshgrid(cell_viability_values, knockdown_values)
grid_reshaped = np.stack([grid[0].ravel(), grid[1].ravel()], axis=1)

# Scale grid inputs
grid_scaled = input_scaler.transform(grid_reshaped)
grid_tensor = torch.tensor(grid_scaled, dtype=torch.float32)

# Make predictions
model.eval()
with torch.no_grad():
    predictions_scaled = model(grid_tensor)
    predictions = output_scaler.inverse_transform(predictions_scaled.numpy())

# Create results dataframe
results = pd.DataFrame(grid_reshaped, columns=['cell viability', 'Knockdown'])
results[['PEI Ratio', 'NP Ratio', 'PBA Ratio']] = predictions

# Compute score (adjusted to use only the two features)
results['Score'] = results['cell viability'] - results['Knockdown']

# Get top 5 results based on score
top_results = results.sort_values('Score', ascending=False).head(5)

# Display top results
print("\nTop 5 Ratios Predicted to Achieve Desired Outputs:")
print(top_results[['PEI Ratio', 'NP Ratio', 'PBA Ratio', 'Score']])

# Save results to CSV
results.to_csv('optimal_ratios.csv', index=False)
