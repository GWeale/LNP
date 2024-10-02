# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler


data = 'flow_cytometry_summary.csv'

df = pd.read_csv(data)


controls = df[(df['PEI Ratio'] == 0) & (df['NP Ratio'] == 0) & (df['PBA Ratio'] == 0)]
controls.reset_index(drop=True, inplace=True)

df = df[~((df['PEI Ratio'] == 0) & (df['NP Ratio'] == 0) & (df['PBA Ratio'] == 0))]
df.reset_index(drop=True, inplace=True)


input_columns = ['Comp-Pacific Blue-A subset', 'Mean', 'After Mean', 'q1', 'q2', 'q3', 'q4']
X = df[input_columns].values


y = df[['PEI Ratio', 'NP Ratio', 'PBA Ratio']].values


input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_scaled = input_scaler.fit_transform(X)
y_scaled = output_scaler.fit_transform(y)


X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

model = Net(input_size=len(input_columns), output_size=3) 

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


num_epochs = 500
for epoch in range(num_epochs):
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


comp_pacific_values = np.linspace(df['Comp-Pacific Blue-A subset'].max(), df['Comp-Pacific Blue-A subset'].min(), num=20)
mean_values = np.linspace(df['Mean'].min(), df['Mean'].max(), num=20)
after_mean_values = np.linspace(df['After Mean'].min(), df['After Mean'].max(), num=20)
q1_values = np.linspace(df['q1'].min(), df['q1'].max(), num=5)
q2_values = np.linspace(df['q2'].min(), df['q2'].max(), num=5)
q3_values = np.linspace(df['q3'].min(), df['q3'].max(), num=5)
q4_values = np.linspace(df['q4'].min(), df['q4'].max(), num=5)

grid = np.meshgrid(comp_pacific_values, mean_values, after_mean_values, q1_values, q2_values, q3_values, q4_values)
grid_reshaped = np.stack([g.ravel() for g in grid], axis=1)

grid_scaled = input_scaler.transform(grid_reshaped)
grid_tensor = torch.tensor(grid_scaled, dtype=torch.float32)

with torch.no_grad():
    predictions_scaled = model(grid_tensor)
    predictions = output_scaler.inverse_transform(predictions_scaled.numpy())

results = pd.DataFrame(grid_reshaped, columns=input_columns)
results[['PEI Ratio', 'NP Ratio', 'PBA Ratio']] = predictions

results['Score'] = results['Comp-Pacific Blue-A subset'] - results['Mean'] - results['After Mean']

top_results = results.sort_values('Score', ascending=False).head(5)

print("\nTop 5 Ratios Predicted to Achieve Desired Outputs with q-values:")
print(top_results[['PEI Ratio', 'NP Ratio', 'PBA Ratio', 'Score']])

results.to_csv('optimal_ratios_with_q_values.csv', index=False)
