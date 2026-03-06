# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from model import generate_data, gradient_descent, compute_loss

st.title("Interactive Gradient Descent on House Prices")

# Sidebar controls
lr = st.sidebar.slider("Learning Rate", 0.001, 0.5, 0.01, 0.001)
epochs = st.sidebar.slider("Epochs", 1, 100, 50)
n_points = st.sidebar.slider("Number of Houses", 5, 50, 20)

# Generate housing data
X, y = generate_data(n_points)

# Let user modify a house price interactively
for i in range(len(y)):
    y[i, 0] = st.sidebar.slider(f"Price for house {i+1} (in 1000s)", float(y[i,0]-20), float(y[i,0]+20), float(y[i,0]))

# Run gradient descent
theta_final, history = gradient_descent(X, y, lr=lr, epochs=epochs)

# Create loss surface
theta0_range = np.linspace(np.min(history[:,0,0])-20, np.max(history[:,0,0])+20, 50)
theta1_range = np.linspace(np.min(history[:,1,0])-0.5, np.max(history[:,1,0])+0.5, 50)
Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
Loss = np.zeros_like(Theta0)

for i in range(Theta0.shape[0]):
    for j in range(Theta0.shape[1]):
        Loss[i,j] = compute_loss(np.array([[Theta0[i,j]], [Theta1[i,j]]]), X, y)

# 3D Plot with interactive gradient path
fig = go.Figure(data=[go.Surface(z=Loss, x=Theta0, y=Theta1, colorscale='Viridis', opacity=0.8)])
loss_path = [compute_loss(h, X, y) for h in history]
theta0_path = history[:,0,0]
theta1_path = history[:,1,0]

fig.add_trace(go.Scatter3d(
    x=theta0_path,
    y=theta1_path,
    z=loss_path,
    mode='lines+markers',
    line=dict(color='red', width=5),
    marker=dict(size=3)
))

fig.update_layout(scene=dict(
    xaxis_title='Theta0 (Bias)',
    yaxis_title='Theta1 (Slope)',
    zaxis_title='Loss'
))

st.plotly_chart(fig)
st.write("Final Parameters (Bias, Slope):", theta_final.ravel())
st.write("You can adjust learning rate, epochs, or house prices using the sidebar to see how the gradient descent path changes.")
