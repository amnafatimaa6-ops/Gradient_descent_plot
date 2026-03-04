st.title("Gradient Descent Visualizer (3D)")

# Sidebar controls
lr = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Epochs", 1, 100, 50)
n_points = st.sidebar.slider("Data Points", 10, 200, 50)

# Generate data
X, y = generate_data(n_points)

# Run gradient descent
theta_final, history = gradient_descent(X, y, lr=lr, epochs=epochs)

# Create loss surface
theta0_range = np.linspace(-1, 10, 50)
theta1_range = np.linspace(0, 6, 50)
Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
Loss = np.zeros_like(Theta0)

for i in range(Theta0.shape[0]):
    for j in range(Theta0.shape[1]):
        Loss[i, j] = compute_loss(np.array([[Theta0[i, j]], [Theta1[i, j]]]), X, y)

# 3D Plot
fig = go.Figure(data=[go.Surface(z=Loss, x=Theta0, y=Theta1, colorscale='Viridis', opacity=0.8)])

# Add gradient descent path
loss_path = [compute_loss(h, X, y) for h in history]
theta0_path = history[:, 0, 0]
theta1_path = history[:, 1, 0]
fig.add_trace(go.Scatter3d(
    x=theta0_path,
    y=theta1_path,
    z=loss_path,
    mode='lines+markers',
    line=dict(color='red', width=5),
    marker=dict(size=3)
))

fig.update_layout(scene=dict(
    xaxis_title='Theta0',
    yaxis_title='Theta1',
    zaxis_title='Loss'
))

st.plotly_chart(fig)
st.write("Final Parameters:", theta_final.ravel())
