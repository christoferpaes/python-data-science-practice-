import random
import matplotlib.pyplot as plt

def estimate_pi(num_points):
    points_inside_circle = 0
    points_total = 0
    x_inside_circle = []
    y_inside_circle = []
    x_outside_circle = []
    y_outside_circle = []

    for _ in range(num_points):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        distance = x**2 + y**2

        if distance <= 1:
            points_inside_circle += 1
            x_inside_circle.append(x)
            y_inside_circle.append(y)
        else:
            x_outside_circle.append(x)
            y_outside_circle.append(y)

        points_total += 1

    estimated_pi = 4 * (points_inside_circle / points_total)

    # Visualization
    plt.figure(figsize=(6, 6))
    plt.scatter(x_inside_circle, y_inside_circle, color='blue', label='Inside Circle')
    plt.scatter(x_outside_circle, y_outside_circle, color='red', label='Outside Circle')
    plt.title('Monte Carlo Simulation: Estimating π')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return estimated_pi

# Perform the Monte Carlo simulation to estimate pi and visualize the results
num_points = 1000
pi_estimate = estimate_pi(num_points)
print("Estimated pi:", pi_estimate)
