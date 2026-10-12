import math
import random
import matplotlib.pyplot as plt


def input_positive_number(prompt):
    # Grabs a positive number from user
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                print("That's not a positive number silly")
            else:
                return value
        except ValueError:
            print("That is not a valid input")


def input_number(prompt):
    # Grabs any number from user
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("That is not a valid input")


def derivatives(x, y, vx, vy, kx, kvx, ky, kvy):
    # Helper for RK4
    ux, uy = PD(x, y, vx, vy, kx, kvx, ky, kvy)

    dxdt = vx
    dydt = vy
    dvxdt = ux
    dvydt = uy - g

    return dxdt, dydt, dvxdt, dvydt


def RK4Step(x, y, vx, vy, kx, kvx, ky, kvy):
    # Implements the fourth order RK4 algorithm
    k1 = derivatives(x, y, vx, vy, kx, kvx, ky, kvy)

    k2 = derivatives(
        x + 0.5 * dt * k1[0],
        y + 0.5 * dt * k1[1],
        vx + 0.5 * dt * k1[2],
        vy + 0.5 * dt * k1[3],
        kx,
        kvx,
        ky,
        kvy,
    )

    k3 = derivatives(
        x + 0.5 * dt * k2[0],
        y + 0.5 * dt * k2[1],
        vx + 0.5 * dt * k2[2],
        vy + 0.5 * dt * k2[3],
        kx,
        kvx,
        ky,
        kvy,
    )

    k4 = derivatives(
        x + dt * k3[0],
        y + dt * k3[1],
        vx + dt * k3[2],
        vy + dt * k3[3],
        kx,
        kvx,
        ky,
        kvy,
    )

    x += (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    y += (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    vx += (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    vy += (dt / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

    return x, y, vx, vy


def PD(x, y, vx, vy, kx, kvx, ky, kvy):
    # Updates the controller. The rocket has a maximum possible thrust
    safety_factor = 0.7
    ux = -kx * x - kvx * vx
    height_error = max(y - y_target, 0)
    hoirzontal_error = x
    ax_brake = uxmax
    ay_brake = uymax - g

    # Control the horizontal movement
    vx_desired = -math.copysign(
        safety_factor * math.sqrt(2 * ax_brake * abs(hoirzontal_error)),
        hoirzontal_error,
    )
    if abs(hoirzontal_error) < tolerance:
        vx_desired = 0.0

    if hoirzontal_error * vx > 0 and abs(vx) > abs(vx_desired):
        ux = -math.copysign(uxmax, vx)
    else:
        ux = -kvx * (vx - vx_desired)

    # Control the vertical movement
    vy_desired = -safety_factor * math.sqrt(2 * ay_brake * height_error)

    if vy < vy_desired:
        uy = uymax
    else:
        uy = -kvy * (vy - vy_desired) + g

    # Implement maximum allowed thrust
    ux = max(min(ux, uxmax), -uxmax)
    uy = max(min(uy, uymax), 0)

    return ux, uy


def Simulation(kx, ky, vx, vy):
    # Runs the simulation

    # Enforces critical damping
    kvx, kvy = 2 * (kx**0.5), 2 * (ky**0.5)

    # Initialize variables
    x_vals, y_vals = [x_0], [y_0]
    x, y = x_0, y_0
    control_effort = 0
    crash = False
    step = 0

    # Continue simulation until we are close enough to the target and slow enough, or hit the maximum number of steps
    while (
        abs(x) > tolerance
        or abs(y - y_target) > tolerance
        or abs(vx) > tolerance
        or abs(vy) > tolerance
    ) and (step < max_steps):

        # Gives a score for how much thrust we use. Will use to optimize k value
        ux, uy = PD(x, y, vx, vy, kx, kvx, ky, kvy)
        control_effort += (ux**2 + uy**2) * dt

        # Update the pd and positions
        x, y, vx, vy = RK4Step(x, y, vx, vy, kx, kvx, ky, kvy)

        # Checks to see if we crashed
        if y < 0:
            y = 0
            crash = True
            break

        # Record new position
        x_vals.append(x)
        y_vals.append(y)
        step += 1

    # Makes vector for each time point
    t_vals = [i * dt for i in range(len(x_vals))]

    return (
        x,
        y,
        x_vals,
        y_vals,
        t_vals,
        vx,
        vy,
        step,
        step * dt,
        crash,
        control_effort,
    )


def Landing_Cost(kx, ky, vx, vy):
    # Creates a score so we can optimize the k values
    x, y, x_vals, y_vals, t_vals, vx, vy, step, time, crash, effort = Simulation(
        kx, ky, vx, vy
    )
    if crash:
        return 1e9
    position_error = abs(x) + abs(y - y_target)
    velocity_error = abs(vx) + abs(vy)
    return 1000 * position_error + 1000 * velocity_error + 1.0 * time + 0.1 * effort


def Log_Uniform(low, high):
    # Picks out a logarithmically unifrom number in the interval [low, high)
    return math.exp(random.uniform(math.log(low), math.log(high)))


def Random_Search(trials, vx, vy, minx, maxx, miny, maxy):
    # Searches in a logarithmically uniform way a range to find optimal k values
    best_cost = float("inf")
    best_kx = None
    best_ky = None

    for i in range(trials):
        kx = Log_Uniform(minx, maxx)
        ky = Log_Uniform(miny, maxy)
        cost = Landing_Cost(kx, ky, vx, vy)
        if cost < best_cost:
            best_cost = cost
            best_kx = kx
            best_ky = ky
    return (best_kx, best_ky, best_cost)


# Sets the random seed
random.seed(0)

# Initialize variables
g, dt = 9.81, 0.01
uxmax, uymax = 5, 100
tolerance, max_steps = 1e-2, 200000
kx, ky = 1.0, 1.0

# Enforce critical damping
kvx, kvy = 2 * (kx**0.5), 2 * (ky**0.5)

# Grabs starting data from user
y_0 = input_positive_number("What is the starting height (m)?")
y_target = input_positive_number("What is the desired height (m)?")
x_0 = input_number("How far left (negative) or right (positive) do we need to go (m)?")
vy = input_number("What is our current vertical velocity (m/s)?")
vx = input_number("What is our current horizontal velocity (m/s)?")

# Finds a "good" value in a starting range, then shrinks the range and finds another "good" value, then repeats once more
best_kx, best_ky, best_cost = Random_Search(100, vx, vy, 0.05, 10, 0.05, 10)

trial_kx, trial_ky, trial_cost = Random_Search(
    100, vx, vy, best_kx / 3, best_kx * 3, best_ky / 3, best_ky * 3
)

if trial_cost < best_cost:
    best_kx, best_ky, best_cost = trial_kx, trial_ky, trial_cost

trial_kx, trial_ky, trial_cost = Random_Search(
    100, vx, vy, best_kx / 2, best_kx * 2, best_ky / 2, best_ky * 2
)

if trial_cost < best_cost:
    best_kx, best_ky, best_cost = trial_kx, trial_ky, trial_cost

print("Best kx:", best_kx)
print("Best ky:", best_ky)
print("Best cost:", best_cost)

x, y, x_vals, y_vals, t_vals, vx, vy, step, time, crash, effort = Simulation(
    best_kx, best_ky, vx, vy
)

# Plots the data
plt.figure()
plt.plot(t_vals, y_vals)
plt.xlabel("Time (s)")
plt.ylabel("Height (y)")
plt.title("Height vs Time")
plt.grid()

plt.figure()
plt.plot(t_vals, x_vals)
plt.xlabel("Time (s)")
plt.ylabel("Horizontal Position (x)")
plt.title("X Position vs Time")
plt.grid()

plt.figure()
plt.plot(x_vals, y_vals)
plt.xlabel("X Position")
plt.ylabel("Height (y)")
plt.title("Rocket Trajectory")
plt.grid()
plt.gca()
plt.show()
