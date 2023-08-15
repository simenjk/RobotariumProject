import rps.robotarium as robotarium
from rps.utilities.misc import *
from rps.utilities.controllers import *
import numpy as np


def find_distance(start_x, end_x, start_y, end_y):
    return np.sqrt(np.square(start_x - end_x) + np.square(start_y - end_y))


def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cosine_angle)


# Number of robots and initial conditions used in the Robotarium object
N = 1
initial_conditions = np.array(np.mat('0.8; 0.6; 0'))
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,
                          sim_in_real_time=True)
current_pos = r.get_poses()  # Current position of the robot

# Controllers
position_controller = create_si_position_controller()
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()


# Making the two circles that illustrates the path the robot can take
circleRadius = float(0.4)
circle_drawing1 = plt.Circle((-circleRadius, 0), circleRadius, fill=False, color='#083070')
circle_drawing2 = plt.Circle((circleRadius, 0), circleRadius, fill=False, color='#910f15')
r.axes.add_artist(circle_drawing1)
r.axes.add_artist(circle_drawing2)

countMax = 2000  # Maximum number of steps the robot takes
num_waypoints = 100  # Number of waypoints the robot will drive through. Higher number = less stuttering


# Array with values between 0 and 2π
th_vec = np.zeros(num_waypoints, dtype=np.float64)
for i in range(num_waypoints):
    th_vec[i] = 2 * np.pi * i / num_waypoints

# Initializing waypoint-array
waypoints = np.zeros((2, 2 * num_waypoints), dtype=np.float64)

# Waypoints for left circle
for i in range(num_waypoints):
    theta = th_vec[i]
    waypoints[0, i] = (circleRadius - circleRadius * np.cos(theta)) * (-1)
    waypoints[1, i] = circleRadius * np.sin(theta)

# Waypoints for right circle
for i in range(num_waypoints):
    theta = th_vec[i]
    waypoints[0, i - num_waypoints] = circleRadius - circleRadius * np.cos(theta)
    waypoints[1, i - num_waypoints] = circleRadius * np.sin(theta)

# Sets an arbitrary waypoint as the closest so it has something to compare it to
closestDistance = find_distance(current_pos.item(0), waypoints.item(0, 0), current_pos.item(1), waypoints.item(1, 0))
waypointNow = waypoints[:, 0]
waypointIndexNow = 0

# Finds the closest waypoint by checking all the waypoints
for i in range(num_waypoints * 2):
    currentDistance = find_distance(float(current_pos.item(0)), waypoints.item(0, i), float(current_pos.item(1)), waypoints.item(1, i))
    if currentDistance < closestDistance:
        closestDistance = currentDistance
        waypointNow = waypoints[:, i]
        waypointIndexNow = i

# Flag that tells if we have to reverse the route
reverse_flag = False


# Finds the waypoint with the best starting angle
# Checks if the robot starts outside the circle. Have to find the tangent in the opposite circle if it starts inside.
if np.abs(current_pos.item(0)) > circleRadius * 2 or np.abs(current_pos.item(1)) > circleRadius:
    angle_counter = waypointIndexNow

    if angle_counter >= num_waypoints:
        circle_limit = num_waypoints * 2
    else:
        circle_limit = num_waypoints

    # Finds out if we are closest to negative or positive x-circle and save the middle of the circles
    if current_pos.item(0) > 0:
        circle_centre = np.array([circleRadius, float(0)])
    else:
        circle_centre = np.array([-circleRadius, float(0)])

    # Store robot position in numpy.array
    robot_pos = np.array([current_pos.item(0), current_pos.item(1)])

    # Set start angle to the closest one we have found
    best_angle = np.abs(
        get_angle(circle_centre, (waypoints[0, angle_counter], waypoints[1, angle_counter]), robot_pos) - np.pi / 2)
    best_angle_index = waypointIndexNow
    for i in range(int(np.floor(num_waypoints))):
        currentWaypoint = ([waypoints[0, angle_counter], waypoints[1, angle_counter]])
        # Finds angle for every waypoint
        current_angle = get_angle(circle_centre, currentWaypoint, robot_pos)

        # We look for angles close to 3π/2 and π/2
        if current_angle > np.pi:
            currentAngle = 2 * np.pi - current_angle

        # Subtract π/2 and find abs.value, so we can compare all numbers
        current_angle = np.abs(current_angle - np.pi / 2)

        if current_angle < best_angle:
            best_angle = current_angle
            waypointNow = waypoints[:, angle_counter]
            best_angle_index = angle_counter
            # Turns in origo if it comes from negative y and x = 0, have to add an extra condition
            if angle_counter < waypointIndexNow and angle_counter != 0:
                reverse_flag = True
        angle_counter = angle_counter + 1
        if angle_counter >= circle_limit:
            angle_counter = angle_counter - num_waypoints

    waypointIndexNow = best_angle_index
else:
    # If robot starts inside an circle
    angle_counter = waypointIndexNow

    if angle_counter >= num_waypoints:
        circle_limit = num_waypoints
    else:
        circle_limit = num_waypoints * 2

    if current_pos.item(0) > 0:
        circle_centre = np.array([-circleRadius, float(0)])
    else:
        circle_centre = np.array([circleRadius, float(0)])

    robot_pos = np.array([current_pos.item(0), current_pos.item(1)])

    best_angle = np.abs(
        get_angle(circle_centre,
                  (waypoints[0, num_waypoints * 2 - angle_counter], waypoints[1, num_waypoints - angle_counter]),
                  robot_pos) - np.pi / 2)
    best_angle_index = num_waypoints * 2 - angle_counter
    angle_counter = num_waypoints * 2 - angle_counter

    for i in range(int(np.floor(num_waypoints))):
        currentWaypoint = ([waypoints[0, angle_counter], waypoints[1, angle_counter]])
        current_angle = get_angle(circle_centre, currentWaypoint, robot_pos)

        if current_angle > np.pi:
            currentAngle = 2 * np.pi - current_angle

        current_angle = np.abs(current_angle - np.pi / 2)

        if current_angle < best_angle:
            best_angle = current_angle
            waypointNow = waypoints[:, angle_counter]
            best_angle_index = angle_counter
            if angle_counter > waypointIndexNow:
                reverse_flag = True
        angle_counter = angle_counter + 1
        if angle_counter >= circle_limit:
            angle_counter = angle_counter - num_waypoints

    waypointIndexNow = best_angle_index

# Set waypointNow to array with 2 elememts with 1 value in each
waypointNow.shape = (2, 1)

# How close the robot has to be to go to the next waypoint
distance_tolerance = 0.01

# Starts simulation
r.step()
count = 0

while count < countMax:
    current_pos = r.get_poses()
    count = count + 1

    # Find present location and orientation
    xRob = float(current_pos.item(0))
    yRob = float(current_pos.item(1))
    psiRob = float(current_pos.item(2))

    # Gets coordination to current waypoint
    wayPtX = waypointNow[0]
    wayPtY = waypointNow[1]

    # Checks how far the robot is from current waypoint
    distance = np.square(find_distance(xRob, wayPtX, yRob, wayPtY))

    # If robot is closer than we have sat distance_tolerance, change waypoint
    if distance <= distance_tolerance:
        # Drives backwards in waypoints array if reverse_flag = true
        if reverse_flag:
            waypointIndexNow = waypointIndexNow - 1
        else:
            waypointIndexNow = waypointIndexNow + 1

        # Checks if robot has been in all waypoints
        if waypointIndexNow >= 2 * num_waypoints:
            waypointIndexNow = 0
        elif waypointIndexNow <= 0:
            waypointIndexNow = num_waypoints * 2 - 1

        waypointNow = waypoints[:, waypointIndexNow]
        waypointNow.shape = (2, 1)

    # Store array with current coordinates
    pos = np.array([xRob, yRob])
    pos.shape = (2, 1)

    # Get velocity from rps
    vel = position_controller(pos, waypointNow)

    # Convert velocity so it can be used on unicycle
    uni_vel = si_to_uni_dyn(vel, current_pos)
    r.set_velocities(0, uni_vel)

    # Simulates one step
    r.step()

r.call_at_scripts_end()
