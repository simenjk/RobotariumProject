import rps.robotarium as robotarium
from rps.utilities.misc import *
from rps.utilities.controllers import *
import numpy as np


# Antall roboter
N = 1

# Setter startpunkt for robot
initial_conditions = np.array(np.mat('0.8; 0.6; 0'))

# Initialiserer robotobjektet med Robotariumklassen

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,
                          sim_in_real_time=True)


# funksjon som finner distanse.
def find_distance(start_x, end_x, start_y, end_y):
    return np.sqrt(np.square(start_x - end_x) + np.square(start_y - end_y))


# funksjon som finner vinkel, bruker den til å finne hvilket waypoint roboten skal gå til først
def get_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cosine_angle)


# lage kontrollere
position_controller = create_si_position_controller()
si_to_uni_dyn = create_si_to_uni_dynamics_with_backwards_motion()

# lagre current position
x = r.get_poses()

# radius på sirkelene
circleRadius = float(0.4)

# hvor mange steps vi tar før avslutting
countMax = 2000

# lage 2 sirkler og plotte de inn
circle_drawing1 = plt.Circle((-circleRadius, 0), circleRadius, fill=False, color='#083070')
circle_drawing2 = plt.Circle((circleRadius, 0), circleRadius, fill=False, color='#910f15')
r.axes.add_artist(circle_drawing1)
r.axes.add_artist(circle_drawing2)

# definere antall waypoints
num_waypoints = 40

# array med verdier mellom 0 og 2pi med samme mellomrom
th_vec = np.zeros(num_waypoints, dtype=np.float64)
for i in range(num_waypoints):
    th_vec[i] = 2 * np.pi * i / num_waypoints

# initialiserer waypoints-arrayet
waypoints = np.zeros((2, 2 * num_waypoints), dtype=np.float64)

# Waypoints til venstre sirkel
for i in range(num_waypoints):
    theta = th_vec[i]
    waypoints[0, i] = (circleRadius - circleRadius * np.cos(theta)) * (-1)
    waypoints[1, i] = circleRadius * np.sin(theta)

# waypoints til høyre sirkel
for i in range(num_waypoints):
    theta = th_vec[i]
    waypoints[0, i - num_waypoints] = circleRadius - circleRadius * np.cos(theta)
    waypoints[1, i - num_waypoints] = circleRadius * np.sin(theta)

# setter et vilkårlig waypoint som nærmeste så den har noe å sammenlikne med
closestDistance = find_distance(x.item(0), waypoints.item(0, 0), x.item(1), waypoints.item(1, 0))
waypointNow = waypoints[:, 0]
waypointIndexNow = 0

# sjekker alle andre waypoints for å finne det som er nærmest
for i in range(num_waypoints * 2):
    currentDistance = find_distance(float(x.item(0)), waypoints.item(0, i), float(x.item(1)), waypoints.item(1, i))
    if currentDistance < closestDistance:
        closestDistance = currentDistance
        waypointNow = waypoints[:, i]
        waypointIndexNow = i

# flagg som sier om vi må reversere ruten eller ikke
reverse_flag = False

# kode som finner det waypointet med best vinkel for roboten å starte i
# sjekker først om roboten starter utenfor sirkelen, man må finne tangent i motsatt sirkel hvis den starter inni
if np.abs(x.item(0)) > circleRadius * 2 or np.abs(x.item(1)) > circleRadius:

    angle_counter = waypointIndexNow

    # hvis vi er i øvre halvdel av waypoints setter vi limit til max og vice versa
    if angle_counter >= num_waypoints:
        circle_limit = num_waypoints * 2
    else:
        circle_limit = num_waypoints

    # finner ut om vi er nærmest negativ eller positiv x-sirkel og lagrer sentrum av sirklene
    if x.item(0) > 0:
        circle_centre = np.array([circleRadius, float(0)])
    else:
        circle_centre = np.array([-circleRadius, float(0)])

    # lagrer robots posisjon i np.array
    robot_pos = np.array([x.item(0), x.item(1)])

    # setter startvinkel til den nærmeste vi har funnet
    best_angle = np.abs(
        get_angle(circle_centre, (waypoints[0, angle_counter], waypoints[1, angle_counter]), robot_pos) - np.pi / 2)
    best_angle_index = waypointIndexNow
    for i in range(int(np.floor(num_waypoints))):
        currentWaypoint = ([waypoints[0, angle_counter], waypoints[1, angle_counter]])
        # finner vinkelen til hvert waypoint
        current_angle = get_angle(circle_centre, currentWaypoint, robot_pos)

        # vi er ute etter vinkeler nærme 3pi/2 og pi/2 og må sammenligne de
        if current_angle > np.pi:
            currentAngle = 2 * np.pi - current_angle

        # trekker fra pi/2 og finner absoluttverdi så vi kan sammenligne alle tallene
        current_angle = np.abs(current_angle - np.pi / 2)

        if current_angle < best_angle:
            best_angle = current_angle
            waypointNow = waypoints[:, angle_counter]
            best_angle_index = angle_counter
            # den snur i origo hvis den kommer fra negativ y og x=0 så må legge inn ekstra betingelse
            if angle_counter < waypointIndexNow and angle_counter != 0:
                reverse_flag = True
        angle_counter = angle_counter + 1
        if angle_counter >= circle_limit:
            angle_counter = angle_counter - num_waypoints

    waypointIndexNow = best_angle_index
else:
    # dette er hvis den starter inni en sirkel
    angle_counter = waypointIndexNow

    if angle_counter >= num_waypoints:
        circle_limit = num_waypoints
    else:
        circle_limit = num_waypoints * 2

    if x.item(0) > 0:
        circle_centre = np.array([-circleRadius, float(0)])
    else:
        circle_centre = np.array([circleRadius, float(0)])

    robot_pos = np.array([x.item(0), x.item(1)])

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

# sette waypointNow til array som har 2 elementer med 1 verdi i hver
waypointNow.shape = (2, 1)

# hvor nærme vi må være waypoint før vi går til neste
distance_tolerance = 0.01

# starte simulasjon
r.step()
count = 0

while count < countMax:
    x = r.get_poses()
    count = count + 1

    # finne present location og orientation
    xRob = float(x.item(0))
    yRob = float(x.item(1))
    psiRob = float(x.item(2))

    # hente koordinatene til current waypoint
    wayPtX = waypointNow[0]
    wayPtY = waypointNow[1]

    # sjekker hvor langt unna roboten er current waypoint
    distance = np.square(find_distance(xRob, wayPtX, yRob, wayPtY))

    # hvis vi er nærmere enn det vi har satt som distance_tolerance så bytter vi waypoint
    if distance <= distance_tolerance:
        # går bakover i waypoints array hvis reverse_flag=true
        if reverse_flag:
            waypointIndexNow = waypointIndexNow - 1
        else:
            waypointIndexNow = waypointIndexNow + 1

        # sjekker om vi har gått igjennom alle waypoints
        if waypointIndexNow >= 2 * num_waypoints:
            waypointIndexNow = 0
        # må ha elif her eller så bytter den fra 2*num til 0 for så å bytte med en gang til num*2-1 igjen og igjen
        elif waypointIndexNow <= 0:
            waypointIndexNow = num_waypoints * 2 - 1

        waypointNow = waypoints[:, waypointIndexNow]
        waypointNow.shape = (2, 1)

    # lager array med current koordinater
    pos = np.array([xRob, yRob])
    pos.shape = (2, 1)

    # hente velocity fra rps funksjonen
    vel = position_controller(pos, waypointNow)

    # konvertere velocity så den kan brukes på unicycle
    uni_vel = si_to_uni_dyn(vel, x)
    r.set_velocities(0, uni_vel)

    # simuler 1 step
    r.step()

r.call_at_scripts_end()
