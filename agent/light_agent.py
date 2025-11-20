import random
import time

# Device states
lights = "OFF"
fans = "OFF"

def sense_environment():
    """Generate random sensor readings."""
    occupancy = random.randint(0, 40)         # number of students
    natural_light = random.randint(100, 900)  # lux
    time_of_day = random.choice(["Lecture", "Break"])
    return occupancy, natural_light, time_of_day


def decide_action(occupancy, natural_light, time_of_day):
    """Return updated states of lights and fans."""
    # --- LIGHT RULES ---
    if occupancy > 0 and natural_light < 300:
        lights_state = "ON"
    else:
        lights_state = "OFF"

    # --- FAN RULES ---
    if occupancy > 0 and time_of_day == "Lecture":
        fans_state = "ON"
    else:
        fans_state = "OFF"

    return lights_state, fans_state


def run_agent(cycles=6):
    print("=== Smart Classroom Lighting Agent (No Class) ===")

    global lights, fans

    for i in range(cycles):
        print(f"\nCycle {i+1}:")

        # Step 1: Sense
        occupancy, natural_light, time_of_day = sense_environment()
        print(f"Occupancy: {occupancy}")
        print(f"Natural Light: {natural_light} Lux")
        print(f"Time: {time_of_day}")

        # Step 2: Decide
        lights, fans = decide_action(occupancy, natural_light, time_of_day)

        # Step 3: Act
        print(f"Lights: {lights}, Fans: {fans}")

        time.sleep(1)


# Run
run_agent()