import random
import time

# ---------------------------
# Environment Setup
# ---------------------------
ROWS = 5
COLS = 5

# Tiles:
# 0 = clean
# 1 = dirt
# -1 = obstacle
# C = charging station

# Initialize environment
env = [[0 for _ in range(COLS)] for _ in range(ROWS)]

# Place some fixed obstacles
env[1][2] = -1
env[3][1] = -1

# Charging station
charge_pos = (0, 0)

# Initial position
agent_pos = [0, 0]

battery = 20
MAX_BATTERY = 20


# ---------------------------
# Utility Functions
# ---------------------------
def display_env():
    for r in range(ROWS):
        row = ""
        for c in range(COLS):
            if (r, c) == tuple(agent_pos):
                row += " A "
            elif (r, c) == charge_pos:
                row += " C "
            elif env[r][c] == -1:
                row += " X "
            elif env[r][c] == 1:
                row += " D "
            else:
                row += " . "
        print(row)
    print()


def generate_random_dirt():
    r = random.randint(0, ROWS - 1)
    c = random.randint(0, COLS - 1)
    if env[r][c] == 0:  # only put dirt on clean tile
        env[r][c] = 1


def move_agent():
    global battery

    # Possible moves: Up, Down, Left, Right
    moves = [(0,1), (1,0), (0,-1), (-1,0)]
    random.shuffle(moves)

    for dr, dc in moves:
        nr = agent_pos[0] + dr
        nc = agent_pos[1] + dc

        # Check boundaries and obstacles
        if 0 <= nr < ROWS and 0 <= nc < COLS and env[nr][nc] != -1:
            agent_pos[0], agent_pos[1] = nr, nc
            battery -= 1
            return


def clean_tile():
    r, c = agent_pos
    if env[r][c] == 1:
        env[r][c] = 0
        print("Cleaned dirt at", (r, c))


def go_to_charger():
    """Move agent one step toward charger."""
    global battery

    ar, ac = agent_pos
    cr, cc = charge_pos

    # Move vertically toward charging station
    if ar < cr:
        agent_pos[0] += 1
    elif ar > cr:
        agent_pos[0] -= 1
    # Move horizontally
    elif ac < cc:
        agent_pos[1] += 1
    elif ac > cc:
        agent_pos[1] -= 1

    battery -= 1


# ---------------------------
# AGENT LOOP (Main Behavior)
# ---------------------------
steps = 50  # run simulation for 50 steps

for t in range(steps):
    print("\nStep:", t + 1)
    display_env()

    # Random dirt appears occasionally
    if random.random() < 0.3:
        generate_random_dirt()

    # 1. Battery low → Return to charging station
    if battery <= 5 and tuple(agent_pos) != charge_pos:
        print("Battery low! Returning to charger...")
        go_to_charger()
        continue

    # 2. At charging station → recharge
    if tuple(agent_pos) == charge_pos:
        battery = MAX_BATTERY
        print("Recharging... Battery full!")
        move_agent()
        continue

    # 3. Clean if dirty
    clean_tile()

    # 4. Move to next tile
    move_agent()

    print("Battery =", battery)

    time.sleep(0.2)
