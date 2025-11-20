import random
import collections

W, H = 5, 5
OBSTACLE_COUNT = 5
STEPS = 20
MAX_BATTERY = 15
FIXED_OBSTACLE_LOCATIONS = [(1, 1), (2, 2), (3, 3), (1, 3), (3, 1)]

OBSTACLE = -1
EMPTY = 0
DIRT = 1
RECHARGE_STATION = 2

class VacuumAgent:
    def __init__(self, W, H, obstacle_count, max_battery):
        self.W = W
        self.H = H
        self.MAX_BATTERY = max_battery
        self.grid = [[EMPTY] * W for _ in range(H)]

        self.recharge_y, self.recharge_x = 0, 0
        self.agent_y, self.agent_x = H - 1, W - 1
        self.battery = self.MAX_BATTERY
        self.dirt_seed_count = 5

        self._place_station()
        self._place_obstacles(obstacle_count)
        self._seed_dirt(self.dirt_seed_count)

    def _place_station(self):
        self.grid[self.recharge_y][self.recharge_x] = RECHARGE_STATION

    def _place_obstacles(self, count):
        for y, x in FIXED_OBSTACLE_LOCATIONS:
            if self.grid[y][x] == EMPTY:
                self.grid[y][x] = OBSTACLE

    def _seed_dirt(self, count):
        i = 0
        while i < count:
            y, x = random.randrange(self.H), random.randrange(self.W)
            if self.grid[y][x] == EMPTY:
                self.grid[y][x] = DIRT
                i += 1

    def get_neighbors(self, y, x):
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.H and 0 <= nx < self.W and self.grid[ny][nx] != OBSTACLE:
                yield ny, nx

    def bfs(self, start, targets):
        q = collections.deque([start])
        prev = {start: None}

        if not targets:
            return None

        while q:
            cur = q.popleft()
            if cur in targets:
                path = []
                while cur:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]

            for n in self.get_neighbors(*cur):
                if n not in prev:
                    prev[n] = cur
                    q.append(n)
        return None

    def get_current_dirt_locations(self):
        return {(y, x) for y in range(self.H) for x in range(self.W) if self.grid[y][x] == DIRT}

    def make_decision_and_move(self):
        current_pos = (self.agent_y, self.agent_x)
        recharge_pos = (self.recharge_y, self.recharge_x)
        dirt_locations = self.get_current_dirt_locations()

        path = None

        station_path = self.bfs(current_pos, {recharge_pos})
        cost_to_station = len(station_path) - 1 if station_path else float('inf')

        if self.battery <= cost_to_station and station_path:
            path = station_path
        elif dirt_locations:
            dirt_path = self.bfs(current_pos, dirt_locations)
            if dirt_path:
                cost_to_dirt = len(dirt_path) - 1
                dirt_target = dirt_path[-1]
                path_back_home = self.bfs(dirt_target, {recharge_pos})
                if path_back_home:
                    cost_back_home = len(path_back_home) - 1
                    total_mission_cost = cost_to_dirt + cost_back_home
                    if self.battery > total_mission_cost:
                        path = dirt_path
                    else:
                        path = station_path
                else:
                    path = station_path
            else:
                path = station_path
        else:
            path = station_path

        next_pos = None

        if path and len(path) > 1:
            next_pos = path[1]
        elif not path:
            candidates = list(self.get_neighbors(self.agent_y, self.agent_x))
            if candidates:
                next_pos = random.choice(candidates)
        if next_pos is None and (self.agent_y, self.agent_x) == recharge_pos and self.battery == self.MAX_BATTERY and dirt_locations:
            candidates = list(self.get_neighbors(self.agent_y, self.agent_x))
            candidates = [c for c in candidates if c != recharge_pos]
            if candidates:
                next_pos = random.choice(candidates)
        if next_pos:
            self.agent_y, self.agent_x = next_pos
        if (self.agent_y, self.agent_x) == recharge_pos:
            self.battery = self.MAX_BATTERY
        else:
            self.battery = max(0, self.battery - 1)
        if self.grid[self.agent_y][self.agent_x] == DIRT:
            self.grid[self.agent_y][self.agent_x] = EMPTY
        if not self.get_current_dirt_locations():
            self._seed_dirt(self.dirt_seed_count)

    def display(self, step):
        dirt_remaining = len(self.get_current_dirt_locations())
        print(f"Step {step}: Battery {self.battery} | Remaining Dirt {dirt_remaining}")

        for r in range(self.H):
            print(''.join(
                'A' if (r, c) == (self.agent_y, self.agent_x) else
                'R' if self.grid[r][c] == RECHARGE_STATION else
                'O' if self.grid[r][c] == OBSTACLE else
                'D' if self.grid[r][c] == DIRT else
                '.' for c in range(self.W)
            ))
        print("-" * (self.W + 8))

if __name__ == "__main__":
    print("--- Smart Vacuum Cleaner Agent Simulation ---")
    agent = VacuumAgent(W, H, OBSTACLE_COUNT, MAX_BATTERY)
    dirt_remaining = len(agent.get_current_dirt_locations())
    for t in range(1, STEPS + 1):
        agent.make_decision_and_move()
        if t == 1 or t == STEPS or agent.battery <= 0 or dirt_remaining != len(agent.get_current_dirt_locations()):
            agent.display(t)
            dirt_remaining = len(agent.get_current_dirt_locations())
            if agent.battery <= 0 and (agent.agent_y, agent.agent_x) != (agent.recharge_y, agent.recharge_x):
                print("CRITICAL ERROR: AGENT RAN OUT OF POWER BEFORE REACHING THE STATION. SIMULATION HALTED.")
                break
    print("--- Simulation Finished ---")