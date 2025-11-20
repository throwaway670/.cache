from collections import deque

def solve_maze(maze, start, end):
    queue = deque([(start, [start])])
    visited = set()
    
    while queue:
        (x, y), path = queue.popleft()
        
        if (x, y) == end:
            return path
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]: # Directions
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
                
    return "No path found"

# 0 = Open, 1 = Wall
grid = [
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 0, 0]
]

print("--- Maze Solver ---")
path = solve_maze(grid, (0,0), (3,3))
print(f"Shortest Path: {path}")