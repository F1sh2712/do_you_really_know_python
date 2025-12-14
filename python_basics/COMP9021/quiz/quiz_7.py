# Written by Eric Martin for COMP9021


# Write a program that generates a random grid
# and computes how many constrained paths exist
# between two user-specified points.
# A constrained path:
# - moves only through open cells ('*');
# - never visits the same cell more than once;
# - never continues in the same direction (North, South, East, West)
#   for more than two steps in a row.
#
# Input
# The program first asks for four strictly positive integers:
# - seed: random seed (integer)
# - density: controls how likely each cell is open
# - width, height: grid dimensions
# It then asks for two points, using 1-based coordinates.
#
# Grid generation
# The grid is randomly filled with:
# - '*' for open cells,
# - ' ' (space) for blocked cells.
# You do not need to implement input handling or grid generation
# yourself; this code is already provided in the stub.
# The function that displays the grid is already written for you.
#
# Output
# After the grid is displayed, the program prints:
#   Computing all constrained paths from start (x1, y1) to target (x2, y2)...
# Then it reports one of the following:
# - No constrained paths exist.
# - Exactly one constrained path exists.
# - Total number of constrained paths found: N
# You must implement the code that produces this output.
#
# Implementation Details
# - The grid is available as a list of lists named grid.
# - Points are represented using a namedtuple Point(x, y).
# - The main function count_constrained_paths(start, target)
#   is provided in the stub. You are advised not to modify it;
#   if you do, then that is at your own risk.
# - You may define additional helper functions if needed.


from collections import namedtuple, defaultdict
from random import seed, randrange
import sys


Point = namedtuple('Point', 'x y')


def to_internal(x, y):
    return Point(x - 1, y - 1)

def print_grid():
    border = '-' * (2 * width + 3)
    print(' ', border)
    for row in grid:
        print('  |', *row, '|') 
    print(' ', border)

def is_valid(pt):
    return 0 <= pt.x < width and 0 <= pt.y < height


def count_constrained_paths(start, target):
    if start == target:
        return 1
    
    # No constrained paths exist if start or target is not open cell
    if grid[start.y][start.x] != '*' or grid[target.y][target.x] != '*':
        return 0

    direction_counter = {} # Counter of num of same directions
    visited_list = [(start.x, start.y)]
    num_path = [0] # Recursion cannot change int global

    find_path(start.x, start.y, target.x, target.y, visited_list, direction_counter, num_path)

    return num_path[0]
    # REPLACE THE RETURN STATEMENT ABOVE WITH YOUR CODE

# POSSIBLY DEFINE OTHER FUNCTIONS    
def find_path(current_x, current_y, target_x, target_y, visited_list, counter, num_path):
    directions_dic = {"N":(0, -1), "S":(0, 1), "E":(1, 0), "W":(-1, 0)}
    
    # If reach target point
    if current_x == target_x and current_y == target_y:
        num_path[0] += 1
        return
    
    for direction, move in directions_dic.items():
        # Check if same direction
        new_counter = {}
        if direction in counter:
            new_counter = counter
            new_counter[direction] += 1 
        else:
            new_counter = {direction:1}  
        
        next_x = current_x + move[0]
        next_y = current_y + move[1]

        # Cannot exceed grid range
        if 0 <= next_y < height and 0 <= next_x < width:
            # Reassign a new direction until it is open cell    
            if grid[next_y][next_x] != '*':
                continue

            # Reassign a new direction if goes same direction more than twice
            if new_counter[direction] > 2:
                continue

            if grid[next_y][next_x] == '*' and (next_x, next_y) not in visited_list:
                visited_list.append((next_x, next_y))
                find_path(next_x, next_y, target_x, target_y, visited_list, new_counter, num_path)
                # Prevent visited point for one path affect other paths
                visited_list.remove((next_x, next_y))

try:
    for_seed, density, width, height = (int(i) for i in
                    input('Enter four strictly positive integers '
                          '(seed, density, width, height): '
                         ).split()
                                       )
    if for_seed <= 0 or density <= 0 or height <= 0 or width <= 0:
        raise ValueError
except ValueError:
    print('Incorrect input, giving up.')
    sys.exit()
seed(for_seed)
grid = [[{0: ' ', 1: '*'}[randrange(density) != 0]
             for _ in range(width)
        ] for _ in range(height)
       ]
print('Generated grid '
      "(where '*' marks open cells and ' ' marks blocked cells):"
     )
print_grid()
try:
    start = to_internal(*(int(i) for i in
                           input('Enter coordinates of the starting point (x y), '
                                 'using 1-based indexing: '
                             ).split()
                         ) 
                       )
    if not is_valid(start):
        raise ValueError
except (TypeError, ValueError):
    print('Incorrect input, giving up.')
    sys.exit()
try:
    target = to_internal(*(int(i) for i in
                           input('Enter coordinates of the target point (x y), '
                                 'using 1-based indexing: '
                                ).split()
                          )
                        )
    if not is_valid(target):
        raise ValueError
except (TypeError, ValueError):
    print('Incorrect input, giving up.')
    sys.exit()

# ADD CODE GENERATING OUTPUT HERE
paths = count_constrained_paths(start, target)

print(f"Computing all constrained paths from start ({start.x+1}, {start.y+1}) to target ({target.x+1}, {target.y+1})...")
if paths == 0:
    print("No constrained paths exist.")
elif paths == 1:
    print("Exactly one constrained path exists.")
else:
    print(f"Total number of constrained paths found: {paths}")