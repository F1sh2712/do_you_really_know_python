# Written by *** for COMP9021


# Write a function travel(s) that
# interprets a sequence of compass directions
# (given as a space-separated string),
# encodes the sequence numerically,
# and then traces the path on a grid.
#
# The argument s is a string that is possibly empty
# and has at most 9 directions.
#
# Each direction is mapped to a number 1–8:
# N=1, NE=2, E=3, SE=4, S=5, SW=6, W=7, NW=8.
#
# Step 1: The sequence of directions is first converted
# into a base-9 number (since 9 is larger than the
# maximum code value 8).
#
# Step 2: The same number is then interpreted in base-10.
#
# Step 3: The directions are simulated as moves on a grid:
#    - Starting at (0, 0),
#    - Each direction moves one step in its compass direction,
#    - Each visited location is recorded with the step number.
#
# The path creation stops as soon as the next location
# would already have been visited. A message is then printed
# indicating how many moves were performed.
# The word "move" is written in singular if exactly one move
# was completed, and in plural ("moves") otherwise.
#
# Finally, the function prints a simple text-based map
# of the visited path:
#   - The output uses the minimal number of lines,
#   - The drawing is shifted as far left as possible,
#   - No line has any trailing spaces,
#   - Each cell shows the step number when it was first visited,
#   - Empty cells are shown as spaces.
#
# Example 1 (all moves completed):
# travel("N SE S") prints
# Base-9 encoding of moves: 145
# In base-10, this number is 122.
#
# Path of travel (numbers show step order):
# 1
# 02
#  3
#
# Example 2 (not all moves completed):
# travel("N W S E") prints
# Base-9 encoding of moves: 1753
# In base-10, this number is 1344.
# I could perform only 3 moves.
#
# Path of travel (numbers show step order):
# 21
# 30
#
# (The fourth move E is not performed, since it would
# return to an already visited location.)
#
# Hint: One might consider representing the grid as a list of lists,
# but a much better approach is to use dictionaries:
#   - One dictionary, keyed by (y, x) coordinates,
#     maps each visited location to its step number,
#   - Another dictionary, keyed by y-coordinate,
#     tracks the maximum x reached in each row.
# Indeed, in this exercise, the path is guaranteed to fit within a 10×10 grid, 
# so storage and efficiency are not critical, but in a more general case
# where the paths could be very long, with no upper bound on their length,
# using dictionaries in this way would be much more convenient and scalable.


def travel(s):
    # If s is empty or s only contains spaces
    if not s or s.strip() == "":
        s = "DEFAULT"

    # Assigen DEFAULT to 0
    directions_dict = {"DEFAULT":"0", "N":"1", "NE":"2", "E":"3", "SE":"4", "S":"5", "SW":"6", "W":"7", "NW":"8"}
    
    # Direction N, (x, y+1)
    #           NE, (x+1, y+1)
    #           E, (x+1, y)
    #           SE, (x+1, y-1)
    #           S, (x, y-1)
    #           SW, (x-1, y-1)
    #           W, (x-1, y)
    #           NW, (x-1, y+1)
    directions_coordinates = {"DEFAULT":(0, 0), 
                        "N":(0, 1), "NE":(1, 1), 
                        "E":(1, 0), "SE":(1, -1), 
                        "S":(0, -1), "SW":(-1, -1), 
                        "W":(-1, 0), "NW":(-1, 1)}
    
    # Initial variables
    move_count = 0
    x, y = 0, 0 
    visited_location = {(x, y):0} # Start from origin
    directions_list = s.split()
    base_9_number = ""

    # Need to get base-9 number for all directions first, then determine if the path stops early
    # So need 2 individual for loops
    for i in directions_list:
        direction_number = directions_dict[i]
        # Covert directions string to base 9 number(string type)
        base_9_number = base_9_number + direction_number

    for i in directions_list:
         # Record visited location and its step number
        direction_x, direction_y = directions_coordinates[i] # x, y changes for this direction
        x, y = x + direction_x, y + direction_y # x, y reached after this move
        # Path creation stops if next location has been visited
        if (x, y) in visited_location:
            break
        move_count = move_count + 1
        visited_location[(x, y)] = move_count
    
    # Convert base 9 number to base 10 number(int type)
    base_10_number = int(base_9_number, 9)

    # Create grid columns and rows through y and x coordinates in the path
    x_list = []
    y_list = []

    for coordinate in visited_location:
        x_list.append(coordinate[0])
        y_list.append(coordinate[1])

    max_y, min_y = max(y_list), min(y_list) # Define rows
    max_x, min_x = max(x_list), min(x_list) # Defind columns

    # Generate path from top to bottom, from left to right
    path = []
    for row in range(max_y, min_y - 1, -1):
        path_line = ""
        for col in range(min_x, max_x + 1):
            if (col, row) in visited_location:
                path_line = path_line + str(visited_location[(col, row)]) # If visited, get step number
            else:
                path_line = path_line + " " #If not visited, space
        path.append(path_line)

    # Print result
    print(f"Base-9 encoding of moves: {base_9_number}")
    print(f"In base-10, this number is {str(base_10_number)}.")

    if move_count < len(directions_list) and move_count > 1:
        print(f"I could perform only {move_count} moves.")
    elif move_count < len(directions_list) and move_count == 1:
        print("I could perform only 1 move.")
    
    print("\nPath of travel (numbers show step order):")
    for line in path:
        print(line.rstrip()) # No trailing space
    # REPLACE THE PASS STATEMENT ABOVE WITH YOUR CODE

# POSSIBLY DEFINE HELPER FUNCTIONS

if __name__ == '__main__':
    travel("N S N S N")