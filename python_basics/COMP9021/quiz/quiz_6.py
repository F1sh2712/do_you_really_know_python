# Written by ***s for COMP9021


# Write a program that:
# 1. Reads two integers from the user: a random seed and a strictly
#    positive density value.
# 2. Generates a 10x10 grid of '*' and ' ' (space) characters using
#    the given seed and density. Each cell contains '*' with
#    probability 1 - 1/density and ' ' otherwise.
# 3. Displays the generated grid.
# 4. Computes and prints the area of the largest parallelogram with
#    horizontal top sides. The parallelogram may have straight,
#    left-slanted, or right-slanted sides.
# 5. If no parallelogram exists, prints an appropriate message.
#
# Notes:
# - Steps 1, 2, and 3 are already implemented in the provided stub.
#   Only steps 4 and 5 need be implemented.
# - Implement the function size_of_largest_parallelogram() and, if
#   necessary, any helper functions to compute the largest area.
# - The grid is available as a list of lists called `grid`.
# - Do not modify the grid generation code.
#
# Definition of a parallelogram:
# - A parallelogram consists of a line with at least 2 consecutive '*',
#   with at least one line below that has the same number of consecutive '*'.
# - All those lines are aligned vertically (rectangle), e.g.:
#       ***
#       ***
#       ***
#       ***
# - Or consecutive lines move to the left by one position, e.g.:
#       ***
#      ***
#     ***
#    ***
# - Or consecutive lines move to the right by one position, e.g.:
#       ***
#        ***
#         ***
#          ***
#
# Hints:
# - Consider each possible top row segment as a potential top side.
# - For each segment, check how far it can extend downwards
#   in each direction while keeping the parallelogram shape.
# - Keep track of the maximum area found.


from random import seed, randrange
import sys


dim = 10

def display_grid():
    print(' ', '-' * (2 * dim + 3))
    for row in grid:
        print('  |', *row, '|') 
    print(' ', '-' * (2 * dim + 3))

def size_of_largest_parallelogram():
    size_list = [] 
    for row in range(dim):
        for col in range(dim):
            # If this position is *, make it as starting point
            if grid[row][col] == '*':
                temp_rec = []
                temp_left = []
                temp_right = []

                # The min size of parallelogram is 2*2
                check_rectangle(row, col, 2, 2, temp_rec)
                check_left_parallelogram(row, col, 2, 2, temp_left)
                check_right_parallelogram(row, col, 2, 2, temp_right)

                if temp_rec:
                    size_list.append(max(temp_rec))
                if temp_left:
                    size_list.append(max(temp_left))
                if temp_right:
                    size_list.append(max(temp_right))
    
    if size_list:
        return max(size_list)
    else:
        return

    # REPLACE PASS ABOVE WITH YOUR CODE

# Base represents how many columns and height represents how many rows
def check_rectangle(row, col, base, height, temp_rec):
    for h in range(height):
        for b in range(base):
            # Should in the grid range 10*10
            if 0 <= (row + h) < dim and 0 <= (col + b) < dim:
                # If there is a space in this range, rectangle does not exist
                if grid[row + h][col + b] == ' ':
                    return
            else:
                return
            
    size = base * height
    temp_rec.append(size)

    check_rectangle(row, col, base + 1, height, temp_rec)
    check_rectangle(row, col, base, height + 1, temp_rec)

def check_left_parallelogram(row, col, base, height, temp_left):
    for h in range(height):
        # Gonging left makes col - 1 for each row's starting point
        current_col = col - h
        for b in range(base):
            # Should in the grid range 10*10
            if 0 <= (row + h) < dim and 0 <= (current_col) < dim and 0 <= (current_col + b) < dim:
                # If there is a space in this range, rectangle does not exist
                if grid[row + h][current_col + b] == ' ':
                    return
            else:
                return
    
    size = base * height
    temp_left.append(size)

    check_left_parallelogram(row, col, base + 1, height, temp_left)
    check_left_parallelogram(row, col, base, height + 1, temp_left)

def check_right_parallelogram(row, col, base, height, temp_right):
    for h in range(height):
        # Gonging right makes col + 1 for each row's starting point
        current_col = col + h
        for b in range(base):
            # Should in the grid range 10*10
            if 0 <= (row + h) < dim and 0 <= (current_col) < dim and 0 <= (current_col + b) < dim:
                # If there is a space in this range, rectangle does not exist
                if grid[row + h][current_col + b] == ' ':
                    return
            else:
                return
    
    size = base * height
    temp_right.append(size)

    check_right_parallelogram(row, col, base + 1, height, temp_right)
    check_right_parallelogram(row, col, base, height + 1, temp_right)
# POSSIBLY DEFINE OTHER FUNCTIONS

try: 
    for_seed, density = (int(x) for x in input('Enter two integers, the second '
                                               'one being strictly positive: '
                                              ).split()
                    )
    if density <= 0:
        raise ValueError
except ValueError:
    print('Incorrect input, giving up.')
    sys.exit()

seed(for_seed)
grid = [[{0: ' ', 1: '*'}[randrange(density) != 0]
             for _ in range(dim)
        ] for _ in range(dim)
       ]
print('Here is the grid that was generated:')
display_grid()
size = size_of_largest_parallelogram()

if size:
    print(f"The largest parallelogram with horizontal sides has a size of {size}.")
else:
    print("There is no parallelogram with horizontal sides.")
# ADD CODE HERE