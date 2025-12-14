# Written by *** for COMP9021


# Write a function monotone_lists(L) that
# splits the list L into consecutive sublists,
# each strictly monotonic (either strictly increasing
# or strictly decreasing).
#
# Assume that L is non-empty,
# all elements in L are strictly positive integers, and
# no two consecutive elements in L are equal.
#
# The function treats the input as if it were
# extended by a boundary value (0) at both the start
# and end, to help detect changes in monotonicity.
#
# It processes elements from left to right,
# grouping them into monotone sublists.
#
# When the monotone trend breaks, a new sublist begins.
#
# The output is a list of these monotone sublists,
# including the boundary values.
#
# Example 1: For L = [1, 2, 3, 2, 1, 2], the function returns
# [[0, 1, 2, 3], [3, 2, 1], [1, 2], [2, 0]].
#
# Example 2: For L = [1, 3, 2, 5, 4], the function returns
# [[0, 1, 3], [3, 2], [2, 5], [5, 4, 0]].
#
# Hint: The function uses comparisons among recent elements
# to decide whether to continue the current sublist or start a new one.

def monotone_lists(L):
    result_list = []
    
    # Add boundary value 0
    L.insert(0,0) # Start
    L.append(0) # End

    sub_list = [L[0]]

    # Boolean to determine it is increasing(T) or decreasing(F), default true increasing
    is_increasing = True

    # Comparison to make sublists
    for i in range(1, len(L)):

        # If increasing
        if is_increasing:
            if L[i] > L[i - 1]:
                sub_list.append(L[i])
            else:
                is_increasing = False
                # Add sublist to result list
                result_list.append(sub_list)
                # Reassign sublist to decreasing
                sub_list = [L[i - 1], L[i]]
        # If decreasing
        elif not is_increasing:
            if L[i] < L[i - 1]:
                sub_list.append(L[i])
            else:
                is_increasing = True
                # Add sublist to result list
                result_list.append(sub_list)
                # Reassign sublist to increasing
                sub_list = [L[i - 1], L[i]]

    # Add last sublist to result because final sublist with 0 in the end is out of loop
    result_list.append(sub_list)
    return result_list
    # REPLACE THE RETURN STATEMENT ABOVE WITH YOUR CODE

# ------------------------------------------------------------


# Write a function cycles(L) that
# takes a list L representing a permutation of
# the set {0, 1, ..., n - 1} for some n ≥ 0
# (when n is equal to 0, the set is empty).
#
# For each index i in L, the function finds a sequence
# of elements by repeatedly applying the mapping defined
# by L, starting at i, until the sequence returns to i.
#
# The function returns a dictionary where each key is an
# index i, and the value is the list of elements encountered
# in this cycle.
#
# Example 1: For L = [2, 0, 1], the function returns
# {0: [2, 1, 0], 1: [0, 2, 1], 2: [1, 0, 2]}.
#
# Example 2: For L = [2, 0, 1, 3, 5, 4], the function returns
# {
#     0: [2, 1, 0],
#     1: [0, 2, 1],
#     2: [1, 0, 2],
#     3: [3],
#     4: [5, 4],
#     5: [4, 5]
# }
#
# Hint: Consider tracking elements from i onward by
# following the mapping until the starting index is reached again.

def cycles(L):
    result_dict = {}
    
    for i in range(len(L)):
        value_list = []

        current_index = i

        # Keep cycling until the sequence returns to i
        while L[current_index] != i:
            value_list.append(L[current_index])
            current_index = L[current_index]

        # Add the last element in sequence to value list becasue it is out of the loop
        value_list.append(L[current_index])

        # Add key value pair to dictionary
        result_dict[i] = value_list

    return result_dict
    # REPLACE THE RETURN STATEMENT ABOVE WITH YOUR CODE

# if __name__ == '__main__':
#     print(monotone_lists([1, 2, 3, 2, 1, 2]))
#     print(cycles([2, 0, 1, 3, 5, 4]))