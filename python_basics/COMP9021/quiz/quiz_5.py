# Written by *** for COMP9021


# Write a program that reads a non-negative integer interpreted as an
# encoding of a finite set of integers (positive, negative, and zero),
# and outputs both the decoded set and the corresponding running-sum set.
# Each set is encoded as an integer whose binary representation specifies
# which integers belong to it, according to the following rule:
# - Even bit positions (0, 2, 4, …) correspond to non-negative integers
# 0, 1, 2, …
# - Odd bit positions (1, 3, 5, …) correspond to negative integers
# –1, –2, –3, …
# Thus, for example, bit 0 encodes 0, bit 1 encodes –1, bit 2 encodes 1,
# bit 3 encodes –2, and so on.
#
# Step 1: Prompt the user repeatedly until a valid non-negative integer
#         is entered. If the input is invalid (non-integer or negative),
#         display a message and ask again.
# Step 2: Decode the integer into the set it represents.
#         Display the decoded set in standard mathematical notation,
#         with elements sorted numerically, e.g. {–2, 0, 3}.
# Step 3: Compute the running-sum set from the decoded set:
#         - Sort the decoded set from smallest to largest,
#         - Traverse the sorted elements in order,
#         - After each element, add it to a running sum;
#           if the new sum from all the previous ones,
#           include it in the running-sum set.
# Step 4: Encode the running-sum set as an integer using the same
#         bit-position rule.
# Step 5: Display the results compactly:
#         - One line for the original decoded set
#         - One line for the running-sum set and its encoding integer
#
# Example: Input:
#          Input a non-negative integer: 41
#            Since 41 in binary is 101001,
#            bits 0, 3, and 5 are set, corresponding to elements 0, -2, –3,
#            resulting in the set, displaying elements from smallest to
#            largest, {-3, -2, 0}.
#            Running sums: -3, -5, -5
#            Encoding integer of running-sum set: 544
#          Program output:
#          Original set (decoded from 41): {-3, -2, 0}
#          Running-sum set (encoded as 544): {-5, -3}
#
# Hint: You may find the Python module itertools useful.
#       Try using help(itertools) to discover the functions it provides.
#
# Note:
# - If the input integer is 0, no bits are set,
#   so it represents the empty set, here displayed as {}.
# - The running-sum set provides a compact representation of cumulative sums
#   of the elements in the original set.
# - The printed sets are sorted for clarity and ease of interpretation.

import itertools


# INSERT YOUR CODE HERE

# help(itertools)

# Check if input is valid
def validate(input):
    try:
        input = input.strip()
        integer = int(input)

        # Only accept non-negative integer
        if integer >= 0:
            return True
        else:
            return False
        
        # If cannot convert to integer, return false
    except Exception:
        return False

# Decode non negative integer to a set
def decode_integer(integer):
    decode_set = []
    binary_str = bin(integer)[2:]
    binary_str = binary_str[::-1] # Reverse binary

    # Check 1 for each position
    for i in range(len(binary_str)):
        if binary_str[i] == "1":
            # If in odd position
            if i % 2 == 1:
                decode_set.append(-i // 2)
            # If in even position
            else:
                decode_set.append(i // 2)
    
    # Sort set
    decode_set = sorted(decode_set)

    return decode_set

# Running sum set
def running_sum(decode_set):
    new_sum = 0
    sum_set = []

    for i in decode_set:
        # Differ from all of previous ones
        new_sum = new_sum + i
        if new_sum not in sum_set:   
            sum_set.append(new_sum)
        else:
            continue

    # Sort set
    sum_set = sorted(sum_set)

    return sum_set

# Encode running sum set to integer
def encode_set(sum_set):
    bit_position = [] # Store which position has bit
    binary_str = ""

    if sum_set == []:
        return 0
    
    for num in sum_set:
        # If odd position
        if num < 0:
            bit_position.append(-2 * num - 1)
        # If even position
        else:
            bit_position.append(2 * num)
    
    for i in range(max(bit_position) + 1):
        if i in bit_position:
            binary_str = binary_str + "1"
        else:
            binary_str = binary_str + "0"

    binary_str = binary_str[::-1] # Reverse to get binary 
    encode_integer = int(binary_str, 2)

    return encode_integer

# print(decode_integer(41))
# print(running_sum([-3, -2, 0]))
# print(encode_set([-5, -3]))

# Repeatedly prompt
round = 0
while True:
    round = round + 1

    if round == 1:
        user_input = input("Input a non-negative integer: ")
    else:
        user_input = input("Incorrect input, please enter a non-negative integer: ")

    # Check validation, if not valid, prompt again
    if not validate(user_input):
        continue
    else:
        user_input = user_input.strip()
        decode_numbers = decode_integer(int(user_input))

        sum_numbers = running_sum(decode_numbers)

        encode_as = encode_set(sum_numbers)

        # Transfer list in set format {}
        decode_set = ""
        for dn in decode_numbers:
            decode_set = decode_set +  (str(dn)) + ", "
        decode_set = "{" + decode_set.rstrip(", ") + "}"

        sum_set = ""
        for sn in sum_numbers:
            sum_set = sum_set + (str(sn)) + ", "
        sum_set = "{" + sum_set.rstrip(", ") + "}"

        # Print result
        print(f"Original set (decoded from {int(user_input)}): {decode_set}")
        print(f"Running-sum set (encoded as {encode_as}): {sum_set}")

        break