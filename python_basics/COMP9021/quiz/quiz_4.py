# Written by *** for COMP9021


# Write a function products(max_product, *pairs_of_bounds) that
# computes all possible products of prime numbers chosen from
# several given integer ranges, not exceeding a given limit.
#
# The helper function sieve_of_primes_up_to(n) is provided.
#
# The arguments are:
#   - max_product: the largest allowed product value,
#   - *pairs_of_bounds: one or more pairs (a, b) defining
#     inclusive integer ranges, whose order does not matter.
#
# Step 1: Determine how far the sieve must go based on the
#         largest endpoint among all given ranges.
#
# Step 2: For each range, consider only the prime numbers
#         that lie within it.
#
# Step 3: Combine primes from each range successively,
#         keeping only results that do not exceed max_product.
#
# Step 4: When all ranges are processed, return the remaining
#         products in sorted order.
#
# Example 1 (two sets of bounds):
# The first range provides primes {2, 3, 5, 7},
# the second provides primes {3, 5, 7, 11}.
# All cross-products of one prime from each range ≤ 200 are kept:
# products(200, (2, 10), (3, 12)) returns
# [6, 9, 10, 14, 15, 21, 22, 25, 33, 35, 49, 55, 77]
#
# Example 2 (one set of bounds):
# Primes from 2 to 20 are {2, 3, 5, 7, 11, 13, 17, 19}.
# Since there is only one range, the result simply lists
# all those primes not exceeding 100:
# products(100, (2, 20)) returns
# [2, 3, 5, 7, 11, 13, 17, 19]
#
# Example 3 (three sets of bounds):
# Ranges yield primes {2, 3, 5, 7}, {5, 7, 11, 13}, and {3, 5, 7, 11}.
# All products of one prime from each range not exceeding 500 are kept:
# products(500, (2, 10), (5, 15), (3, 12)) returns
# [30, 42, 45, 50, 63, 66, 70, 75, 78, 98, 99, 105, 110, 117, 125,
# 130, 147, 154, 165, 175, 182, 195, 231, 242, 245, 273, 275, 286,
# 325, 343, 363, 385, 429, 455]
#
# Hint:
#   Starting from the product 1,
#   maintain a set of current products, and update it after each range
#   with the new valid products before moving on to the next range.
#
# Note: max_product and the bounds of the ranges
# can be much larger than in the sample examples.


from math import sqrt


def sieve_of_primes_up_to(n):
    sieve = [True] * (n + 1)
    for p in range(2, round(sqrt(n)) + 1):
        if sieve[p]:
            for i in range(p * p, n + 1, p):
                sieve[i] = False
    return sieve

def products(max_product, *pairs_of_bounds):
    # Decide sieve largest ranges
    ranges = []
    for a, b in pairs_of_bounds:
        if not a or not b or (not a and not b):
            return []
        else:
            large = min(max_product, max(a, b)) # This is the largest endpoint for each pair, any number greater than max_products is meaningless
            ranges.append(large)
        
    # print(ranges)
    # Get primes for the largest for all endpoints
    all_sieve = sieve_of_primes_up_to(max(ranges))
    # all_primes = []
    # for i in range(2, len(all_sieve)): # Ignore 0 and 1
    #     if all_sieve[i]:
    #         all_primes.append(i)
    # print(all_primes)
    
    # Gnerate products
    result_list = [1]
    for a, b in pairs_of_bounds:
        products_list = []

        for i in range(min(a, b), max(a, b) + 1): # Sequnce in a, b is not sensitive like (12, 3)
           if i > max(ranges):
               break
            
           # if i in all_primes: this is too slow for large number
           if i > 1 and all_sieve[i] == True: # Ignore 0 and 1 for True
               for j in result_list:
                   product = i * j

                   if product <= max_product:
                       products_list.append(product)
                   else:
                       break
        
        # Sort products_list and update result_list
        # Unordered result_list causes for loop stops earlier
        products_list.sort()
        result_list = products_list

    # Delete duplicate products in result and sort
    result_list = list(set(result_list))
    result_list.sort()
    return result_list
    # REPLACE THE RETURN STATEMENT ABOVE WITH YOUR CODE


if __name__ == '__main__':
    # print(products(200, (2, 10), (3, 12)))
    # print(products(500, (2, 10), (5, 15), (3, 12)))
    # print(products(200, (2, 10), (3, 1234567890)))
    # print(products(500, (10, 2), (15, 5), (12, 3)))
    # print(products(123456789, (1234, 22345), (123, 1253), (456, 4678)))
    print(products(200, (2, 10), (201, 87987987987987)))