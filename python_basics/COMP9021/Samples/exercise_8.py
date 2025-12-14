# You can assume that the function argument, s,
# is a string consisting of nothing but distinct letters
# (a letter can occur twice, once in uppercase, once in
# lowercase).
#
# Returns the set of all strings consisting of letters in s,
# MAINTAINING THE ORDER THEY HAVE in s,
# a vowel in the string either ending the string or being
# followed by a consonant, a consonant in the string either
# ending the string or being followed by a vowel.
#
# Vowels are A, E, I, O, U and their lowercase counterparts.


def f(s):
    '''
    >>> f('')
    {''}
    >>> f('A') == {'', 'A'}
    True
    >>> f('aeio') == {'', 'e', 'a', 'o', 'i'}
    True
    >>> f('AB') == {'', 'A', 'B', 'AB'}
    True
    >>> f('ABE') == {'', 'E', 'A', 'B', 'ABE', 'BE', 'AB'}
    True
    >>> f('AEB') == {'', 'A', 'AB', 'B', 'E', 'EB'}
    True
    >>> f('bBaA') == {'', 'a', 'ba', 'B', 'A', 'b', 'bA', 'BA', 'Ba'}
    True
    >>> f('BCADae') == {'', 'CAD', 'BADe', 'e', 'Ca', 'CADe', \
'ADa', 'Ba', 'BADa', 'Ce', 'ADe', 'a', 'Da', 'CA', 'Be', \
'BA', 'C', 'D', 'A', 'CADa', 'AD', 'B', 'BAD', 'De'}
    True
    '''
    # INSERT YOUR CODE HERE
    solutions = set()

    recursion(s, 0, "", solutions)

    return solutions
# POSSIBLY DEFINE OTHER FUNCTIONS
def check_valid(letters, next_char):
    vowels = "AEIOUaeiou"

    if not letters:
        return True
    if letters[-1] not in vowels and next_char in vowels:
        return True
    if letters[-1] in vowels and next_char not in vowels:
        return True
    
def recursion(s, index, letters, solutions):
    if len(letters) != len(set(letters)):
        return
    
    solutions.add(letters)

    for next in range(index, len(s)):
        next_char = s[next]
        if check_valid(letters, next_char):
            recursion(s, next + 1, letters + next_char, solutions)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
