# Note that NONE OF THE LINES THAT ARE OUTPUT HAS TRAILING SPACES.
#
# You can assume that vertical_bars() is called with nothing but
# integers at least equal to 0 as arguments (if any).


def vertical_bars(*x):
    '''
    >>> vertical_bars()
    >>> vertical_bars(0, 0, 0)
    >>> vertical_bars(4)
    *
    *
    *
    *
    >>> vertical_bars(4, 4, 4)
    * * *
    * * *
    * * *
    * * *
    >>> vertical_bars(4, 0, 3, 1)
    *
    *   *
    *   *
    *   * *
    >>> vertical_bars(0, 1, 2, 3, 2, 1, 0, 0)
          *
        * * *
      * * * * *
    '''
    
    # REPLACE PASS ABOVE WITH YOUR CODE
    if not x:
        return
    
    if all(num == 0 for num in x):
        return
    
    idx = len(x) - 1

    while idx >= 0 and x[idx] == 0:
        idx -= 1

    x = x[:idx + 1]

    from itertools import zip_longest

    max_length = max(x)

    columns = []

    for num in x:
        col = [' '] * (max_length - num) + ['*'] * num
        columns.append(col)

    for row in zip_longest(*columns, fillvalue=""):
        line = " ".join(row).rstrip()

        print(line)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
