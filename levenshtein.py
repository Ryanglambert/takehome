"""
##### Problem #####
levenshtein distance between two strings

Types of operations allowed in computing levenshtein distance

Each of these counts as "1 unit of distance"
    - Insertions
    - Replacements

##### Solution ####
I chose to solve this with dynamic programming for two reasons: 
    - I try to avoid recursion as it is less readable and less pythonic
    - Dynamic programming avoids recalculating information twice

This is the top down approach. (exhaustive O(N^2))
We use this approach because it has to be deterministic.
Although it's O(n^2),
it's not too much of a concern given that words are typically on
the order of magnitude of 1s or 10s of letters.

Bottom up would be iterative and approximate. (what is used in reinforcement learning), not what we want.

"""


def _initialize_position_scores(len_1: int, len_2: int):
    "Initializes a dictionary mapping positions in a dp matrix to their corresponding scores"
    # this is done so in this function so that each character comparison has a
    # `upper_left`, `upper` and `left` count to use in it's argmin + 1 calculation
    position_scores = {}

    # initialize col 0
    for i in range(len_1 + 1):
        position_scores[(i, 0)] = i

    # initialize row 0
    for i in range(1, len_2 + 1):
        position_scores[(0, i)] = i
    return position_scores
    


def distance(str1: str, str2: str):
    """
    """
    str_1_len = len(str1)
    str_2_len = len(str2)
    position_scores = _initialize_position_scores(str_1_len, str_2_len)
    # since row 0 and col 0 are initialized with distances from a null character
    # it's appropriate for the indices in enumerate to start at
    # 1 instead of 0.
    for i, letter_1 in enumerate(str1, start=1):
        for j, letter_2 in enumerate(str2, start=1):
            if letter_1 != letter_2:
                left = position_scores[(i, j-1)]
                up = position_scores[(i-1, j)]
                up_left = position_scores[(i-1, j-1)]
                lowest = min((left, up, up_left))
                position_scores[(i, j)] = lowest + 1

            else:
                up_left = position_scores.get((i - 1, j - 1), 0)
                position_scores[(i, j)] = up_left
    final_position_score = position_scores[(str_1_len, str_2_len)]
    return final_position_score


if __name__ == '__main__':
    print('in module quick tests...')
    assert distance('azced', 'abcdef') == 3
    assert distance('s', 'stuff') == 4
    assert distance('stuffs', 'stuff') == 1
    assert distance('stuff', 'stuff') == 0
    assert distance('cats', 'stacks') == 4
    assert distance('cat', 'cut') == 1
    assert distance('geek', 'gesek') == 1
    assert distance('sunday', 'saturday') == 3
