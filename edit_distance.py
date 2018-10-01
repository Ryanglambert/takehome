"""
levenshtein distance between two strings

Types of operations allowed in computing levenshtein distance

Each of these counts as "1 unit of distance"
- Insertions
- Replacements

"""


def _initialize_position_scores(len_1: int, len_2: int):
    position_scores = {}

    # initialize col 0
    for i in range(len_1 + 1):
        position_scores[(i, 0)] = i

    # initialize row 0
    for i in range(1, len_2 + 1):
        position_scores[(0, i)] = i
    return position_scores
    


def distance(str1: str, str2: str):
    # Goal: What is the levenshtein distance between str1 and str2
    """
    This problem is best solved as a top down dynamic programming problem. 

    Top down (all combinations) because it has to be 
    """
    str_1_len = len(str1)
    str_2_len = len(str2)
    position_scores = _initialize_position_scores(str_1_len, str_2_len)
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
    print('in module quick tests')
    assert distance('azced', 'abcdef') == 3
    assert distance('s', 'stuff') == 4
    assert distance('stuffs', 'stuff') == 1
    assert distance('stuff', 'stuff') == 0
    assert distance('cats', 'stacks') == 4
