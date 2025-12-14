from itertools import chain
from random import seed, shuffle
from collections import defaultdict


def get_card_pic(card):
    # card 表示 0 - 32 数字
    i, j = card // 8, card % 8
    # 四种花色
    card_decks = [127153, 127169, 127185, 127137]
    # 分别表示  7, 8, 9, 10, 11, 12 13  A
    offsets = [6, 7, 8, 9, 10, 12, 13, 0]
    # 返回对应的花色
    return chr(card_decks[i] + offsets[j])

# if all four cards share the same suit, then all four cards are discarded.
def check_four_cards_has_same_suits(counter, four_suits):
    if len(counter) == 1:
        # _, values = counter.popitem()
        values = list(counter.values())[0]
        # 都有相同的颜色
        if len(values) == 4:
            for i in values:
                # discarded
                four_suits[i] = -1
            # 都有相同的颜色
            return True

# there are two cards each of two different suits,
def check_two_cards_has_different_suits(counter, four_suits):
    #  there are two cards each of two different suits,
    if len(counter) == 2:
        suits1, suits2 = counter.values()
        if len(suits1) == len(suits2) == 2:
            for i in suits1 + suits2:
                four_suits[i] = -1
            # 返回结果
            return True
# exactly three cards of a suit are present, the two in the lowest-numbered locations are discarded.
def check_exactly_three_cards_same_suit(counter, four_suits):
    for suits in counter.values():
        if len(suits) == 3:
            # the two in the lowest-numbered locations are discarded.
            for i in suits[:2]:
                four_suits[i] = -1
            return True
        
# exactly two cards of a suit are present, both are discarded.
def check_exactly_two_cards_same_suit(counter, four_suits):
    for suits in counter.values():
        if len(suits) == 2:
            #  exactly two cards of a suit are present, both are discarded.
            for i in suits:
                four_suits[i] = -1
            return True

first_line = []
second_line = []
    
# INSERT YOUR CODE HERE
# TODO: recursion的形式（算法，递归的讲解）
def play_game(num, simulate = False):
    if not simulate:
        first_line.clear()
        second_line.clear()

    cards = list(range(32))
    seed(num)
    shuffle(cards)
    counter = defaultdict(list)
    four_suits = [-1, -1, -1, -1]
    round = 0
    while True:
        # 少了这个判断-
        if not cards:
            break
        round +=1
        # 某一个位置空了来补齐
        for i in range(len(four_suits)):
            if four_suits[i] == -1 and cards:
                four_suits[i] = cards.pop()
        # these are then placed at locations 1 and 2
        if not simulate:
            four_suits = [card for card in four_suits if card > -1]

            first_line.extend(four_suits[:2])
            second_line.extend(four_suits[2:])
        # 统计花色
        counter.clear()
        # TODO 如果不用defaultdict 怎么写？统计计数那个地方的算法，需要去看一下
        for i in range(len(four_suits)):
            card = four_suits[i]
            if card > -1:
                counter[card // 8].append(i)
        
        if check_four_cards_has_same_suits(counter, four_suits) \
            or check_exactly_three_cards_same_suit(counter, four_suits) \
            or check_two_cards_has_different_suits(counter, four_suits) \
            or check_exactly_two_cards_same_suit(counter, four_suits):
            continue
        else:
            # 算法结束了
            break
    # cards.extend([card for card in four_suits if card > -1])
    win = True
    if cards or [card for card in four_suits if card > -1]:
        win = False
    # 返回结果
    return len(cards), round, win
    

def simulate(n, seed_num):
    pass

def print_deck(deck):
    if deck:
        line = []
        for k in range(0, len(deck), 2):
            temp = []
            for card in deck[k: k + 2]:
                if card > -1:
                    temp.append(get_card_pic(card))
            line.append(" ".join(temp))
        line = "    " + "    ".join(line)
        print(line.rstrip())

def play():
    num = int(input("Enter an integer to pass to the seed() function: "))
    _, _, win = play_game(num)
    # 判断是否赢了
    if win:
        print("So glad that you won! 😊")
    else:
        print("So sorry that you lost! ☹")
   
    # 多行数据
    for i in range(0, len(first_line), 12):
        print()
        first = first_line[i:  i + 12]
        print_deck(first)
                
        second = second_line[i:  i + 12]
        print_deck(second)
# 不能放在这里 from solitaire import *
# play()

if __name__ == "__main__":
    play()
    # simulate(10000, 0)        
        