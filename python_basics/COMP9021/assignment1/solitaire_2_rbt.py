from itertools import chain
from random import seed, shuffle
from collections import defaultdict

lines = []


def get_pic_card(card):
    card_decks = [127153, 127169, 127185, 127137]
    offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
    i, j = card // 13, card % 13
    return chr(card_decks[i] + offsets[j])

def print_all_stacks(cards,  discard_cards, column_cards, row_cards):
    lines.append("]" * len(cards))
    if discard_cards:
        lines.append("[" * (len(discard_cards) - 1) + get_pic_card(discard_cards[-1]))
    else:
        lines.append("")
    # 构建一个grid
    grid = [['' for _ in range(6)] for _ in range(7) ]
    # 绑定行的的数据
    for i in range(len(row_cards)):
        cards = row_cards[i]
        for card in cards:
            grid[i][-1] += get_pic_card(card)
        grid[i][-1] = " ".join(grid[i][-1])
    # 绑定行的数据
    for i in range(len(column_cards)):
        cards = column_cards[i]
        for j in range(len(cards)):
            card = cards[j]
            grid[-1 - j][i] = get_pic_card(card)
    # 开始输出图形
    for row in grid:
        lines.append("\t".join(row).rstrip())

    lines.append("")

# 检查是否可以往牌桌上放
def check_card_to_tables(card, table_cards, start = 0):
    # 返回值有两个
    # 1. 表示可以放置
    # 2. 表示是否是开头
    for cards in table_cards:
        if cards:
            last = cards[-1]
            # 判断是否是异色
            if last // 26 != card // 26:
                # 表示异色, 判断是否是连续的
                if card % 13 == last % 13 + 2:
                    cards.append(card)
                    return True, False
        elif card % 13 == start:
            cards.append(card)
            return True, True
    # 不可以放置
    return False, False


def check_card_can_be_drawn(last, column_cards, row_cards, from_stack = True):
    # 检查牌是否可以放到列上
    res, start = check_card_to_tables(last, column_cards, 0)
    if res:
        if start:
            lines.append("Starting a column 😊️")
        elif from_stack:
            lines.append("Extending a column by using the card drawn from the stack 😊️")
        else:
            lines.append("Extending a column by using the top card of the discard pile 😊️")
        # 返回了
        return res
    # 检查牌是否可以放到行上
    res, start = check_card_to_tables(last, row_cards, 1)
    if res:
        if start:
            lines.append("Starting a row 😊️")
        elif from_stack:
            lines.append("Extending a row by using the card drawn from the stack 😊️")
        else:
            lines.append("Extending a row by using the top card of the discard pile 😊️")
        # 返回了
        return res

# INSERT YOUR CODE HERE
def play_rounds(num, simulate = False):
    column_cards = [[], [], [], []]
    row_cards = [[], [], [], []]
    discard_cards = []
    rounds = ["first", "second", "third"]
    cards = list(range(52))
    seed(num)
    # 写好
    shuffle(cards)

    for round in rounds:
        # 没有牌剩余
        if not cards and not discard_cards:
            break
        cards.extend(discard_cards[::-1])
        discard_cards.clear()
        # 开始玩游戏
        lines.append(f"Starting the {round} round...")
        lines.append("")
        while cards:
            # 检查最后一张牌
            if check_card_can_be_drawn(cards[-1], column_cards, row_cards, True):
                cards.pop()
                print_all_stacks(cards, discard_cards, column_cards, row_cards)
                # 检查discard
                while discard_cards:
                    if check_card_can_be_drawn(discard_cards[-1], column_cards, row_cards, False):
                        discard_cards.pop()
                        print_all_stacks(cards, discard_cards, column_cards, row_cards)
                    else:
                        lines.append("Cannot use the top card of the discard pile ☹️")
                        lines.append("")
                        break
            else:
                discard_cards.append(cards.pop())
                lines.append("Cannot use the card drawn from the stack ☹️")
                lines.append("]" * len(cards))
                lines.append("[" * (len(discard_cards) - 1) + get_pic_card(discard_cards[-1]))
                lines.append("")

        lines.append("")
    return len(cards + discard_cards)


def play():
    num = int(input("Enter an integer to pass to the seed() function: "))
    print()
    nb_cards = play_rounds(num)
    if nb_cards == 0:
        print("You placed all cards, you won 👍")
    elif nb_cards == 1:
        print("You could not place 1 card, you lost 👎")
    else:
        print(f"You could not place {nb_cards} cards, you lost 👎")
    print()
    while not lines[-1]:
        lines.pop()
    print(f"There are {len(lines)} lines of output; what do you want me to do?")
    print()
    for line in lines:
        print(line)

def simulate(n, seed_num):
    pass

if __name__ == "__main__":
    # simulate(30, 2)
    play()


