from itertools import chain
from random import seed, shuffle
from collections import defaultdict

# INSERT YOUR CODE HERE

# Spade: from U0001F0A1 to U0001F0AE (no 1F0AC)
# Heart: from U0001F0B1 to U0001F0BE (no 1F0BC)
# Diamond: from U0001F0C1 to U0001F0CE (no 1F0CC)
# Club: from U0001F0D1 to U0001F0DE (no 1F0DC)

# Numbers 0 to 7 represent the Hearts, from Seven to Ace.
# Numbers 8 to 15 represent the Diamonds, from Seven to Ace.
# Numbers 16 to 23 represent the Clubs, from Seven to Ace.
# Numbers 24 to 31 represent the Spades, from Seven to Ace.

# Get deck after suffling
def get_shuffled_deck(seed_num):
    deck = list(range(32))
    seed(seed_num)
    shuffle(deck)

    return deck

# Get card suit and rank from deck in card unicode in base 10
def get_suit_rank(card_num):
    suit = card_num // 8
    rank = card_num % 8

    # Suit code in base 10
    heart_code = int("1F0B1", 16)
    diamond_code = int("1F0C1", 16)
    club_code = int("1F0D1", 16)
    spade_code = int("1F0A1", 16)
    suit_dic = {0:heart_code, 1:diamond_code, 2:club_code, 3:spade_code}

    # Rank number and unicode base 10 number pair
    # For example: number 0 is heart of 7 but it is the 7th in unicode of heart
    rank_dic = {0:6, 1:7, 2:8, 3:9, 4:10, 5:12, 6:13, 7:0}

    card_unicode = suit_dic[suit] + rank_dic[rank]

    return chr(card_unicode)

# Check if discard
def check_discard(revealed_cards):
    counter = defaultdict(list)
    # Get suit for each among 4 cards
    for i in range(len(revealed_cards)):
            card = revealed_cards[i]
            if card is not None:
                suit = card // 8
                counter[suit].append(i)

    values = counter.values()

    # If all 4 cards share the same suit
    if len(counter) == 1:
        index_list = list(values)[0]
        if len(index_list) == 4:
            for i in index_list:
                # Discard all 4 cards
                revealed_cards[i] = None
            return True
        
    #  If there are two cards each of two different suits
    if len(counter) == 2:
        first_suit, second_suit = values
        if len(first_suit) == len(second_suit) == 2:
            for i in first_suit + second_suit:
                # Discard all 4 cards
                revealed_cards[i] = None
            return True
    
    # If there are more than 2 suits
    for suit in values:
        # If exactly two cards of a suit are present
        if len(suit) == 2:
            for i in suit:
                # Discard these 2 cards only
                revealed_cards[i] = None
            return True
        
        # If exactly 3 cards of a suit are present
        if len(suit) == 3:
            for i in suit[:2]:
                # Discard 2 cards in lower positions
                revealed_cards[i] = None
            return True

# Check if win
def check_win(deck, revealed_cards):
    is_exist = False # If there is no revealed cards 

    for card in revealed_cards:
        if card is not None:
            is_exist = True

    # If deck is not empty or there exists some cards revealed, then lose, else win
    if deck or is_exist:
        return False
    else:
        return True

first_square_line = []
second_square_line = []

# Play one game
def play_one_game(seed_num):
    deck = get_shuffled_deck(seed_num)
    revealed_cards = [None, None, None, None] # Show 4 cards revealed, none means no card in this position

    while True:
        if not deck:
            break

        for i in range(4):
            # Fill from top of the deck which is the last number in deck
            # if this position is empty and if deck is not empty
            if revealed_cards[i] == None and deck:
                revealed_cards[i] = deck.pop()

        # Add card to final result of squares
        first_square_line.extend(revealed_cards[:2])
        second_square_line.extend(revealed_cards[2:])

        # Check discard
        if check_discard(revealed_cards):
            continue
        else:
            break
        
    # Check if win
    is_win = check_win(deck, revealed_cards)

    return is_win

# Play simulate game
def play_simulate(seed_num):
    deck = get_shuffled_deck(seed_num)
    revealed_cards = [None, None, None, None] # Show 4 cards revealed, none means no card in this position
    round = 0

    while True:
        if not deck:
            break

        round = round + 1

        for i in range(4):
            # Fill from top of the deck which is the last number in deck
            # if this position is empty and if deck is not empty
            if revealed_cards[i] == None and deck:
                revealed_cards[i] = deck.pop()

        # Add card to final result of squares
        first_square_line.extend(revealed_cards[:2])
        second_square_line.extend(revealed_cards[2:])

        # Check discard
        if check_discard(revealed_cards):
            continue
        else:
            break

    return round, len(deck)

# Print squares
def print_squares(deck):
    if not deck:
        return
    
    line = []

    # A square contains 2 cards in one line
    for i in range(0, len(deck), 2):
        card_number = []
        for card in deck[i: i + 2]:
            if card is not None:
                card_number.append(get_suit_rank(card))
        line.append(" ".join(card_number))

    line = "    " + "    ".join(line)
    print(line.rstrip())

# Print simulate result
def simulate(num_of_games, seed_num):
    counter = defaultdict(dict)

    for i in range(num_of_games):
        round, cards_left = play_simulate(seed_num + i)

        # cards_left is the primary key and round is the sub key in a dict
        if round not in counter[cards_left]:
            counter[cards_left][round] = 0 # Add sub key of play rounds and initial count 0
        
        counter[cards_left][round] = counter[cards_left][round] + 1
    
    if counter:
        cards_left_title = "Number of cards left"
        round_title = "Number of rounds"
        freq_title = "Frequency"

        # Sort by cards left and number of rounds, and extract dict into list of tuples
        result_list = list(chain.from_iterable(
            ((num_of_left, num_of_rounds, count) for num_of_rounds, count in sorted(round_counter.items()))
            for num_of_left, round_counter in sorted(counter.items())
        ))
        # print(result_list)

        total_rounds = 0 
        total_rounds_by_cards_left = {} # Store rounds number for each cards left group
        count_by_cards_left = {} # Store how many records for each cards left group
        rounds_by_card_left = {} # Store rounds played situation for each cards left group

        for n, r, c in result_list:
            total_rounds = total_rounds + c

            if n not in total_rounds_by_cards_left:
                total_rounds_by_cards_left[n] = 0
            total_rounds_by_cards_left[n] = total_rounds_by_cards_left[n] + c

            if n in count_by_cards_left:
                count_by_cards_left[n] = count_by_cards_left[n] + 1
            else:
                count_by_cards_left[n] = 1

            if n not in rounds_by_card_left:
                rounds_by_card_left[n] = []
            else:
                rounds_by_card_left[n].append(r)

        all_single = False
        
        # Print result
        print(f"{cards_left_title} | {round_title} | {freq_title}")
        print("---------------------------------------------------")

        # Need to check if there is only 1 record in each cards left group
        # It determines where to print frequency
        # Number of cards left and Frequency only show for the first in each group
        is_exist = []
        for n, r, c in result_list: 
            # Left padding for number of rounds column
            padding = (2 - len(str(r))) * " "

            if count_by_cards_left[n] == 1:
                is_exist.append(n)
                frequency = f"{c / total_rounds * 100:.2f}"
                number_round_content = f"{padding}{str(r)}"
                print(f"{str(n).rjust(len(cards_left_title))} | {number_round_content.ljust(len(round_title))} | {frequency.rjust(len(freq_title) - 1)}%")
            elif n not in is_exist:
                is_exist.append(n)
                frequency = f"{total_rounds_by_cards_left[n] / total_rounds * 100:.2f}"
                group_freq = c / total_rounds * 100
                number_round_content = f"{padding}{str(r)} ({group_freq:.2f}%)"
                print(f"{str(n).rjust(len(cards_left_title))} | {str(number_round_content).ljust(len(round_title))} | {frequency.rjust(len(freq_title) - 1)}%")
            else:
                group_freq = c / total_rounds * 100
                number_round_content = f"{padding}{str(r)} ({group_freq:.2f}%)"
                print(f"{''.rjust(len(cards_left_title))} | {str(number_round_content).ljust(len(round_title))} |")

# Run the game
def run():
    seed_num = int(input("Enter an integer to pass to the seed() function: "))
    is_win = play_one_game(seed_num)
    
    if is_win:
        print("So glad that you won! 😊")
    else:
        print("So sorry that you lost! ☹️")
   
    # Each line at most contains 6 squares, then 12 cards
    for i in range(0, len(first_square_line), 12):
        print()
        first = first_square_line[i : i + 12]
        print_squares(first)
                
        second = second_square_line[i : i + 12]
        print_squares(second)


if __name__ == "__main__":
    run()
    # simulate(76589, 4569)
    # simulate(300, 7)
    # simulate(2, 1)
    # simulate(100000, 111111)