from itertools import chain
from random import seed, shuffle
from collections import defaultdict

# INSERT YOUR CODE HERE

# Numbers 0 to 12 represent the Hearts, from Ace to King.
# Numbers 13 to 25 represent the Diamonds, from Ace to King.
# Numbers 26 to 38 represent the Clubs, from Ace to King.
# Numbers 39 to 51 represent the Spades, from Ace to King.

output_lines = []

# Get deck after suffling
def get_shuffled_deck(seed_num):
    deck = list(range(52))
    seed(seed_num)
    shuffle(deck)

    return deck

# Get card suit and rank from deck in card unicode in base 10
def get_suit_rank(card_num):
    suit = card_num // 13
    rank = card_num % 13

    # Suit code in base 10
    heart_code = int("1F0B1", 16)
    diamond_code = int("1F0C1", 16)
    club_code = int("1F0D1", 16)
    spade_code = int("1F0A1", 16)
    suit_dic = {0:heart_code, 1:diamond_code, 2:club_code, 3:spade_code}

    # Rank number and unicode base 10 number pair
    # Note: rank Q number 11 is the 12th card in a suit
    if rank > 10:
        card_unicode = suit_dic[suit] + rank + 1
    else:
        card_unicode = suit_dic[suit] + rank

    return chr(card_unicode)

# Place the card to col, row
def place_card(card_drawn, cards_placed, card_to):
    # To where to place this card
    # 0:columns
    # 1:rows
    is_placed = False 
    is_new = False

    for list in cards_placed:
        # If there has cards in the list
        if list:
            last_card = list[-1]
            # If suits are different                     If strictly 2 ranks higher 
            if (card_drawn // 26 != last_card //26) and (card_drawn % 13 == last_card % 13 + 2):
                list.append(card_drawn)
                # Extend a list
                is_placed = True
                return is_placed, is_new
        else:
            # If no card in the list, check if can start a new list
            # Only A(number 0) and 2(number 1) in each suit can start a new list
            if card_drawn % 13 == card_to:
                list.append(card_drawn)
                # Start a new list
                is_placed = True
                is_new = True
                return is_placed, is_new
    
    # Cannot extend a list or start a new list
    return is_placed, is_new


# Check where to put the card, columns? rows? or discard pile? And generate outputs
def check_card(cards_to_draw, col_cards, row_cards, card_from):
    # From where to draw this card
    # 0:stack
    # 1:discard pile

    card_drawn = cards_to_draw[-1] # Draw from the top

    # Check if can place to columns
    col_placed, col_new = place_card(card_drawn, col_cards, 0)
    if col_placed:
        if col_new:
            output_lines.append("Starting a column 😊️")
        elif card_from == 0:
            output_lines.append("Extending a column by using the card drawn from the stack 😊️")
        elif card_from == 1:
            output_lines.append("Extending a column by using the top card of the discard pile 😊️")
        return True # The card is placed
    
    # Check if can place to rows
    row_placed, row_new = place_card(card_drawn, row_cards, 1)
    if row_placed:
        if row_new:
            output_lines.append("Starting a row 😊️")
        elif card_from == 0:
            output_lines.append("Extending a row by using the card drawn from the stack 😊️")
        elif card_from == 1:
            output_lines.append("Extending a row by using the top card of the discard pile 😊️")
        return True # The card is placed 

# Print undrawn cards only
def print_undrawn_cards(stack, discard_pile):
    # Print stack cards
    output_lines.append("]" * len(stack))

    # Print discard cards, and top card face up
    if discard_pile:
        top_card_character = get_suit_rank(discard_pile[-1])
        output_lines.append("[" * (len(discard_pile) - 1) + top_card_character)
    else:
        output_lines.append("")

# Print all cards
def print_all_cards(stack, discard_pile, col_cards, row_cards):
    # Print undrawn cards
    print_undrawn_cards(stack, discard_pile)

    # Print placed cards
    row_range = 7 # The whole pattern contains 7 rows

    for r in range(row_range):
        col_cards_in_line = []
        # Print cols, from the last card to the first card in the list
        for c in range(len(col_cards)):
            col_list = col_cards[c]
            
            if len(col_list) > row_range - r - 1:
                character = get_suit_rank(col_list[row_range - r - 1])
                col_cards_in_line.append(character)
            else:
                col_cards_in_line.append("")

        # Print rows, if exist, rows only show on the first 4 lines
        if r < len(row_cards):
            row_cards_in_line = []
            for card in row_cards[r]:
                character = get_suit_rank(card)
                row_cards_in_line.append(character)
            row_cards_str = " ".join(row_cards_in_line)

            line = "\t".join(col_cards_in_line) + "\t" * (6 - len(col_cards_in_line)) + row_cards_str
        else:
            line = "\t".join(col_cards_in_line)

        output_lines.append(line.rstrip())

    output_lines.append("")
        
# Play one game
def play_one_game(seed_num):
    # Set initial values
    col_cards = [[], [], [], []]
    row_cards = [[], [], [], []]
    discard_pile = []
    round_dict = {1:"first", 2:"second", 3:"third"}
    stack = get_shuffled_deck(seed_num) # Get shuffled cards

    for i in round_dict:
        if not stack and not discard_pile:
            break

        # For a new run, all cards from stack are drawn, the discard pile is turned over to become the new stack
        stack.extend(discard_pile[::-1])
        discard_pile.clear()

        output_lines.append(f"Starting the {round_dict[i]} round...")
        output_lines.append("")

        while stack:
            # If the top card from stack is placed
            is_placed_stack = check_card(stack, col_cards, row_cards, 0)
            if is_placed_stack:
                stack.pop()
                print_all_cards(stack, discard_pile, col_cards, row_cards)
                # Try to place the top card from discard pile
                while discard_pile:
                    is_placed_discard = check_card(discard_pile, col_cards, row_cards, 1)
                    if is_placed_discard:
                        discard_pile.pop()
                        # Print all cards from stack, discard pile, cols and rows
                        print_all_cards(stack, discard_pile, col_cards, row_cards)
                    else:
                        output_lines.append("Cannot use the top card of the discard pile ☹️")
                        output_lines.append("")
                        break
            else:
                # If the top card from stack cannot be placed, place it to discard pile
                top_card = stack.pop()
                discard_pile.append(top_card)
                output_lines.append("Cannot use the card drawn from the stack ☹️")
                print_undrawn_cards(stack, discard_pile)
                output_lines.append("")
        
        output_lines.append("")

    return len(stack) + len(discard_pile)

# Run simulate game
def simulate(num_of_games, seed_num):
    counter = defaultdict(dict)

    for i in range(num_of_games):
        cards_left = play_one_game(seed_num + i)

        if cards_left not in counter:
            counter[cards_left] = 0
        
        counter[cards_left] = counter[cards_left] + 1

    if counter:
        cards_left_title = "Number of cards left"
        freq_title = "Frequency"
        total_round = 0

        for n in counter:
            total_round = total_round + counter[n]

        # Print result
        print(f"{cards_left_title} | {freq_title}")
        print("--------------------------------")

        for num_cards_left, count in sorted(counter.items()):
            frequncy = f"{count / total_round * 100:.2f}"
            print(f"{str(num_cards_left).rjust(len(cards_left_title))} | {frequncy.rjust(len(freq_title) - 1)}%")

# User interaction
def accept_input(output):
    num_lines = len(output)

    while True:
        print("Enter: q to quit")
        # Following line starting from position of q
        print(f"       a last line number (between 1 and {num_lines})")
        print(f"       a first line number (between -1 and -{num_lines})")
        print(f"       a range of line numbers (of the form m--n with 1 <= m <= n <= {num_lines})")
        user_input = input("       ").strip()

        # Exit
        if user_input == "q":
            break
        
        is_valid = False # Check if input is valid

        # If it is a range
        if "--" in user_input:
            input_range = user_input.split("--")
            fr = int(input_range[0])
            to = int(input_range[1])

            if 1 <= fr <= to <= num_lines:
                for i in range(fr -1, to):
                    print(output[i])
                print()
                is_valid = True

        # If it is one positive integer
        elif user_input.isdigit():
            integer = int(user_input)
            if 1 <= integer <= num_lines:
                for i in range(integer):
                    print(output[i])
                print()
                is_valid = True

        # If it is one nagetive integer
        elif user_input[1:].isdigit():
            integer = int(user_input)
            if -1 >= integer >= -num_lines:
                for i in range(num_lines + integer, num_lines):
                    print(output[i])
                print()
                is_valid = True     

        if not is_valid:
            print()
            continue

# Run the game
def run():
    seed_num = int(input("Enter an integer to pass to the seed() function: "))
    cards_left = play_one_game(seed_num)

    if cards_left == 0:
        print("You placed all cards, you won 👍")
    elif cards_left == 1:
        print("You could not place 1 card, you lost 👎")
    else:
        print(f"You could not place {cards_left} cards, you lost 👎")
    print()

    # Ask user input
    while not output_lines[-1]:
        output_lines.pop()
    print(f"There are {len(output_lines)} lines of output; what do you want me to do?")
    print()
    
    accept_input(output_lines)

if __name__ == "__main__":
    run()