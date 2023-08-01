
DEBUG_FLAG = False

def human_selector(choice_list, initial_phrase, selection_phrase, allow_zero):
    if len(choice_list) == 0:
        print("Empty possibility list")
        if allow_zero:
            return 0
        else:
            return None

    while True:
        print(initial_phrase)

        for i, territory in enumerate(choice_list):
            print(f"{i+1}. {territory}")

        try:
            choice = int(input(selection_phrase))

            if choice == 0:
                if allow_zero:
                    return 0
                else:
                    raise ValueError

            if choice < 0 or choice > len(choice_list):
                raise ValueError

            selected_territory = choice_list[choice - 1]

            return selected_territory

        except (ValueError, IndexError):
            print("Invalid input. Please try again.")

def debug_print(str, debug=False):
    if DEBUG_FLAG or debug:
        print(str)