

def territory_selector(territory_list, initial_phrase, selection_phrase, allow_zero):
    while True:
        print(initial_phrase)

        for i, territory in enumerate(territory_list):
            print(f"{i+1}. {territory}")

        try:
            choice = int(input(selection_phrase))

            if choice == 0 and allow_zero:
                return 0

            if choice < 0 or choice > len(territory_list):
                raise ValueError

            selected_territory = territory_list[choice - 1]

            return selected_territory

        except (ValueError, IndexError):
            print("Invalid input. Please try again.")