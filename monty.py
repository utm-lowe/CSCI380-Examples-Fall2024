#!/usr/bin/env python
import random

class Monty:
    def __init__(self, doors=3):
        self.doors = doors
        self.car = random.randrange(doors)
        self.board = ['D'] * doors

    def __str__(self):
        return ' '.join(self.board) 

    def play(self, agent):
        print(self)
        choice = agent(self.board, choice=None) 
        print("Player chooses door number {}".format(choice))

        # select a door to reveal
        reveal = list(range(self.doors))
        reveal.remove(choice-1)
        if self.car in reveal:
            reveal.remove(self.car)
        reveal = random.choice(reveal)
        print("Let's see what's behind door number {}".format(reveal+1))
        self.board[reveal] = 'G'
        print(self)

        # switch or stay choice
        choice = agent(self.board, choice)
        print("Player chooses do number {}. Let's see what they've won.".format(choice))
        self.board = ['G' for i in range(self.doors)]
        self.board[self.car] = 'C'
        print(self)

        if choice == self.car+1:
            print("Player wins!")
            return True
        else:
            print("Player loses!")
            return False

def human(board, choice = None):
    if choice is None:
        return int(input("Choose a door number: "))
    else:
        if int(input("Switch or stay? (1 to switch, 0 to stay): ")) == 1:
            return board.index('D') + 1
        else:
            return choice

def always_switch(board, choice = None):
    return random.choice([i+1 for i in range(len(board)) if board[i] == 'D'])

def never_switch(board, choice = None):
    if choice == None:
        return random.choice([i+1 for i in range(len(board)) if board[i] == 'D'])
    else:
        return choice

def main():
    agents = [human, always_switch, never_switch]
    for i in range(len(agents)):
        print("Agent {}: {}".format(i, agents[i].__name__))
    agent = agents[int(input("Choose an agent: "))]
    runs = int(input("How many runs? "))
    wins = 0
    for i in range(runs):
        monty = Monty()
        if monty.play(agent):
            wins += 1
    print("Agent {} won {} out of {} games".format(agent.__name__, wins, runs))
    print("Win rate: {:.2f}%".format(wins/runs*100))

if __name__ == '__main__':
    main()