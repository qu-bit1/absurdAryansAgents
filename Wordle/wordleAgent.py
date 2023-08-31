import random
import numpy as np
import pandas as pd

class Agent:
    def __init__(self, game, f_name='words.csv'):
        self.vowels = ['A','E','I','O','U','Y']
        dict = pd.read_csv(f_name)
        dict = dict[dict['words'].str.len()==game.letters]
        dict['words'] = dict['words'].str.upper()
        dict['v-count'] = dict['words'].apply(lambda x: ''.join(set(x))).str.count('|'.join(self.vowels)) #Count amount of vowels in words
        self.dict = dict
        self.game = game
        self.prediction = ['' for _ in range(game.letters)]
        self.y_letters = {}
        self.g_letters = []

    def calc_letter_probs(self):
        for x in range(self.game.letters):
            counts = self.dict['words'].str[x].value_counts(normalize=True).to_dict()
            self.dict[f'p-{x}'] = self.dict['words'].str[x].map(counts)

    def parse_board(self):
        if self.game.g_count > 0:
            g_hold = []
            for x, c in enumerate(self.game.colours[self.game.g_count - 1]):
                letter = self.game.board[self.game.g_count - 1][x]
                if c == 'Y':
                    if letter not in self.y_letters:
                        self.y_letters[letter] = [x]
                    else:
                        if x not in self.y_letters[letter]:
                            self.y_letters[letter].append(x)
                elif c == 'G':
                    self.prediction[x] = letter
                else:
                    if letter in self.prediction:
                        if letter not in self.y_letters:
                            self.y_letters[letter] = [x]
                        else:
                            self.y_letters[letter].append(x)
                    elif letter not in self.g_letters:
                        self.g_letters.append(letter)
            self.g_letters = [l for l in self.g_letters if l not in self.y_letters and l not in self.prediction]

    def choose_action(self):
        self.parse_board()
        if len(self.g_letters) > 0:
            self.dict = self.dict[~self.dict['words'].str.contains('|'.join(self.g_letters))]
            self.g_letters = []
        if len(self.y_letters) > 0:
            y_str = '^' + ''.join(fr'(?=.*{l})' for l in self.y_letters)
            self.dict = self.dict[self.dict['words'].str.contains(y_str)]
            for s, p in self.y_letters.items():
                for i in p:
                    self.dict = self.dict[self.dict['words'].str[i]!=s]
            self.y_letters = {}
        for i, s in enumerate(self.prediction):
            if s != '':
                self.dict = self.dict[self.dict['words'].str[i]==s]
        self.dict['w-score'] = [0] * len(self.dict)
        if len(self.dict) > 5:
            self.calc_letter_probs()
        for x in range(self.game.letters):
            if self.prediction[x] == '':
                self.dict['w-score'] += self.dict[f'p-{x}']
        if True not in [True for s in self.prediction if s in self.vowels]:
            self.dict['w-score'] += self.dict['v-count'] / self.game.letters
        mv_bank = self.dict[self.dict['w-score']==self.dict['w-score'].max()]
        result = random.choice(mv_bank['words'].tolist())
        return result


from copy import deepcopy
class Wordle:
    def __init__(self, word, rows=6, letters=5):
        self.g_count = 0
        self.word = word
        self.w_hash_table = {}
        if word is not None:
            for x, l in enumerate(word):
                if l in self.w_hash_table:
                    self.w_hash_table[l]['count'] += 1
                    self.w_hash_table[l]['pos'].append(x)
                else:
                    self.w_hash_table[l] = {'count':1, 'pos':[x]}
        self.rows = rows
        self.letters = letters
        self.board = [['' for _ in range(letters)] for _ in range(rows)]
        self.colours = [['' for _ in range(letters)] for _ in range(rows)]
        self.alph = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    def is_end(self):
        if self.board[-1] != ['' for _ in range(self.letters)]:
            return True
        else:
            r = self.game_result()
            if r[0] == True:
                return True
            else:
                return False

    def game_result(self):
        win = (False, 99)
        for i, r in enumerate(self.board):
            if self.word == ''.join(r):
                win = (True, i)
                break
        return win

    def update_board(self, u_inp):
        w_hash_table = deepcopy(self.w_hash_table)
        i_hash_table = {}
        for x, l in enumerate(str(u_inp).upper()):
            self.board[self.g_count][x] = l
            if l in i_hash_table:
                i_hash_table[l].append(x)
            else:
                i_hash_table[l] = [x]
        colours = {'G':[],'B':[],'Y':[]}
        for l in i_hash_table:
            if l in w_hash_table:
                g_hold = []
                for p in i_hash_table[l]:
                    if p in w_hash_table[l]['pos']:
                        g_hold.append(p)
                for p in g_hold:
                    i_hash_table[l].remove(p)
                colours['G'] += g_hold
                if len(g_hold) < w_hash_table[l]['count']:
                    y_hold = []
                    for p in i_hash_table[l]:
                        y_hold.append(p)
                        if len(y_hold) == w_hash_table[l]['count']:
                            break
                    for p in y_hold:
                        i_hash_table[l].remove(p)
                    colours['Y'] += y_hold
                for p in i_hash_table[l]:
                    colours['B'].append(p)
            else:
                colours['B'] += i_hash_table[l]
                i_hash_table[l] = []
        for c in colours:
            for p in colours[c]:
                self.colours[self.g_count][p] = c
        self.g_count += 1

    def valid_guess(self, u_inp):
        if len(u_inp) == 5 and False not in [False for s in str(u_inp).upper() if s not in self.alph]:
            return True
        else:
            return False

from tqdm import tqdm


ROWS = 6
LETTERS = 5
GAMES = int(input("Enter the number of games you want to play: "))

dict = pd.read_csv('words.csv')
dict = dict[dict['words'].str.len() == LETTERS]
dict['words'] = dict['words'].str.upper()

control = input('What would you like to do?\n\n-Agnet[A]\n-Play Game [P]\n\n')

if 'A' in str(control).upper() or 'P' in str(control).upper():
    if 'P' in str(control).upper():
        print('PLAY GAME SELECTED\n')
    else:
        print('AGENT SELECTED\n')

    results = []
    if 'P' in str(control).upper():
        silent = True
        GAMES = 1
    else:
        silent = False

    for _ in tqdm(range(GAMES), desc='GAMES', disable=silent):
        word = random.choice(dict['words'].tolist())
        game = Wordle(
            word,
            rows=ROWS,
            letters=LETTERS
        )
        bot = Agent(game)

        while game.is_end() == False:
            u_inp = bot.choose_action()
            print(f"Agent's guess: {u_inp}")
            if game.valid_guess(u_inp):
                game.update_board(u_inp)
            else:
                print('ERROR - WORDS MUST BE 5 LETTERS')

        r = game.game_result()

        if r[0] == True:
            if r[1] > 0:
                print(f'\nCONGRATS YOU WON IN {r[1] + 1} GUESSES!\n')
            else:
                print(f'\nCONGRATS YOU WON IN {r[1] + 1} GUESS!\n')
        else:
            print(f'\nSORRY YOU DID NOT WIN.\n')

        print(np.array(game.board), '\n')
        results.append({'word': word, 'result': r[0], 'moves': r[1] + 1})

    results = pd.DataFrame(results)
    print(results)
    print(f'Win Percent = {(len(results[results["result"]==True]) / len(results)) * 100}%\nAverage Moves = {results[results["result"]==True]["moves"].mean()}')