from collections import defaultdict
import random
import time
import numpy as np
import re

class HangmanAPI(object):	
	def __init__(self):
		self.guessed_letters = []
		
		full_dictionary_location = "words_250000_train.txt"
		self.full_dictionary = self.build_dictionary(full_dictionary_location)        
		self.letter_set = sorted(set().union(*self.full_dictionary))
		self.incorrect_guesses = []
		self.probability = [0] * len(self.letter_set)
		current_dictionary = self.full_dictionary
		self._1_gram =   self.build_1_gram(current_dictionary)
		self._2_gram =   self.build_2_gram(current_dictionary)
		self._3_gram =   self.build_3_gram(current_dictionary)
		self._4_gram =   self.build_4_gram(current_dictionary)
		self._5_gram =   self.build_5_gram(current_dictionary)
		self._6_gram =   self.build_6_gram(current_dictionary)

	def guess(self, word):
		self.incorrect_guesses = list(set(self.guessed_letters) - set(word))
	   
		if len(self.guessed_letters) >0 and self.guessed_letters[-1] in self.incorrect_guesses :
			current_dictionary = [word for word in self.full_dictionary if not set(word).intersection(self.incorrect_guesses)]
			
			self._1_gram =   self.build_1_gram(current_dictionary)
			self._2_gram =   self.build_2_gram(current_dictionary)
			self._3_gram =   self.build_3_gram(current_dictionary)
			self._4_gram =   self.build_4_gram(current_dictionary)
			self._5_gram =   self.build_5_gram(current_dictionary)
			self._6_gram =   self.build_6_gram(current_dictionary)

		self.probability = [0] * len(self.letter_set)
		clean_word = word[::2].replace("_",".")
		
		# print(clean_word)

		return self.find_5_gram_probability(clean_word)

	def build_6_gram(self, dictionary):
		_6_gram = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))))
		
		for word in dictionary:
			
			for i in range(len(word) - 5):
				_6_gram[word[i]][word[i+1]][word[i+2]][word[i+3]][word[i+4]][word[i+5]] += 1

		return _6_gram
	
	def build_5_gram(self, dictionary):
		_5_gram = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))
		
		for word in dictionary:
			
			for i in range(len(word) - 4):
				_5_gram[word[i]][word[i+1]][word[i+2]][word[i+3]][word[i+4]] += 1

		return _5_gram
	
	def build_4_gram(self, dictionary):
		_4_gram = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
		
		for word in dictionary:
			
			for i in range(len(word) - 3):
				_4_gram[word[i]][word[i+1]][word[i+2]][word[i+3]] += 1

		return _4_gram
	
	def build_3_gram(self, dictionary):
		_3_gram = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
		
		for word in dictionary:
			
			for i in range(len(word) - 2):
				_3_gram[word[i]][word[i+1]][word[i+2]] += 1

		return _3_gram
	
	def build_2_gram(self, dictionary):
		_2_gram = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
		
		for word in dictionary:
			
			for i in range(len(word) - 1):
				_2_gram[len(word)][word[i]][word[i+1]] += 1

		return _2_gram
	
	def build_1_gram(self, dictionary):
		_1_gram = defaultdict(lambda: defaultdict(int))
		
		for word in dictionary:
			
			for i in (set(word)):
				_1_gram[len(word)][i] += 1

		return _1_gram

	def find_6_gram_probability(self, word):
		
		letter_count = [0] * len(self.letter_set)
		probs = [0] * len(self.letter_set)
		total_count = 0
		j=0
		for i in range(len(word) - 5):
			# case 1: ".XXXXX"
			if word[i] == '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] != '.' and word[i+5] != '.':
				known_1= word[i+1]
				known_2= word[i+2]
				known_3= word[i+3]
				known_4= word[i+4]
				known_5= word[i+5]
				
				for (j, letter) in enumerate(self.letter_set):
					if self._6_gram[letter][known_1][known_2][known_3][known_4][known_5] != 0 and letter not in self.guessed_letters:
						total_count += self._6_gram[letter][known_1][known_2][known_3][known_4][known_5]
						letter_count[j] += self._6_gram[letter][known_1][known_2][known_3][known_4][known_5]
			
		  # case 2: "X.XXXX"
			if word[i] != '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] != '.' and word[i+5] != '.':
				known_1= word[i]
				known_2= word[i+2]
				known_3= word[i+3]
				known_4= word[i+4]
				known_5= word[i+5]
				
				for (j, letter) in enumerate(self.letter_set):
					if self._6_gram[known_1][letter][known_2][known_3][known_4][known_5] != 0 and letter not in self.guessed_letters:
						total_count += self._6_gram[known_1][letter][known_2][known_3][known_4][known_5]
						letter_count[j] += self._6_gram[known_1][letter][known_2][known_3][known_4][known_5]
						
			# case 3: "XX.XXX"
			if word[i] != '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] != '.' and word[i+4] != '.' and word[i+5] != '.':
				known_1= word[i]
				known_2= word[i+1]
				known_3= word[i+3]
				known_4= word[i+4]
				known_5= word[i+5]
				
				for (j, letter) in enumerate(self.letter_set):
					if self._6_gram[known_1][known_2][letter][known_3][known_4][known_5] != 0 and letter not in self.guessed_letters:
						total_count += self._6_gram[known_1][known_2][letter][known_3][known_4][known_5]
						letter_count[j] += self._6_gram[known_1][known_2][letter][known_3][known_4][known_5]
						
			# case 4: "XXX.XX"
			if word[i] != '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] == '.' and word[i+4] != '.' and word[i+5] != '.':
				known_1= word[i]
				known_2= word[i+1]
				known_3= word[i+2]
				known_4= word[i+4]
				known_5= word[i+5]
				
				for (j, letter) in enumerate(self.letter_set):
					if self._6_gram[known_1][known_2][known_3][letter][known_4][known_5] != 0 and letter not in self.guessed_letters:
						print(total_count, self._6_gram[letter][known_1][known_2][known_3][letter], known_4, known_5)
						total_count += self._6_gram[letter][known_1][known_2][known_3][letter][known_4][known_5]
						letter_count[j] += self._6_gram[letter][known_1][known_2][known_3][letter][known_4][known_5]
						
			# case 5: "XXXX.X"
			if word[i] != '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] == '.' and word[i+5] != '.':
				known_1= word[i]
				known_2= word[i+1]
				known_3= word[i+2]
				known_4= word[i+3]
				known_5= word[i+5]
				
				for (j, letter) in enumerate(self.letter_set):
					if self._6_gram[known_1][known_2][known_3][known_4][letter][known_5] != 0 and letter not in self.guessed_letters:
						total_count += self._6_gram[known_1][known_2][known_3][known_4][letter][known_5]
						letter_count[j] += self._6_gram[known_1][known_2][known_3][known_4][letter][known_5]
						
			# case 6: "XXXXX."
			if word[i] != '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] != '.' and word[i+5] == '.':
				known_1= word[i]
				known_2= word[i+1]
				known_3= word[i+2]
				known_4= word[i+3]
				known_5= word[i+4]
				
				for (j, letter) in enumerate(self.letter_set):
					if self._6_gram[known_1][known_2][known_3][known_4][known_5][letter] != 0 and letter not in self.guessed_letters:
						total_count += self._6_gram[known_1][known_2][known_3][known_4][known_5][letter]
						letter_count[j] += self._6_gram[known_1][known_2][known_3][known_4][known_5][letter]
		
		if total_count != 0:
			for i in range(len(self.letter_set)):
				probs[i] = letter_count[i] / total_count
		
		i=0
		while i < len(self.probability):
			self.probability[i] = self.probability[i] + probs[i] * (0.55)
			i+=1
		
		return self.find_5_gram_probability(word)
	  
	def find_5_gram_probability(self, word):
		
		letter_count = [0] * len(self.letter_set)
		probs = [0] * len(self.letter_set)
		total_count = 0
		j=0
		for i in range(len(word) - 4):
						
			# case 3: ".XXXX "
			if word[i] == '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] != '.':
				known_1= word[i+1]
				known_2= word[i+2]
				known_3= word[i+3]
				known_4= word[i+4]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._5_gram[letter][known_1][known_2][known_3][known_4] != 0 and letter not in self.guessed_letters:
						total_count += self._5_gram[letter][known_1][known_2][known_3][known_4]
						letter_count[j] += self._5_gram[letter][known_1][known_2][known_3][known_4]
			
		   # case 3: "X.XXX "
			if word[i] != '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] != '.':
				known_1= word[i]
				known_2= word[i+2]
				known_3= word[i+3]
				known_4= word[i+4]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._5_gram[known_1][letter][known_2][known_3][known_4] != 0 and letter not in self.guessed_letters:
						total_count += self._5_gram[known_1][letter][known_2][known_3][known_4]
						letter_count[j] += self._5_gram[known_1][letter][known_2][known_3][known_4]
						
			# case 3: "XX.XX "
			if word[i] != '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] != '.' and word[i+4] != '.':
				known_1= word[i]
				known_2= word[i+1]
				known_3= word[i+3]
				known_4= word[i+4]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._5_gram[known_1][known_2][letter][known_3][known_4] != 0 and letter not in self.guessed_letters:
						total_count += self._5_gram[known_1][known_2][letter][known_3][known_4]
						letter_count[j] += self._5_gram[known_1][known_2][letter][known_3][known_4]
						
			# case 3: "XXX.X "
			if word[i] != '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] == '.' and word[i+4] != '.':
				known_1= word[i]
				known_2= word[i+1]
				known_3= word[i+2]
				known_4= word[i+4]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._5_gram[known_1][known_2][known_3][letter][known_4] != 0 and letter not in self.guessed_letters:
						total_count += self._5_gram[known_1][known_2][known_3][letter][known_4]
						letter_count[j] += self._5_gram[known_1][known_2][known_3][letter][known_4]
						
			# case 5: "XXXX."
			if word[i] != '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] != '.' and word[i+4] == '.':
				known_1= word[i]
				known_2= word[i+1]
				known_3= word[i+2]
				known_4= word[i+3]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._5_gram[known_1][known_2][known_3][known_4][letter] != 0 and letter not in self.guessed_letters:
						total_count += self._5_gram[known_1][known_2][known_3][known_4][letter]
						letter_count[j] += self._5_gram[known_1][known_2][known_3][known_4][letter]
		   
		
		if total_count != 0:
			for i in range(len(self.letter_set)):
				probs[i] = letter_count[i] / total_count
		
		i=0
		while i < len(self.probability):
			self.probability[i] = self.probability[i] + probs[i] * (0.45)
			i+=1
		
		return self.find_4_gram_probability(word)
	
	def find_4_gram_probability(self, word):
		
		letter_count = [0] * len(self.letter_set)
		probs = [0] * len(self.letter_set)
		total_count = 0
		j=0
		for i in range(len(word) - 3):
						
			# case 3: ".XXX "
			if word[i] == '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] != '.':
				known_1= word[i+1]
				known_2= word[i+2]
				known_3= word[i+3]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._4_gram[letter][known_1][known_2][known_3] != 0 and letter not in self.guessed_letters:
						total_count += self._4_gram[letter][known_1][known_2][known_3]
						letter_count[j] += self._4_gram[letter][known_1][known_2][known_3]
			
		   # case 3: "X.XX "
			if word[i] != '.' and word[i+1] == '.' and word[i+2] != '.' and word[i+3] !='.':
				known_1= word[i]
				known_2= word[i+2]
				known_3= word[i+3]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._4_gram[known_1][letter][known_2][known_3] != 0 and letter not in self.guessed_letters:
						total_count += self._4_gram[known_1][letter][known_2][known_3]
						letter_count[j] += self._4_gram[known_1][letter][known_2][known_3]
						
			# case 3: "XX.X "
			if word[i] != '.' and word[i+1] != '.' and word[i+2] == '.' and word[i+3] !='.':
				known_1= word[i]
				known_2= word[i+1]
				known_3= word[i+3]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._4_gram[known_1][known_2][letter][known_3] != 0 and letter not in self.guessed_letters:
						total_count += self._4_gram[known_1][known_2][letter][known_3]
						letter_count[j] += self._4_gram[known_1][known_2][letter][known_3]
						
			# case 3: "XXX. "
			if word[i] != '.' and word[i+1] != '.' and word[i+2] != '.' and word[i+3] =='.':
				known_1= word[i]
				known_2= word[i+1]
				known_3= word[i+2]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._4_gram[known_1][known_2][known_3][letter] != 0 and letter not in self.guessed_letters:
						total_count += self._4_gram[known_1][known_2][known_3][letter]
						letter_count[j] += self._4_gram[known_1][known_2][known_3][letter]

		if total_count != 0:
			for i in range(len(self.letter_set)):
				probs[i] = letter_count[i] / total_count
		
		i=0
		while i < len(self.probability):
			self.probability[i] = self.probability[i] + probs[i] * (0.35)
			i+=1

		return self.find_3_gram_probability(word)
	
	def find_3_gram_probability(self, word):
		
		letter_count = [0] * len(self.letter_set)
		probs = [0] * len(self.letter_set)
		total_count = 0
		i=0
		j=0
		for i in range(len(word) - 2):

			if word[i] == '.' and word[i+1] != '.' and word[i+2] != '.':
				known_1 = word[i+1]
				known_2= word[i+2]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._3_gram[letter][known_1][known_2] != 0 and letter not in self.guessed_letters:
						total_count += self._3_gram[letter][known_1][known_2]
						letter_count[j] += self._3_gram[letter][known_1][known_2]

			if word[i] != '.' and word[i+1] == '.' and word[i+2] != '.':
				known_1 = word[i]
				known_2 = word[i+2]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._3_gram[known_1][letter][known_2] != 0 and letter not in self.guessed_letters:
						total_count += self._3_gram[known_1][letter][known_2]
						letter_count[j] += self._3_gram[known_1][letter][known_2]

			if word[i] != '.' and word[i+1] != '.' and word[i+2] == '.':
				known_1 = word[i]
				known_2 = word[i+1]
				
				for j in range(len(self.letter_set)):
					letter=self.letter_set[j]
					if self._3_gram[known_1][known_2][letter] != 0 and letter not in self.guessed_letters:
						total_count += self._3_gram[known_1][known_2][letter]
						letter_count[j] += self._3_gram[known_1][known_2][letter]
		   
		if total_count != 0:
			for i in range(len(self.letter_set)):
				probs[i] = letter_count[i] / total_count
		i=0
		while i < len(self.probability):
			self.probability[i] = self.probability[i] + probs[i] * (0.25)
			i+=1
		return self.find_2_gram_probability(word)
	
	def find_2_gram_probability(self, word):
		
		letter_count = [0] * len(self.letter_set)
		probs = [0] * len(self.letter_set)
		total_count = 0
		j=0

		for i in range(len(word) - 1):
			# case 1: ".X"
			if word[i] == '.' and word[i+1]!= '.':
				known = word[i+1]
				
				for j in range(len(self.letter_set)):
					letter = self.letter_set[j]
					if self._2_gram[len(word)][letter][known] != 0 and letter not in self.guessed_letters:
						total_count += self._2_gram[len(word)][letter][known]
						letter_count[j] += self._2_gram[len(word)][letter][known]
						
			# case 2: "X."
			if word[i] != '.' and word[i+1] == '.':
				known = word[i]
				
				for j in range(len(self.letter_set)):
					letter = self.letter_set[j]
					if self._2_gram[len(word)][known][letter] != 0 and letter not in self.guessed_letters:
						total_count += self._2_gram[len(word)][known][letter]
						letter_count[j] += self._2_gram[len(word)][known][letter]
				  
																	
		if total_count != 0:
			for i in range(len(self.letter_set)):
				probs[i] = letter_count[i] / total_count

		i = 0
		while i < len(self.probability):
			self.probability[i] = self.probability[i] + probs[i] * 0.15
			i += 1

		return self.find_1_gram_probability(word)
	
	def find_1_gram_probability(self, word):
	
		letter_count = [0] * len(self.letter_set)
		probs = [0] * len(self.letter_set)
		
		total_count = 0
		

		for i in range(len(word)):
			# case 1: ". only"
			if word[i] == '.':
				j=0           
				while j < len(self.letter_set):
					letter = self.letter_set[j]
					if self._1_gram[len(word)][letter] != 0 and letter not in self.guessed_letters:
						total_count += self._1_gram[len(word)][letter]
						letter_count[j] += self._1_gram[len(word)][letter]
					j += 1
					   
		if total_count != 0:
			for i in range(len(self.letter_set)):
				probs[i] = letter_count[i] / total_count
				
		i = 0
		while i < len(self.probability):
			self.probability[i] = self.probability[i] + probs[i] * 0.06
			i += 1
		
		max_prob = 0
		guess_letter = ''
		i = 0
		while i < len(self.letter_set):
			if self.probability[i] > max_prob:
				max_prob = self.probability[i]
				guess_letter = self.letter_set[i]
			i += 1
		
		# Put special focus on letters which have higher frequency in English Language Words (E is highest)
		if guess_letter == '':
			letters = self.letter_set.copy()
			random.shuffle(letters)
			letters_random =  ['e','a','r','i','o','t','n','u'] + letters 
			for letter in letters_random:
				if letter not in self.guessed_letters:
					return letter
			
		return guess_letter

	def build_dictionary(self, dictionary_file_location):
		text_file = open(dictionary_file_location,"r")
		full_dictionary = text_file.read().splitlines()
		text_file.close()
		return full_dictionary

	def reset(self):
		self.guessed_letters = []
		self.current_dictionary = self.full_dictionary

	def upd(self, prev_letter):
		self.guessed_letters.append(prev_letter)
 
def main():
	with open('app/base', 'r') as f:
		hangman_words = [line.strip() for line in f]

	games = len(hangman_words)
	guesses = 0
	wins = 0

	hangman = HangmanAPI()

	for word in hangman_words:

		incorrect_guesses = 0
		mask = ['_', ' '] * len(word)
		while True:	
			guess = hangman.guess(''.join(mask))
			hangman.upd(guess)
			if guess in word:
				for i in range(len(word)):
					if word[i] == guess:
						mask[2*i] = guess
			else:
				incorrect_guesses += 1

			if('_' not in mask):
				print("You win!")
				wins += 1
				guesses += incorrect_guesses
				break
			elif incorrect_guesses == 6:
				print("You lose!")
				break
		
		print(f'Game over: word : {word}, mask : {mask}')

		hangman.reset()

	print(f'Accuracy of model is {wins/games} with an average of {guesses / games} guesses per game')
if __name__ == '__main__':
	main()