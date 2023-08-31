import nltk
from nltk.corpus import wordnet

class WordLadderAPI:
    def __init__(self):
        nltk.download('wordnet')

    def is_valid_word(self, word, word_list):
        return word in word_list

    def get_word_neighbors(self, word, word_list):
        neighbors = set()
        for i in range(len(word)):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                neighbor = word[:i] + char + word[i+1:]
                if self.is_valid_word(neighbor, word_list) and neighbor != word:
                    neighbors.add(neighbor)
        return list(neighbors)

    def word_transformer(self, start_word, end_word, word_list):
        start_word = start_word
        end_word = end_word

        if not (self.is_valid_word(start_word, word_list) and self.is_valid_word(end_word, word_list)):
            return "Invalid words."

        visited = set()
        queue = [[start_word]]

        while queue:
            path = queue.pop(0)
            current_word = path[-1]

            if current_word == end_word:
                return path

            if current_word not in visited:
                visited.add(current_word)
                neighbors = self.get_word_neighbors(current_word, word_list)

                for neighbor in neighbors:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        return queue if len(queue) > 0 else "No valid transformation found."
