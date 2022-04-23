import requests

requests_sentences = requests.get(r'https://d3c33hcgiwev3.cloudfront.net/_3a8d746cf4d86fba2f31586f239d11fd_sentences.txt?Expires=1560902400&Signature=JQAoPnhRiM7g6rW4LEHJzMWC2HXUJQ3ycHN4JA38gq0CKS1fz046SAEF3Zsdq2MnfbRH7lVrX-Yhj7xU69wsH-5fLHN3gnnJtOLriwzcbsvdjurxoDpmY8rmNiFqPoeXllNyJz1~DDxYoc5wfZdTa5HC9pXuAkOaczsJFFFoVNU_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A')
sentences = requests_sentences.text.split('\n')

sentences =  [line.lower() for line in sentences]

import re

# Разбиение каждой строки на слова с помощью регулярного выражения
sentences = [re.split('[^a-z]', str(word)) for word in sentences]
# Преобразование двухмерного списка строк в одномерный список слов
words = sum(sentences,[])
# Удаление "пустых" слов
words = [word for word in words if word]


# Переопределим список слов как set, чтобы избавиться от дубликатов
# Контекст задачи позволяет не учитывать формы слов (к примеру cat и cats считать разными словами)
words = set(words)
# Создадим словарь формата ключ: слово, значение: индекс
dict_of_words = {index: word for index, word in enumerate(words)}
print(dict_of_words) # вот как это выглядит
import numpy as np
from collections import Counter

n = len(sentences) - 1 # пустую строку не учитываем
d = len(dict_of_words)

matrix = np.zeros((n, d), dtype=int)

for i in range(len(matrix)): # перебор индексов строк матрицы
    how_many_words = Counter(sentences[i]) # cловарь формата слово: количество в предложении
    matrix[i] = [
        how_many_words[dict_of_words[j]] for j in range(len(matrix[i]))
    ]

print(matrix.shape) # контроль размера
# первое предложение для примера


from scipy.spatial.distance import cosine

# Определим словарь формата косинусное расстояние: номер предложения
distances = {cosine(matrix[0], matrix[i]): i for i in range(len(matrix))}

answer = []
for distance, num in sorted(distances.items())[1:3]:
    print(distance, num)
    answer.append(str(num))

with open('C1W2 answer 1.txt', 'w') as output_file:
    output_file.write(' '.join(answer))




import matplotlib.pyplot as plot
import math
import scipy


def f(x):
    return math.sin(x / 5) * math.exp(x / 10) + 5 * math.exp(-x / 2)


data_x = np.arange(1, 16, 0.1)
data_y = [f(x) for x in data_x]

plot.plot(data_x, data_y)


def create_matrix(points): # вынесем повторяемый код в функции
    matrix = [
        [x**i for i in range(len(points))]
        for x in points
    ]
    return np.array(matrix)
def approximate(w, vector):
    return np.array([
        sum([w[i]*(v**i) for i in range(len(w))])
        for v in vector
    ])
def solve(points):
    A = create_matrix(points)
    b = [f(x) for x in points]
    w = scipy.linalg.solve(A, b)
    return w, approximate(w, b)
solve([1, 15])


solve([1, 8, 15])



answer = solve([1, 4, 10, 15])[0]
answer = ' '.join([str(a) for a in answer])
print(answer)



with open('C1W2 answer 2.txt', 'w') as output_file:
    output_file.write(answer)

