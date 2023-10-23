import os


def load_data():
    path = "."
    artPath = os.path.join(path, '../bbc_data/bbc.mtx')
    vocPath = os.path.join(path, '../bbc_data/bbc.terms')
    labPath = os.path.join(path, '../bbc_data/bbc.classes')

    # Map holding word as a key and thier index as their value
    vocab = {}
    counter = 0  # to keep track of the indecies

    fvocab = open(vocPath, 'r')
    for line in fvocab:
        vocab[line.replace('\n', '')] = counter
        counter += 1
    fvocab.close()

    farticles = open(artPath, 'r')
    farticles.readline()

    # Read sizing data
    sizing = farticles.readline().split(" ")

    # a list containing the number of occurence of the words in each document
    articles = [[0] * int(sizing[0]) for _ in range(int(sizing[1]))]
    for line in farticles:
        point = line.replace('\n', '').split(" ")
        articles[int(point[1])-1][int(point[0])-1] = float(point[2])
    farticles.close()

    flabels = open(labPath, 'r')
    flabels.readline()
    flabels.readline()

    keyValues = flabels.readline().replace('\n', '')

    # holding the labels as a list, namely business,entertainment,politics,sport,tech
    keys = keyValues.split(" ")[2].split(",")
    flabels.readline()

    labels = []
    for line in flabels:
        point = line.replace('\n', '').split(" ")
        labels.append(int(point[1]))
    flabels.close()

    return vocab, articles, labels, keys
