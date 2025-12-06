import math

def tachdoan(text):
    return text.split()

def train(dataset):
    word_counts = {'spam': {}, 'ham': {}}
    class_counts = {'spam': 0, 'ham': 0}
    vocab = set()

    for label, text in dataset:
        words = tachdoan(text)
        class_counts[label] += 1

        for word in words:
            vocab.add(word)
            word_counts[label][word] = word_counts[label].get(word, 0) + 1

    return word_counts, class_counts, vocab


def predict(text, word_counts, class_counts, vocab):
    words = tachdoan(text)
    total_docs = sum(class_counts.values())
    log_prob = {}

    for label in ['spam', 'ham']:
        log_prob[label] = math.log(class_counts[label] / total_docs)
        total_words = sum(word_counts[label].values())

        for word in words:
            word_freq = word_counts[label].get(word, 0) + 1
            word_prob = word_freq / (total_words + len(vocab))
            log_prob[label] += math.log(word_prob)

    return max(log_prob, key=log_prob.get)