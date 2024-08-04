from collections import Counter
import numpy as np
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from porter import stem # type: ignore
from sklearn.svm import SVC


# Function to load stopwords from a file
def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        return set(line.strip() for line in file)

# Function to build a vocabulary from text data
def build_vocabulary(data_dir, vocab_size=1000, stopwords=None):
    counter = Counter()
    for sentiment in ['pos', 'neg']:
        path = os.path.join(data_dir, sentiment)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as text_file:
                words = (stem(word) for word in text_file.read().lower().split() if word not in stopwords)
                counter.update(words)
    with open('vocabulary.txt', 'w', encoding='utf-8') as vocab_file:
        for word, _ in counter.most_common(vocab_size):
            vocab_file.write(f"{word}\n")

# Function to extract features and labels from text data
def extract_features(data_dir, vocab_file, output_file, stopwords):
    with open(vocab_file, 'r') as file:
        vocab = [line.strip() for line in file]
    vocab_set = set(vocab)
    features, labels = [], []
    document_count = 0
    for label, folder in enumerate(['pos', 'neg']):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text_counts = Counter(stem(word) for word in file.read().lower().split() if word not in stopwords and stem(word) in vocab_set)
                features.append([text_counts.get(word, 0) for word in vocab])
                labels.append(label)
            document_count += 1
            if document_count % 100 == 0:
                print(f"Processed {document_count} documents.")
    np.savez_compressed(output_file, features=np.array(features), labels=np.array(labels))
    print(f"Features and labels saved to {output_file}")

# Function to train and evaluate classifiers
def train_and_evaluate_models(train_file, test_file):
    data_train = np.load(train_file)
    train_features, train_labels = data_train['features'], data_train['labels']
    data_test = np.load(test_file)
    test_features, test_labels = data_test['features'], data_test['labels']

    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }

    results = {}
    for name, model in models.items():
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.2f}")

    return results

# Main execution
if __name__ == "__main__":
    stopwords_file = 'stopwords.txt'
    stopwords = load_stopwords(stopwords_file)

    train_dir = 'D:\\UNIPV\\Year 1\\Semester 2\\Machine Learning\\project1\\movie\\dataset\\train'
    test_dir = 'D:\\UNIPV\\Year 1\\Semester 2\\Machine Learning\\project1\\movie\\dataset\\test'
    validation_dir = 'D:\\UNIPV\\Year 1\\Semester 2\\Machine Learning\\project1\\movie\\dataset\\validation'

    build_vocabulary(train_dir, 1000, stopwords)
    extract_features(train_dir, 'vocabulary.txt', 'train_features.npz', stopwords)
    extract_features(test_dir, 'vocabulary.txt', 'test_features.npz', stopwords)
    extract_features(validation_dir, 'vocabulary.txt', 'validation_features.npz', stopwords)
    results = train_and_evaluate_models('train_features.npz', 'test_features.npz')
    print(results)
