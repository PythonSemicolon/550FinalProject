import json
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import SVC

# Model candidates: Naive Bayes, Logistic Regression, SVM, Decision Tree, Random Forest, etc.
def train_model(author, data):
    """
    Returns an optimized, trained model for an author,
    as well as the optimal hyperparameters.
    """
    X_train = data[author]['x_train']
    y_train = data[author]['y_train']
    X_dev = data[author]['x_dev']
    y_dev = data[author]['y_dev']

    # Combine training and dev sets for predefined split
    # -1 indicates training data, 0 indicates the dev set
    split_indices = [-1] * len(X_train) + [0] * len(X_dev)
    ps = PredefinedSplit(test_fold=split_indices)

    # Create a linear Support Vector Machine (SVM) classifier
    classifier = SVC()

    # Define the parameter grid
    param_grid = {'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']}

    # Create a GridSearchCV object with predefined split
    grid_search = GridSearchCV(classifier, param_grid, cv=ps, scoring='accuracy')

    # Combine the training and dev sets
    X_combined = np.vstack((X_train, X_dev))
    y_combined = np.concatenate((y_train, y_dev))

    # Fit the model to the combined data (training + dev)
    grid_search.fit(X_combined, y_combined)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Training accuracy using the best model
    dev_accuracy = best_model.score(X_dev, y_dev)
    print(f"Best hyperparameters for {author}:", best_params)
    print("Dev accuracy:", dev_accuracy)   

    return best_model, best_params


def main():

    # Read data for translators
    with open("data/translators.json", 'r') as file:
        translators_data = json.load(file)

    # Train 3 translator models
    #translators = os.listdir("data/translators")
    #translator_models = dict.fromkeys(translators, None)
    #for t in translators:
    #    translator_models[t] = train_model(t, translators_data)

    # Evaluate on the dev set
    # Measures how well it generalizes to unseen data
    #best_model = translator_models['borges'][0]
    borges_model = train_model('borges', translators_data)[0]
    kovaleva_model = train_model('rajt-kovaleva', translators_data)[0]
    park_model = train_model('park-jung-so', translators_data)[0]
    """
    X_dev = translators_data['borges']['x_dev']
    y_dev = translators_data['borges']['y_dev']
    dev_accuracy = borges_model.score(X_dev, y_dev)
    print("Dev set accuracy:", dev_accuracy)
    """

    # This is the point to readjust the model if it is overfitting

    # Read data for authors
    with open("data/authors.json", 'r') as file:
        authors_data = json.load(file)

    # Train 3 author models
    # authors = os.listdir("data/authors")
    woolf_model = train_model('virginia_woolf', authors_data)[0]
    vonnegut_model = train_model('kurt_vonnegut', authors_data)[0]
    twain_model = train_model('mark_twain', authors_data)[0]
    """
    X_dev = translators_data['virginia_woolf']['x_dev']
    y_dev = translators_data['virginia_woolf']['y_dev']
    dev_accuracy = woolf_model.score(X_dev, y_dev)
    print("Dev set accuracy:", dev_accuracy)
    """

    # Test Woolf model on Borges' testing data.
    woolf_score = woolf_model.score(translators_data['borges']['x_test'], translators_data['borges']['y_test'])
    # Test Borges' model on Borges' testing data.
    borges_score = borges_model.score(translators_data['borges']['x_test'], translators_data['borges']['y_test'])
    print("Does the model recognize Woolf's style: ", woolf_score)
    print("Does the model recognize Borges' style: ", borges_score)
    print("----------------")

    # Test Vonnegut model on Rajt-Kovaleva's testing data.
    vonnegut_score = vonnegut_model.score(translators_data['rajt-kovaleva']['x_test'], translators_data['rajt-kovaleva']['y_test'])
    # Test Rajt-Kovaleva's model on Rajt-Kovaleva's testing data.
    kovaleva_score = kovaleva_model.score(translators_data['rajt-kovaleva']['x_test'], translators_data['rajt-kovaleva']['y_test'])
    print("Does the model recognize Vonnegut's style: ", vonnegut_score)
    print("DOes the model recognize Kovaleva's style: ", kovaleva_score)
    print("----------------")
    # Test Twain model on Park-Jung-So's testing data.
    twain_score = twain_model.score(translators_data['park-jung-so']['x_test'], translators_data['park-jung-so']['y_test'])
    # Test Park-Jung-So's model on Park-Jung-So's testing data.
    park_score = park_model.score(translators_data['park-jung-so']['x_test'], translators_data['park-jung-so']['y_test'])
    print("Does the model recognize Twain's style: ", twain_score)
    print("Does the model recognize Park-Jung-So's style: ", park_score)
    print("----------------")


    """
    Interpret the results

    (results_author, results_translator)

    The translator models that perform the best indicate that the translators have the most distinguished style of a translator.
    The author models that perform the best indicate the translators that have best conveyed the original author's style.

    (1, 0) -> Translator doesn't have a style of their own.
    (0, 1) -> Translator has a style of their own and couldn't convey the author's style.
    (1, 1) -> Translator has a style of their own and could convey the author's style.
    (0, 0) -> Translator doesn't have a style of their own and could not convey the author's style.
            -> OR model is not good enough to distinguish between the two.
            -> OR language is not a good indicator of style.

    """

if __name__ == '__main__':
    main()