import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def train_model(training_data):
    training = pd.read_csv(training_data)
    cols = training.columns
    cols = cols[:-1]
    x = training[cols]
    y = training["prognosis"]

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42
    )

    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train, y_train)

    scores = cross_val_score(clf, x_test, y_test, cv=3)
    mean_score = scores.mean()

    model = SVC()
    model.fit(x_train, y_train)
    svm_score = model.score(x_test, y_test)

    return mean_score, svm_score, clf, le, cols


def get_symptoms_dict(training_data):
    df = pd.read_csv(training_data)
    symptoms_dict = {symptom: index for index, symptom in enumerate(df.columns)}
    return symptoms_dict


def sec_predict(symptoms_exp, symptoms_dict, clf):
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1
    return clf.predict([input_vector])


def print_disease(node, le):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(
    tree,
    feature_names,
    description_list,
    precautionDictionary,
    disease_input,
    num_days,
    le,
    reduced_data,
):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    if disease_input not in chk_dis:
        return "Enter valid symptom."

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                return recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                return recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node], le)
            if present_disease[0] in reduced_data.index:
                symptoms_given = reduced_data.columns[
                    reduced_data.loc[present_disease].values[0].nonzero()
                ]
                symptoms_exp = []
                for syms in list(symptoms_given):
                    inp = input(syms + "? (yes/no): ")
                    while inp.lower() not in ["yes", "no"]:
                        inp = input("Please provide a valid input (yes/no): ")
                    if inp.lower() == "yes":
                        symptoms_exp.append(syms)

                second_prediction = sec_predict(symptoms_exp, symptoms_dict, clf)
                calc_condition(symptoms_exp, num_days)

                if present_disease[0] == second_prediction[0]:
                    return (
                        "You may have "
                        + present_disease[0]
                        + "\n"
                        + description_list[present_disease[0]]
                    )

                else:
                    return (
                        "You may have "
                        + present_disease[0]
                        + " or "
                        + second_prediction[0]
                        + "\n"
                        + description_list[present_disease[0]]
                        + "\n"
                        + description_list[second_prediction[0]]
                    )

                precution_list = precautionDictionary[present_disease[0]]
                precutions = "\n".join(
                    [str(i + 1) + ") " + j for i, j in enumerate(precution_list)]
                )
                return "Take the following measures: \n" + precutions
            else:
                return "Error: The predicted disease is not present in the data."

    return recurse(0, 1)


def get_description_list():
    description_list = {}
    try:
        with open("symptom_Description.csv", "r") as file:
            reader = csv.reader(file)
            for row in reader:
                symptom = row[0]
                description = row[1]
                description_list[symptom] = description
    except FileNotFoundError:
        print("Symptom descriptions file not found.")
    return description_list


def get_precaution_dict():

    precaution_dict = {}
    try:
        with open("symptom_precaution.csv", "r") as file:
            reader = csv.reader(file)
            for row in reader:
                disease = row[0]
                precautions = row[1:]
                precaution_dict[disease] = precautions
    except FileNotFoundError:
        print("Precaution data file not found.")
    return precaution_dict


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(" ", "_")
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        print("You should take the consultation from a doctor.")
    else:
        print("It might not be that bad, but you should take precautions.")


def main():
    training_data = "Data/Training.csv"
    testing_data = "Data/Testing.csv"

    # Placeholder function calls
    description_list = {}  # Call a function to populate this dictionary
    severityDictionary = {}  # Call a function to populate this dictionary
    precautionDictionary = {}  # Call a function to populate this dictionary

    print(
        "Hello, User"
    )  # Assuming the functionality of getInfo("User") is to print a greeting

    mean_score, svm_score, clf, le, cols = train_model(training_data)
    symptoms_dict = get_symptoms_dict(training_data)
    disease_input = input("Enter the symptom you are experiencing: ")
    num_days = int(input("From how many days are you experiencing this? : "))
    reduced_data = pd.DataFrame()  # Placeholder for reduced data
    tree_to_code(
        clf,
        cols,
        description_list,
        precautionDictionary,
        disease_input,
        num_days,
        le,
        reduced_data,
    )


if __name__ == "__main__":
    main()
