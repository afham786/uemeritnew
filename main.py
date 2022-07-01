import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# load csv file

df = pd.read_csv("morfull2020.csv")

# selection dependant and independant
X = df[["StudyProgram", "MeritScore"]]
Y = df["AddmissionHolder"]


def convert_to_int(word):
    word_dict = {'B.Ed. (Hons)': 1, 'BBA': 2, 'BS Botany': 3, 'BS Chemistry': 4, 'BS English': 5,
                 'BS Information Technology': 6, 'BS Mathematics': 7, 'BS Physics': 8, 'BS Zoology': 9}
    return word_dict[word]


X['StudyProgram'] = X['StudyProgram'].apply(lambda x: convert_to_int(x))

# split the data set into training and testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=50)

# feature scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# intentiate the model

classifier = RandomForestClassifier()

# fit the model

classifier.fit(X_train, Y_train)

# make pickle file for the model

pickle.dump(classifier, open("model.pkl", "wb"))


#   Traning part is complete


from flask import Flask, request, jsonify

# creating app
app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return ("Hello world")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    StudyProgram = request.form.get('StudyProgram')
    Matric = int(request.form.get('Matric'))
    Inter = int(request.form.get('Inter'))

    St = int(StudyProgram)

    a = float(Matric / 1100) * 30
    b = float(Inter / 1100) * 70

    MeritScore = float(a + b)

    input_query = [[St, MeritScore]]
    print(input_query)

    result = model.predict(sc.transform(input_query))
    print(result)
    return jsonify({'placement': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
