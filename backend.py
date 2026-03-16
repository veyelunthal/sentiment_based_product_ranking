from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
@app.route("/", methods=["GET","POST"])
def home():
    rating = None
    review = ""
    if request.method == "POST":

        review = request.form["review"]

        review_vector = vectorizer.transform([review])

        rating = model.predict(review_vector)[0]

    return render_template("frontend.html", rating=rating, review=review)
if __name__ == "__main__":
    app.run(debug=True)
