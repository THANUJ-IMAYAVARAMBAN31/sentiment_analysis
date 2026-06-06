from flask import Flask, render_template, request

from .predict import predict_sentiment

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        review = request.form["review"]

        pred, conf = predict_sentiment(review)

        result = {
            "sentiment":
                "Positive" if pred == 1 else "Negative",
            "confidence":
                round(conf * 100, 2)
        }

    return render_template(
        "index.html",
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)