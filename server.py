from flask import Flask, request, jsonify, render_template

from util import *


def create_app():
    the_app = Flask(__name__)
    load_model()
    return the_app


app = create_app()


@app.route("/")
def home():
    return render_template('form.html')


@app.route("/predict_loan", methods=["POST"])
def predict():
    data = [
        request.form['gender'],
        request.form['married'],
        request.form['dependents'],
        request.form['education'],
        request.form['self_employed'],
        int(request.form['applicant_income']),
        int(request.form['coapplicant_income']),
        int(request.form['loan_amount']),
        request.form['loan_amount_term'],
        request.form['credit_history'],
        request.form['property_area']
    ]

    response = jsonify({
        'approval_status': bool(get_predict_is_loan_approved(data))
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    app.run()
