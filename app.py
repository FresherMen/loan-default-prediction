from logging import debug
from flask import Flask, render_template, request, redirect, url_for
import utils  
from utils import preprocessdata 

app = Flask(__name__) 

@app.route('/') 
def home(): 
    return render_template('index.html') 

@app.route('/predict/', methods=['GET', 'POST'])
def predict():  
    if request.method == 'GET':
        return redirect(url_for('home'))

    # Extract form inputs safely with defaults
    gender = request.form.get('Gender', '1')
    married = request.form.get('Married', '1')
    education = request.form.get('Education', '1')
    self_employed = request.form.get('Self_Employed', '0')  
    applicant_income = request.form.get('ApplicantIncome', '0')  
    coapplicant_income = request.form.get('CoapplicantIncome', '0') 
    loan_amount = request.form.get('LoanAmount', '0')   
    loan_amount_term = request.form.get('Loan_Amount_Term', '360')   
    credit_history = request.form.get('Credit_History', '1')   
    property_area = request.form.get('Property_Area', '1')  

    prediction = utils.preprocessdata(
        gender, married, education, self_employed, applicant_income,
        coapplicant_income, loan_amount, loan_amount_term, credit_history,
        property_area
    )

    return render_template('predict.html', prediction=prediction) 

if __name__ == '__main__': 
    app.run(debug=True) 
