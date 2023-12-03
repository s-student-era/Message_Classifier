from flask import Flask, render_template, request
import pickle


app=Flask(__name__)

model=pickle.load(open("nb_model.pkl","rb"))
vectorizer=pickle.load(open("extractor.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        new_message=request.form["input_message"]
        input_message=[new_message]
        cooked_message=vectorizer.transform(input_message)
        prediction = model.predict(cooked_message)
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__" :
    app.run(debug=True)