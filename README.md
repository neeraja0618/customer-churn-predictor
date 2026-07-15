Customer Churn Predictor
Live demo: https://customer-churn-predictor-kn.streamlit.app/
This is a machine learning project I built to predict whether a telecom customer is likely to churn (cancel their service), using a public dataset of about 7,000 customers. Instead of just outputting a prediction, the app also explains why it made that prediction and suggests what could be done to retain that specific customer.

Why this problem
Telecom companies lose a lot of money when customers leave, and it's much cheaper to retain an existing customer than acquire a new one. If a company can figure out who's likely to churn before it happens, they can step in with an offer or support outreach instead of finding out too late. I wanted to build something that reflected a real business use case rather than just a generic classification exercise.

What I did
- Cleaned the raw dataset — found 11 rows with a data quality issue (blank TotalCharges values, all belonging to brand-new customers with 0 months tenure) and fixed it by filling with 0 instead of dropping the rows
- Explored the data to find what actually correlates with churn — contract type, tenure, internet service type, tech support, etc.
- Encoded the categorical columns and scaled the numeric ones so the model could actually use them properly
- Trained two different models (Logistic Regression and Random Forest) and compared them — not just on accuracy, but on recall for the churn class, since missing an actual churner matters more than a false alarm for this problem
- Built a Streamlit app around the final model, including a breakdown of which factors are driving each individual prediction, and some plain-language retention suggestions based on those factors
- Deployed it so it's actually usable as a live link, not just a local notebook

 Some of what I found in the data
- Month-to-month customers churn at about 43%, compared to under 3% for two-year contracts — by far the strongest single factor
- Customers who churned had been with the company for 18 months on average, versus 37.6 months for those who stayed
- Fiber optic customers churn more than twice as often as DSL customers (42% vs 19%), which was a bit counterintuitive at first — you'd expect the "better" internet option to have happier customers
- No tech support roughly triples the churn rate compared to having it

 Model results
I ended up going with Logistic Regression over Random Forest, even though Random Forest had slightly better accuracy (77% vs 74%). The reason is recall — Logistic Regression catches 79% of actual churners in the test set, compared to 70% for Random Forest. For this specific problem, missing a real churner is worse than occasionally flagging someone who wasn't going to leave, so I optimized for that instead of raw accuracy. Logistic Regression's coefficients are also directly interpretable, which is what powers the "why" explanation in the app.

Metric            -      Logistic Regression 
Accuracy                        74% 
Recall (Churn)                  79% 
Precision (Churn)               51% 

Tech stack
Python, pandas, scikit-learn, Matplotlib, Streamlit

Files in this repo
- Telco-Customer-Churn.csv — the raw dataset
- model_training_and_analysis.py — the full pipeline: cleaning, exploration, encoding, training, comparing both models
- app.py — the Streamlit app
- model.pkl, scaler.pkl, columns.pkl — the saved trained model
- requirements.txt — dependencies

Running it locally
```bash
pip install -r requirements.txt
python model_training_and_analysis.py
streamlit run app.py
```
 Things I'd improve with more time
- Try a gradient boosting model (XGBoost) for comparison
- Use SHAP values for a more rigorous explanation method instead of raw coefficients
- Let the retention threshold be adjustable based on the actual cost of a false alarm vs a missed churner, instead of a fixed 0.5 cutoff
