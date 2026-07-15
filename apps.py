import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('columns.pkl')

AVERAGE_CHURN_RATE = 26.5  # known baseline from the full dataset

FEATURE_INSIGHTS = {
    'Contract_Month-to-month': {
        'reason': "No long-term commitment makes it easy to leave anytime.",
        'action': "Offer a discount to switch to a 1-year or 2-year contract — this is historically the single strongest retention lever."
    },
    'InternetService_Fiber optic': {
        'reason': "Fiber customers churn far more than DSL customers in this data, likely due to price or reliability concerns.",
        'action': "Review fiber pricing/reliability, or offer a loyalty bundle discount for fiber users."
    },
    'TechSupport_No': {
        'reason': "No tech support means unresolved issues can build frustration.",
        'action': "Offer a free trial of tech support — customers with support churn about half as often."
    },
    'OnlineSecurity_No': {
        'reason': "Lack of security add-ons is linked with higher churn.",
        'action': "Bundle in free online security for a few months to increase perceived value."
    },
    'StreamingTV_Yes': {
        'reason': "Streaming subscribers tend to have higher bills, which correlates with churn.",
        'action': "Offer a loyalty discount on the streaming bundle to offset the higher cost."
    },
    'StreamingMovies_Yes': {
        'reason': "Similar to Streaming TV — higher bill, higher churn tendency.",
        'action': "Bundle streaming services at a discount for long-term subscribers."
    },
    'PaymentMethod_Electronic check': {
        'reason': "Electronic check users churn more than customers on automatic payments.",
        'action': "Encourage switching to auto-pay (bank transfer/credit card) with a small incentive — it correlates with much lower churn."
    },
    'PaperlessBilling': {
        'reason': "Paperless billing customers show a slightly higher churn tendency, possibly linked to lower engagement.",
        'action': "No direct action needed — minor factor, monitor alongside bigger risks."
    },
    'TotalCharges': {
        'reason': "Higher accumulated spend can make customers more price-sensitive and likely to shop around.",
        'action': "Offer a loyalty discount or bundle review for high-total-spend customers."
    },
}

TENURE_INSIGHT = {
    'reason': "New customers haven't built loyalty yet — churn is highest in the first few months.",
    'action': "Set up proactive check-ins or a welcome offer during the first 3 months of service."
}

st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's details to estimate churn risk, understand *why*, and get suggested retention actions.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months as customer)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

with col2:
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges = st.number_input("Total Charges ($ so far)", 0.0, 9000.0, float(tenure * monthly_charges))

if st.button("Predict Churn Risk 🔍"):
    raw_input = pd.DataFrame([{
        'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone, 'MultipleLines': multiple_lines,
        'InternetService': internet, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }])

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        raw_input[col] = raw_input[col].map(lambda x: 1 if x in ['Yes', 'Male'] else 0)

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
    encoded_input = pd.get_dummies(raw_input, columns=multi_cols)
    encoded_input = encoded_input.reindex(columns=model_columns, fill_value=0)

    scaled_input = scaler.transform(encoded_input)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — {probability*100:.1f}% estimated probability")
    else:
        st.success(f"✅ Low Churn Risk — {probability*100:.1f}% estimated probability")

    diff = probability * 100 - AVERAGE_CHURN_RATE
    if diff > 0:
        st.write(f"This is **{diff:.1f} percentage points higher** than the average customer's churn rate ({AVERAGE_CHURN_RATE}%).")
    else:
        st.write(f"This is **{abs(diff):.1f} percentage points lower** than the average customer's churn rate ({AVERAGE_CHURN_RATE}%).")

    st.subheader("🔍 Why this prediction — top contributing factors")

    contributions = model.coef_[0] * scaled_input[0]
    contrib_df = pd.DataFrame({
        'Feature': model_columns,
        'Contribution': contributions
    })
    active_mask = (encoded_input.iloc[0] != 0).values
    contrib_df = contrib_df[active_mask].sort_values('Contribution', ascending=False)

    top_risk = contrib_df[contrib_df['Contribution'] > 0].head(4)
    top_protective = contrib_df[contrib_df['Contribution'] < 0].tail(3)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df = pd.concat([top_risk, top_protective]).sort_values('Contribution')
    colors = ['#e74c3c' if v > 0 else '#27ae60' for v in plot_df['Contribution']]
    ax.barh(plot_df['Feature'], plot_df['Contribution'], color=colors)
    ax.set_xlabel('Contribution to churn risk (red = increases, green = decreases)')
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("💡 Suggested retention actions")

    shown_any = False
    for _, row in top_risk.iterrows():
        feat = row['Feature']
        if feat in FEATURE_INSIGHTS:
            info = FEATURE_INSIGHTS[feat]
            st.markdown(f"**{feat.replace('_', ': ')}** — {info['reason']}  \n👉 *{info['action']}*")
            shown_any = True

    if tenure <= 6 and prediction == 1:
        st.markdown(f"**Low tenure ({tenure} months)** — {TENURE_INSIGHT['reason']}  \n👉 *{TENURE_INSIGHT['action']}*")
        shown_any = True

    if not shown_any:
        st.write("No major risk factors detected — this customer's profile looks stable.")
