import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    /* Base theme */
    .stApp {
        background-color: #0E1117;
    }

    /* Title styling */
    .main-title {
        text-align: center;
        color: white;
        font-size: 2.2rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding: 0 1rem;
    }

    /* Top warning banner */
    .warning-banner {
        background-color: #1A1E23;
        border-left: 4px solid #FF6B6B;
        padding: 8px 16px;
        margin: 1rem 0;
        color: #E0E0E0;
        font-size: 0.9rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1A1E23;
        padding: 8px 16px;
        border: none;
        color: #9CA3AF;
        border-radius: 4px;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #37FDD1;
        color: #0E1117;
    }

    /* Category headers */
    .category-header {
        background-color: #1A1E23;
        border-left: 4px solid #37FDD1;
        padding: 12px 16px;
        margin: 1.5rem 0 1rem 0;
        border-radius: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .category-header .icon {
        color: #37FDD1;
    }

    .category-header .text {
        color: #E0E0E0;
        font-size: 1.1rem;
        font-weight: 500;
    }

    /* Form sections */
    .form-section {
        background-color: #1A1E23;
        border: 1px solid #2D3748;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* Input styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #2D3748 !important;
        border: 1px solid #4A5568 !important;
        color: white !important;
        border-radius: 6px !important;
    }

    /* Slider styling */
    .stSlider > div > div {
        background-color: #2D3748 !important;
    }

    .stSlider > div > div > div > div {
        background-color: #37FDD1 !important;
    }

    /* Submit button */
    .submit-container {
        display: flex;
        justify-content: center;
        padding: 2rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%) !important;
        color: white !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        padding: 1.5rem 3rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        min-width: 400px !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4) !important;
        background: linear-gradient(135deg, #FF8E8E 0%, #FF6B6B 100%) !important;
    }

    /* About section styling */
    .about-container {
        padding: 2rem;
        color: #E0E0E0;
    }

    .about-heading {
        color: #37FDD1;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2D3748;
    }

    .about-section {
        background-color: #1A1E23;
        border: 1px solid #2D3748;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .about-section h3 {
        color: #37FDD1;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }

    .about-section p {
        color: #9CA3AF;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .about-section ul {
        list-style-type: none;
        padding-left: 0;
        margin: 0.5rem 0;
    }

    .about-section li {
        color: #9CA3AF;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }

    .about-section li:before {
        content: "•";
        color: #37FDD1;
        position: absolute;
        left: 0;
    }

    /* Error message */
    .error-message {
        background-color: rgba(255, 107, 107, 0.1);
        border-left: 4px solid #FF6B6B;
        color: #FF6B6B;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("loan_data.csv")
    X = data.drop(columns=['loan_status', 'loan_int_rate', 'loan_percent_income'])
    y = data['loan_status']
    categorical_columns = ['person_gender', 'person_education', 'person_home_ownership', 
                          'loan_intent', 'previous_loan_defaults_on_file']
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    return X, y, X.columns.tolist()

# Train model
@st.cache_resource
def train_model(X, y):
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X, y)
    return clf

# Load data and train model
X, y, feature_columns = load_data()
clf = train_model(X, y)

def main():
    # Title
    st.markdown('<h1 class="main-title">🏦 Loan Approval Predictor</h1>', unsafe_allow_html=True)
    
    # Minimal warning banner
    st.markdown("""
        <div class="warning-banner">
            ⚠️ This is a demonstration model for educational purposes only
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["📝 Application", "ℹ️ About"])
    
    with tab1:
        with st.form("loan_form"):
            # Loan Amount Section
            st.markdown("""
                <div class="category-header">
                    <span class="icon">💰</span>
                    <span class="text">Loan Amount</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            loan_amount = st.slider(
                "Select amount ($)",
                min_value=1000,
                max_value=50000,
                step=1000,
                value=1000
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Personal Information
            st.markdown("""
                <div class="category-header">
                    <span class="icon">👤</span>
                    <span class="text">Personal Information</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100)
                income = st.number_input("Annual Income ($)", min_value=0, step=1000)
            with col2:
                employment = st.number_input("Years of Employment", min_value=0)
                gender = st.selectbox("Gender", ["Select Gender", "Male", "Female"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Financial Information
            st.markdown("""
                <div class="category-header">
                    <span class="icon">💳</span>
                    <span class="text">Financial Information</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
                credit_history = st.number_input("Credit History (years)", min_value=0)
            with col4:
                home_ownership = st.selectbox("Home Ownership", ["Select Status", "Rent", "Own", "Mortgage", "Other"])
                defaults = st.selectbox("Previous Defaults", ["Select Option", "No", "Yes"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Additional Details
            st.markdown("""
                <div class="category-header">
                    <span class="icon">📋</span>
                    <span class="text">Additional Details</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            col5, col6 = st.columns(2)
            with col5:
                education = st.selectbox("Education Level", ["Select Level", "Primary", "Secondary", "Tertiary", "Postgraduate"])
            with col6:
                purpose = st.selectbox("Loan Purpose", ["Select Purpose", "Venture", "Personal", "Medical", "Education", "Debt Consolidation"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Submit button
            st.markdown('<div class="submit-container">', unsafe_allow_html=True)
            submitted = st.form_submit_button("Calculate Approval Probability")
            st.markdown('</div>', unsafe_allow_html=True)

            if submitted:
                # Validate form
                if (gender == "Select Gender" or 
                    home_ownership == "Select Status" or 
                    defaults == "Select Option" or 
                    education == "Select Level" or 
                    purpose == "Select Purpose" or 
                    not age or not income or not employment or 
                    not credit_score or not credit_history):
                    st.markdown("""
                        <div class="error-message">
                            Please fill in all required fields before submitting.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # Process prediction
                    with st.spinner("Analyzing your application..."):
                        # Create feature dictionary
                        feature_dict = {col: 0 for col in feature_columns}
                        
                        # Set numeric features
                        numeric_features = {
                            'person_age': age,
                            'person_income': income,
                            'person_emp_exp': employment,
                            'loan_amnt': loan_amount,
                            'cb_person_cred_hist_length': credit_history,
                            'credit_score': credit_score
                        }
                        for key, value in numeric_features.items():
                            if key in feature_dict:
                                feature_dict[key] = value
                        
                        # Encode categorical features
                        gender_code = 'F' if gender == 'Female' else 'M'
                        if f'person_gender_{gender_code}' in feature_dict:
                            feature_dict[f'person_gender_{gender_code}'] = 1
                        if f'person_education_{education}' in feature_dict:
                            feature_dict[f'person_education_{education}'] = 1
                        if f'person_home_ownership_{home_ownership.upper()}' in feature_dict:
                            feature_dict[f'person_home_ownership_{home_ownership.upper()}'] = 1
                        if f'loan_intent_{purpose.replace(" ", "")}' in feature_dict:
                            feature_dict[f'loan_intent_{purpose.replace(" ", "")}'] = 1
                        if f'previous_loan_defaults_on_file_{defaults}' in feature_dict:
                            feature_dict[f'previous_loan_defaults_on_file_{defaults}'] = 1

                        # Make prediction
                        features = np.array([feature_dict[col] for col in feature_columns]).reshape(1, -1)
                        prediction = clf.predict(features)
                        probabilities = clf.predict_proba(features)
                        predicted_class = int(prediction[0])
                        probability = float(probabilities[0][predicted_class])

                        # Display results
                        st.markdown("""
                            <div class="category-header">
                                <span class="icon">🎯</span>
                                <span class="text">Prediction Results</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(probability)
                        
                        if predicted_class == 1:
                            st.success(f"Approval Likely ({probability:.1%} confidence)")
                        else:
                            st.error(f"Approval Less Likely ({probability:.1%} confidence)")

    # About tab with corrected structure
    with tab2:
        st.subheader("About This Model")

        st.markdown("### Overview")
        st.write("This loan prediction model demonstrates how machine learning can be applied "
                "to loan approval processes. It uses a Random Forest algorithm trained on "
                "historical loan data.")

        st.markdown("### Model Information")
        st.write("- Based on historical loan approval patterns")
        st.write("- Uses Random Forest Classification")
        st.write("- Considers multiple financial and personal factors")
        st.write("- Trained on comprehensive loan data")

        st.markdown("### Important Notice")
        st.write("This is a demonstration tool only. Do not use it for actual loan decisions. "
                "Consult with financial professionals for real loan advice.")

        st.markdown("### Privacy Notice")
        st.write("All calculations are performed locally in your browser. No data is stored or transmitted.")

        st.markdown("### Technical Details")
        st.write("- Algorithm: Random Forest Classifier")
        st.write("- Training Data Size: ~30,000 records")
        st.write("- Model Features: Personal and financial indicators")
        st.write("- Prediction Output: Binary classification with confidence score")

        st.markdown("### Limitations")
        st.write("This model has some limitations due to the nature of the dataset and its design:")
        st.write("- **Limited Training Data**: The model was trained on a relatively small dataset, which may not fully represent all loan applicants.")
        st.write("- **Income Influence**: The model’s predictions may be less reliable for applicants with high income, as the training data did not include sufficient representation in this category.")
        st.write("- **Simplified Features**: Not all potential factors influencing loan approval were included, and the model may lack nuance for complex financial situations.")


if __name__ == '__main__':
    main()