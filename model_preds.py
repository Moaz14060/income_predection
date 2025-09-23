import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


# Customizing the page
st.set_page_config(page_title="Predictions", page_icon=":material/online_prediction:")

# Main Title
st.markdown("<h1 style='text-align: center; color: black;'>Welcome</h1>", unsafe_allow_html=True)

st.header("In This App:")
# Some Explanations
st.write("**First**, we are going to preprocess data about adults' demografic information.")
st.write("**Second**, train a **KNN model** on it to predict wheather their income is **" \
"lower/equal to 50K or higher** in a year, based on some features. The trained **KNN model** had accuracy of **83%**.")
st.write("**Third**, we are going to use the model to predict the **income**.")
st.divider()

st.header("Predeciting Outcomes:")

# Slider for picking the age value
age = st.slider("Select your age: ", 16, 90, 30)

# Slider for picking the number of educationl years value
educational_num = st.slider("Select years of education: ", 1, 16, 12)

# Slider for picking the the profit from investments value
capital_gain = st.slider("How much did you profit from investments $: ", 0, 99999, 0)

# Slider for picking the losses from investments value
capital_loss = st.slider("How much did you loss from investments $: ", 0, 4500, 0)

# Slider for picking the hours of work per week
hours_per_week = st.slider("How many do your work per week: ", 1, 99, 25)

# Select box for workclass feature
workclass_choice = ("Private", "Self-emp-not-inc", "Local-gov",  "State-gov", "Self-emp-inc", 
                    "Federal-gov", "Without-pay", "Never-worked", "rather-not-say")
worklclass = st.selectbox("Select the type of the employer: ", workclass_choice)
if worklclass == "rather-not-say":
    worklclass = "unknown"

# Select box for education feature
education_choice = ("HS-grad", "Some-college", "Bachelors", "Masters", "Assoc-voc", "11th", "Assoc-acdm", 
                    "10th", "7th-8th", "Prof-school", "9th", "12th", "Doctorate", "5th-6th", "1st-4th", "Preschool")
education = st.selectbox("Select your highest level of education: ", education_choice)

# Select box for marital status feature
marital_status_choice = ("Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", 
                         "Married-spouse-absent", "Married-AF-spouse")
marital_status = st.selectbox("What is your marital status: ", marital_status_choice)

# Select box for occupation feature
occupation_choice = ("Craft-repair", "Prof-specialty", "Exec-managerial", "Adm-clerical", "Sales", "Other-service", "Machine-op-inspct", "Transport-moving", 
                     "Handlers-cleaners", "Farming-fishing", "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces", "rather-not-say")
occupation = st.selectbox("Select your job type: ", occupation_choice)
if occupation == "rather-not-say":
    occupation = "unknown"

# Select box for relationship feature
relationship_choice = ("Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative")
relationship = st.selectbox("What is yout relationship to your family: ", relationship_choice)

# Select box for race feature
race_choice = ("White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other")
race = st.selectbox("What is your race: ", race_choice)

# Select box for gender feature
gender = st.selectbox("Select your gender: ", ("Male", "Female"))

# A value for predection
value = {
    "age": age, 
    "workclass": worklclass, 
    "education": education, 
    "educational-num": educational_num, 
    "marital-status": marital_status, 
    "occupation": occupation,   
    "relationship": relationship, 
    "race": race, 
    "gender": gender, 
    "capital-gain": capital_gain, 
    "capital-loss": capital_loss, 
    "hours-per-week": hours_per_week, 
}

@st.cache_data
def preprocess(value: dict, _one_hot: OneHotEncoder, _min_max_scaler: MinMaxScaler, _ordinal_encoder: OrdinalEncoder) -> pd.DataFrame:
    """
        The function transforms the entered value into a data frame then passes it thruogh ordinal_encoding, one_hot_encoding
        and min_max_scaling and returns a preprocessed data frame

        Parameters:
            1. value: the values of the features we want to predict (dict)
            2. one_hot: an object of the OneHotEncoder (OneHotEncoder) 
            3. min_max_scaler: an object of the MinMaxScaler (MinMaxScaler)
            4. ordinal_encoder: an object of the OrdinalEncoder (OrdinalEncoder)

        Return Value:
            the preprocessed values as a data frame
    """
    col_for_one_hot = ["education", "relationship", "workclass", "occupation", "marital-status", "race"]
    numer_col = ["age", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
    # Transfrom the dict to a data frame
    value_df = pd.DataFrame([value])

    # 1. One hot encoding
    encoded = one_hot.transform(value_df[col_for_one_hot])
    # Transform into a data frame
    encoded_df = pd.DataFrame(encoded, columns = one_hot.get_feature_names_out(col_for_one_hot), index = value_df.index)
    # Combining the encoded columns with the original df while dropping them before encoding them
    final_value = pd.concat([encoded_df, value_df.drop(columns = col_for_one_hot)], axis = 1)

    # 2. Label encoding
    final_value["gender"] = ordinal_encoder.transform(final_value[["gender"]])

    # 3. Min max scaling
    final_value[numer_col] = min_max_scaler.transform(final_value[numer_col])

    return final_value

# Load Encoders
with open("encoders/one_hot.pkl", "rb") as f:
    one_hot = pickle.load(f)

with open("encoders/scaler.pkl", "rb") as f:
    min_max_scaler = pickle.load(f)

with open("encoders/ordinal.pkl", "rb") as f:
    ordinal_encoder = pickle.load(f)

value_to_pred = preprocess(value, one_hot, min_max_scaler, ordinal_encoder)

# Load KNN Model
with open("models/knn_model.sav", "rb") as f:
    knn_model = pickle.load(f)

# A method to center the button
col1, col2, col3, col4, col5 = st.columns(5)
with col3:
    pred = st.button("Predict", type= "primary", icon = ":material/online_prediction:")
if pred:
    st.success(f"Yearly Income: **{knn_model.predict(value_to_pred)[0]}**")

    
