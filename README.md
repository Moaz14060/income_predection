# About:
- Creating a machine learning model **KNN** to predict a person's yearly income whether it's **lower/equal to 50K or higher**.
# Steps:
- Preprocessing **adult.csv** by cleaning Na values, encoding categorical values and scaling numerical values.
- Performing **EDA** on the data to gain insights.
- Training a **KNN** model and saving it via **pickle** library with accuracy of **83%**.
# Files:
- **notebook.ipynb:** a notebook representing the **experimants** conducted to get from A to B.
- **model_preds.py:** the streamlit app to make the predection process seamless and easy for the user.
- **models folder:** the folder for the **saved models**, right now we have the KNN model only.
- **encoders folder:** the folder for the saved encoders like **one hot**, **ordinal** and **min max scaler**. So we can use them in the app.
- **utils.py:** a helper python file for visualzations, outlier detection and missing values display.
- **requirements.txt:** a file to get the nessecery **liberaries** into the streamlit app.
# How to Run:
- **First Method:** you can go here to see the app right away https://incomepredection-yyl96edv23abvryefsappbv.streamlit.app/
- **Second Method:** download the repository and type
```bash
# Install dependencies
pip install -r requirements.txt  

# Run the app (replace with your path (your directory/model_preds.py) if not in the same directory)
streamlit run model_preds.py
