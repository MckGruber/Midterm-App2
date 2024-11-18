import streamlit as st
import pandas as pd
import pickle
import numpy as np

ADA_BOOST = "AdaBoostClassifier"
DT = "DecisionTreeClassifier"
RF = "RandomForestClassifier"
VC = "VotingClassifier"
XG = "XGBRegressor"
# MODELS = [ADA_BOOST, DT, RF, VC, XG]
MODELS = [XG]


def get_models() -> dict:
  model_dict = {}
  for model in MODELS:
    with open(f"{model}.pickle", "rb") as f:
      model_dict[model] = pickle.load(f)
  return model_dict

def highlight_prediction(s: pd.Series, props=""):
  return np.where(
    s == "Normal", "background-color:lime;color:black", 
    np.where(s == "Suspect", "background-color:yellow", "background-color:orange;color:black"))


def get_example_sheet() -> pd.DataFrame:
  return pd.read_csv("traffic_data_user.csv")


def get_training_data() -> pd.DataFrame:
  return pd.read_csv("Traffic_Volume.csv")

def prepare_sheet(user_frame: pd.DataFrame) -> pd.DataFrame:
  import datetime
  import pytz
  df = user_frame.copy(deep= True)
  df['date_time'] = df['date_time'].map(
    lambda x: datetime.datetime.combine(
      date=datetime.date(month=int((x.split()[0]).split("/")[0]), 
                        day=int((x.split()[0]).split("/")[1]), 
                        year=int("20" + (x.split()[0]).split("/")[2])), 
      time=datetime.time(int(x.split()[1].split(":")[0]), int(x.split()[1].split(":")[0]), 0, 0), 
      tzinfo=pytz.timezone('US/Central')
    )
  )
  df['day_of_week'] = df['date_time'].map(
    lambda x: x.strftime("%A")
  )
  df['hour'] = df['date_time'].map(
    lambda x: int(x.strftime("%H"))
  )
  df['month'] = df['date_time'].map(
    lambda x: x.strftime("%B")
  )
  df.drop(columns=['date_time', 'traffic_volume'], inplace=True)
  return df


st.write("# Traffic volume Predictor")
st.image("traffic_image.gif")
useable_models = get_models() 
example_data = get_example_sheet()
training_data = get_training_data()
alpha = st.slider("Prediction Alpha", value=0.1)
with st.sidebar:
  st.image("traffic_sidebar.jpg")
  st.write("You can either upload your dat or manually enter input features")
  model = st.selectbox("Select Model to use", useable_models.keys()) if len(useable_models) != 0 else useable_models[list(useable_models.keys())[0]]
  with st.expander("Option 1: Upload a CSV File"):
    st.write("Example DataFrame: ")
    st.write(example_data.head())
    csv_file = st.file_uploader("click to upload csv file", type="csv", accept_multiple_files=False)
  with st.expander("Option 2: Fill Out Form"):
    with st.form("user_imputs_form"):
      st.header("Enter the traffic details manually using the form bellow")
      holiday = st.selectbox("Choose whether today is a designated holiday", list(set(training_data['holiday'].tolist())))
      temp = st.number_input("Average tempurature in Kelvin", min(training_data['temp']), max(training_data['temp']))
      rain = st.number_input("Amount in mm of rain that occurred in the hour", 0.0, max(training_data['rain_1h']))
      snow = st.number_input("Amount in mm of snow that occurred in the hour", 0.0 , max(training_data['snow_1h']))
      clouds = st.number_input("Percentage of cloud cover", 1, max(training_data['clouds_all']))
      weather = st.selectbox("Choose the current weather", list(set(training_data['weather_main'].tolist())))
      month = st.selectbox("Choose Month", list(set(prepare_sheet(training_data)['month'].tolist())))
      day_of_week = st.selectbox("Choose day of the week", list(set(prepare_sheet(training_data)['day_of_week'].tolist())))
      hour = st.selectbox("Choose Hour", list(range(0,24)))
      submit_button = st.form_submit_button("Submit Form Data")


if csv_file != None:
  user_df = pd.read_csv(csv_file)
  encoded_df = prepare_sheet(get_training_data())
  encoded_df.loc[len(encoded_df): len(encoded_df) + len(user_df)] = user_df
  encoded_dummy_df = pd.get_dummies(encoded_df)
  user_encoded_df = encoded_dummy_df.tail(len(user_df))
  prediction, intervals = get_models()[model].predict(user_encoded_df, alpha = alpha)
  user_df["Prediction"] = prediction
  user_df['Lower_Limit'] = intervals[:, 0]
  # upper_limit = intervals[:, 1]
  user_df['Upper_Limit'] = intervals[:, 1]
  # user_df.style.apply(highlight_prediction, props="", axis=0, subset=["Prediction"])
  styled = user_df.style.apply(highlight_prediction, props="", axis=0, subset=["Prediction"])
  st.write("# Predicting Travel Volume")
  st.write(f"With a {int((1-alpha)*100)}% confidence interval:")
  st.write(styled)



if submit_button:
  form_output = [holiday, temp, rain, snow, clouds, weather, day_of_week, hour, month]
  default_data = prepare_sheet(training_data.copy(deep=True))
  default_data.loc[len(default_data)] = form_output
  encoded = pd.get_dummies(default_data)
  user_data_encoded = encoded.tail(1)
  prediction, intervals = get_models()[model].predict(user_data_encoded, alpha = alpha)
  pred_value = prediction[0]
  lower_limit = intervals[:, 0]
  upper_limit = intervals[:, 1]
  lower_limit = max(0, lower_limit[0][0])
  upper_limit = min(1, upper_limit[0][0])
  st.write("# Predicting Travel Volume")
  st.metric("Predicted Traffic Volume", int(pred_value))
  st.write(f"With a {int((1-alpha)*100)}% confidence interval:")
  st.write(f"**Confidence Interval**: [{lower_limit* 100:.2f}%, {upper_limit* 100:.2f}%]")



if csv_file != None or submit_button:
  st.subheader("Model Insights")
  tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Feature Importance", 
                              "Histogram of Residuals", 
                              "Predicted Vs. Actual", 
                              "Confusin Matrix",
                              "Classification Report",
                              "Decision Tree"]) 

  with tab1:
      st.write("### Feature Importance")
      st.image(f'{model}_feature_imp.svg')
      st.caption("Relative importance of features in prediction.")
  with tab2:
      st.write("### Histogram of Residuals")
      st.image(f'{model}_residual_plot.svg')
      st.caption("Distribution of residuals to evaluate prediction quality.")
  with tab3:
      st.write("### Plot of Predicted Vs. Actual")
      st.image(f'{model}_pred_vs_actual.svg')
      st.caption("Visual comparison of predicted and actual values.")
  with tab4:
      try:
        st.write("### Confusin Matrix")
        st.image(f'{model}_confusion_mat.svg')
      except:
         st.write(f"{model} does not have a Confusion Matrix")
  with tab5:
      try:
        st.write(pd.read_csv(f'{model}_class_report.csv'))
      except:
        st.write(f"{model} does not have a Classification Report")
  with tab6:
      try:
        st.write("### Decision Tree Visualization")
        st.image(f'{model}_visual.svg')
        st.caption("Visualization of the Decision Tree used in prediction.")
      except:
        st.write(f"{model} Does not have a Decision Tree Visualization")

  