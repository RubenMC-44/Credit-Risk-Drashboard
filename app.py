import streamlit as st 
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import classification_report
model = joblib.load('model.pkl')

df = pd.read_csv('clean_data.csv')

st.set_page_config(
    page_title="Credit RisK Drashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)


def page1(): 
    st.title('Data description & overview')
    
    with st.expander("Dataframe"):
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df)
        
    with st.expander("Describe()"):
        st.dataframe(df.describe())

    riskratio=df['SeriousDlqin2yrs'].mean() * 100
    debtratio_mean= df['DebtRatio'].mean() * 100 #Extreme values, so is a must use median instead of mean
    debtratio_median= df['DebtRatio'].median() * 100 

    #Example of prev ratios 
    prev_debt = 39.87
    prev_risk = 7.45

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚠️ Risk Clients")
        st.metric(
            label="",
            value=f"{riskratio:.2f}%",
            delta=f"{riskratio - prev_risk:.2f}%",
            border=True
        )
        st.caption("Higher = worse")
        

    with col2:
        st.markdown("### 📉 Debt Ratio")
        st.metric(
            label="",
            value=f"{debtratio_median:.2f}%",
            delta=f"{debtratio_median - prev_debt:.2f}%",
            delta_color="inverse",
            border=True
        )
        st.caption("Lower is better")


    #Replacement of names, instead of ussing 0 and 1, i am going to use the simple Yes and No 
    risk_counts = df['SeriousDlqin2yrs'].replace({0: 'NO', 1: 'Yes'}).value_counts().reset_index()
    risk_counts.columns = ['Risk', 'Quantity']


    # Filtering the data frame but positive to risk to work with it
    filtered_df = df[(df['SeriousDlqin2yrs'] == 1)]

    #--- We are filtering the quantity person in each range of time.
    risk30= (filtered_df['NumberOfTime30-59DaysPastDueNotWorse'] > 0).sum()
    risk60=(filtered_df['NumberOfTime60-89DaysPastDueNotWorse'] > 0).sum()
    risk90 = (filtered_df['NumberOfTimes90DaysLate'] > 0).sum()

    #After calculate how many risk we have in each time range i made a specific dataframe to show the graphics
    delay_data = pd.DataFrame({
        'Type of delayed': ['30-59 days', '60-89 days', '90+ days'],
        'Clients with Risk': [risk30, risk60, risk90]
    })

    #The collumn is not divide by ane range, so we need to do some data traetment 
    bins = [18, 25, 40, 60, 100]
    labels = ['Young', 'Adult', 'Middle age', 'Older adult']

    filtered_df['Age range'] = pd.cut(filtered_df['age'],bins= bins,labels=labels,include_lowest=True) #Cut the data frame, within the lables

    #we need to create a new filtered data frame to bua ble and create the graphic
    #Divided by labels
    young_risk =(filtered_df['Age range'] == 'Young').sum()
    adult_risk =(filtered_df['Age range'] == 'Adult').sum()
    middle_age_risk =(filtered_df['Age range'] == 'Middle age').sum()
    older_adult_risk =(filtered_df['Age range'] == 'Older adult').sum()

    age_range_df = pd.DataFrame({
        'Age' : ['18-25 years', '25-40 years', '40-60 years', '60-100 years'],
        'Clients with Risk': [young_risk,adult_risk,middle_age_risk,older_adult_risk]
    }
    )
    #Creating a select box to compress the graphics and choose the one you preffer
    #--Quantity of risk(count)- Risk depends of delayed - relation age with risk

    option = st.selectbox('Select yout view 📊 ',['-','Quantity of Risk persons','Risk depends of delayed days', 'Relation age with risk'])

    if option == 'Quantity of Risk persons':
        st.subheader('📊 Quantity of risk default')

        fig = px.bar(
            risk_counts,
            x='Risk',
            y='Quantity',
            color='Risk',
            color_discrete_map={'No': '#00cc96', 'Yes': '#ef553b'},
            text='Quantity'
        )

        fig.update_layout(
            template='plotly_dark',
            title=None,
            xaxis_title='Risk of default',
            yaxis_title='Quantity',
            showlegend=False
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(fig, use_container_width=True)
        
    elif option == 'Risk depends of delayed days':
        st.subheader('⚠️ Risk by delay category')

        fig = px.bar(
            delay_data,
            x='Type of delayed',
            y='Clients with Risk',
            color='Clients with Risk',
            color_continuous_scale='Reds',
            text='Clients with Risk'
        )

        fig.update_layout(
            template='plotly_dark',
            xaxis_title='Delay category',
            yaxis_title='Clients with risk'
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(fig, use_container_width=True)
        
    elif option == 'Relation age with risk':
        st.subheader('📈 Risk by age')

        fig = px.line(
            age_range_df,
            x='Age',
            y='Clients with Risk',
            markers=True
        )

        fig.update_layout(
            template='plotly_dark',
            xaxis_title='Age',
            yaxis_title='Clients with risk'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Model Metrics")
    metrics = joblib.load('metrics_simple.pkl')

    col1, col2, col3 = st.columns(3)

    col1.metric("Precision", f"{metrics['precision']:.2f}")
    col2.metric("Recall", f"{metrics['recall']:.2f}")
    col3.metric("F1 Score", f"{metrics['f1']:.2f}")


def page2(): 
    st.subheader("🤖 Artificial intelligence - Model predictions 🚀")
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    with st.form("prediction form"):
    
        col_pred_1, col_pred_2, col_pred_3 = st.columns([1,1,1])
    
        with col_pred_1:
            age_input = st.number_input(
                    "Age",
                    min_value=0,
                    max_value=100,
                    )
                    
            number_dependents_input = st.number_input(
                    "Number Of Dependents",
                    min_value = 0,
                    max_value = 10,
                    )
                    
        with col_pred_2:
            monthly_income_input = st.number_input(
                    "Monthly Income,",
                    min_value = 0.0,
                    max_value = 100000.0,
                    )
                    
            debtratio_input = st.number_input(
                    "Debt Ratio",
                    min_value = 0.0,
                    max_value = 100000.0,
                    )
            Unsecured_lines_input = st.number_input(
                    "Unsecured credit utilization",
                    min_value = 0.0,
                    max_value = 1.0,
                    )
        
        with col_pred_3:        
            days30_input = st.number_input(
                    'Times 30-59 days late',
                    min_value=0, max_value=20, 
                    value=0, step=1)
                    
            days60_input = st.number_input(
                    'Times 60-90 days late',
                    min_value=0, max_value=20, 
                    value=0, step=1)
                    
            days90_input = st.number_input(
                    'Times > 90 days late',
                    min_value=0, max_value=20, 
                    value=0, step=1)
            open_credits_input = st.number_input(
                    'Credits Open at the time',
                    min_value=0, max_value=20, 
                    value=0)
            real_state_credit_input = st.number_input(
                    'Real state credits Open at the time',
                    min_value=0, max_value=20, 
                    value=0)
            
        submit_prediction = st.form_submit_button("Submit prediction")
        
        if submit_prediction:
                
            input_vector = [[
                    Unsecured_lines_input,
                    age_input,
                    days30_input,
                    debtratio_input,
                    monthly_income_input,
                    open_credits_input,
                    days90_input,
                    real_state_credit_input,
                    days60_input,
                    number_dependents_input,
                    ]]
            input_scaled = scaler.transform(input_vector)
            y_pred = model.predict_proba(input_scaled)

            col_res_1 ,col_res_2 = st.columns([3,1])

            with col_res_1:
                st.progress(y_pred[0][1])
            with col_res_2:
                if y_pred[0][1] >= 0.35 and y_pred[0][1] <= 0.5:
                    st.error("⚠️ The client is ON Alert Necessary to check")
                elif y_pred[0][1] > 0.5:
                    st.error("⚠️ The client is ON RISK!")
                else:
                    st.success("✅ The client has NO RISK!")


st.title("Credit Drashboard")    
    
pg = st.navigation(
        {"D&A":
            [
                st.Page(page1, title="Data description", icon="🏦"),
            ], 
        "AI":
            [
                st.Page(page2, title = "Artificial intelligence - Predictions", icon = "🤖"),
            ]
        }
    )
pg.run()