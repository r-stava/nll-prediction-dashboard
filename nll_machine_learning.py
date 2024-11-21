# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima
from prophet import Prophet

# Load the Excel files in your Streamlit app
vacancy_rates_df = pd.read_excel("vacancy rates.xlsx")
rental_rates_df = pd.read_excel("rental rates.xlsx")
home_value_rates_df = pd.read_excel("home value rates.xlsx")
states_df = pd.read_excel("states.xlsx")

# User Input: Select County
selected_county = st.text_input("Enter the County Name for Prediction (e.g., 'Los Angeles County')", "")

# Check if user has entered a county name
if selected_county:
    # Preprocess the data
    # Merge the states dataframe with rental rates and home value rates
    rental_rates_df = rental_rates_df.merge(states_df, left_on='State Abbr', right_on='State Abbr', how='left')
    home_value_rates_df = home_value_rates_df.merge(states_df, left_on='State Abbr', right_on='State Abbr', how='left')

    # Filter data for the selected county and get the corresponding state
    filtered_rental_rates = rental_rates_df[rental_rates_df['County'] == selected_county]
    filtered_home_value_rates = home_value_rates_df[home_value_rates_df['County'] == selected_county]

    # Ensure that the county exists in the data
    if not filtered_rental_rates.empty and not filtered_home_value_rates.empty:
        state_for_vacancy = filtered_rental_rates['State Name'].values[0]

        # Filter vacancy rates for the corresponding state
        filtered_vacancy_rates = vacancy_rates_df[vacancy_rates_df['state'] == state_for_vacancy]

        # Step 1: Vacancy Rate Prediction using SARIMA

        # Prepare Vacancy Rate time series data
        vacancy_time_series = filtered_vacancy_rates.drop(columns=['state']).T
        vacancy_time_series.columns = ['Vacancy Rate']
        vacancy_time_series.index = pd.to_datetime(vacancy_time_series.index, format='%Y')
        vacancy_time_series.dropna(inplace=True)

        # Split the data for training and testing
        train_data = vacancy_time_series['Vacancy Rate'][:len(vacancy_time_series)-5]
        test_data = vacancy_time_series['Vacancy Rate'][-5:]

        # Fit SARIMA model using auto_arima
        sarima_model = auto_arima(train_data, seasonal=True, m=1, trace=True, stepwise=True, suppress_warnings=True)
        sarima_forecast = sarima_model.predict(n_periods=len(test_data))

        # Forecast for 3, 5, and 10 years
        forecast_years = [3, 5, 10]
        forecasts = {}
        for years in forecast_years:
            forecast = sarima_model.predict(n_periods=years)
            forecasts[years] = forecast[-1]  # Get the last prediction in the forecast range

        # Step 2: Rental Rate Prediction using Prophet

        # Prepare Rental Rates data for Prophet
        rental_time_series = filtered_rental_rates.set_index('County').drop(columns=['State Abbr', 'State Name', 'StateCodeFIPS', 'MunicipalCodeFIPS']).T
        rental_time_series.columns = ['rental_rate']
        rental_time_series.index = pd.to_datetime(rental_time_series.index)
        rental_data = rental_time_series.reset_index()
        rental_data.columns = ['ds', 'y']

        # Fit Prophet model for Rental Rates
        rental_model = Prophet()
        rental_model.fit(rental_data)
        future_dates_rental = rental_model.make_future_dataframe(periods=10 * 12, freq='M')
        rental_forecast = rental_model.predict(future_dates_rental)

        # Step 3: Home Value Rate Prediction using Prophet

        # Prepare Home Value Rates data for Prophet
        home_value_time_series = filtered_home_value_rates.set_index('County').drop(columns=['State Abbr', 'State Name', 'StateCodeFIPS', 'MunicipalCodeFIPS']).T
        home_value_time_series.columns = ['home_value_rate']
        home_value_time_series.index = pd.to_datetime(home_value_time_series.index)
        home_value_data = home_value_time_series.reset_index()
        home_value_data.columns = ['ds', 'y']

        # Fit Prophet model for Home Value Rates
        home_value_model = Prophet()
        home_value_model.fit(home_value_data)
        future_dates_home_value = home_value_model.make_future_dataframe(periods=10 * 12, freq='M')
        home_value_forecast = home_value_model.predict(future_dates_home_value)

        # Step 4: Display Predictions and Visualizations

        # Display Vacancy Rate Prediction
        st.subheader(f"Vacancy Rate Prediction for {state_for_vacancy}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(vacancy_time_series.index, vacancy_time_series['Vacancy Rate'], label='Historical Data', marker='o')
        ax.plot(np.arange(len(vacancy_time_series.index), len(vacancy_time_series.index) + len(forecast_years)),
                forecasts.values(), label='Predicted Vacancy Rate', linestyle='--')
        ax.set_title(f"Vacancy Rate Trend for {state_for_vacancy}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Vacancy Rate")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Print Predictions
        st.write(f"Vacancy Rate Prediction for 3 years: {forecasts[3]:.2f}")
        st.write(f"Vacancy Rate Prediction for 5 years: {forecasts[5]:.2f}")
        st.write(f"Vacancy Rate Prediction for 10 years: {forecasts[10]:.2f}")

        # Display Rental Rate Prediction
        st.subheader(f"Rental Rate Prediction for {selected_county}")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(rental_forecast['ds'], rental_forecast['yhat'], label='Forecasted', color='blue')
        ax2.set_title(f'Rental Rate Forecast for {selected_county}')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Rental Rate')
        ax2.grid(True)
        st.pyplot(fig2)

        # Display Home Value Rate Prediction
        st.subheader(f"Home Value Rate Prediction for {selected_county}")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(home_value_forecast['ds'], home_value_forecast['yhat'], label='Forecasted', color='green')
        ax3.set_title(f'Home Value Rate Forecast for {selected_county}')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Home Value Rate')
        ax3.grid(True)
        st.pyplot(fig3)

        # Rental and Home Value Predictions for 3, 5, and 10 years
        def get_predictions(forecast, years):
            future_years = [f"{years} years"]
            forecast_years = forecast[forecast['ds'] >= pd.Timestamp(f"{2024 + years}-01-01")].head(1)
            predictions = forecast_years[['ds', 'yhat']]
            return predictions

        # Display 3, 5, and 10 year predictions for Rental and Home Value Rates
        rental_rate_3_years = get_predictions(rental_forecast, 3)
        rental_rate_5_years = get_predictions(rental_forecast, 5)
        rental_rate_10_years = get_predictions(rental_forecast, 10)
        home_value_rate_3_years = get_predictions(home_value_forecast, 3)
        home_value_rate_5_years = get_predictions(home_value_forecast, 5)
        home_value_rate_10_years = get_predictions(home_value_forecast, 10)

        st.write("**Rental Rate Predictions (3, 5, and 10 years):**")
        st.write(rental_rate_3_years)
        st.write(rental_rate_5_years)
        st.write(rental_rate_10_years)

        st.write("**Home Value Rate Predictions (3, 5, and 10 years):**")
        st.write(home_value_rate_3_years)
        st.write(home_value_rate_5_years)
        st.write(home_value_rate_10_years)
        

    else:
        st.error(f"County '{selected_county}' not found in the data.")
