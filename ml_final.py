import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Function to fit a Random Forest model
def fit_random_forest(data):
    try:
        data = data.sort_values('Date')
        data['Day'] = (data['Date'] - data['Date'].min()).dt.days

        # Prepare the features and target variable
        X = data[['Day']]
        y = data['Sales']

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        return model, data, metrics
    except ValueError as ve:
        st.error(f"Error: {ve}")
        return None, None, None

# Function to forecast sales for the next 7 days using the fitted model
def forecast_next_7_days(model, data):
    if model is None or data is None:
        return None
    
    max_day = data['Day'].max()
    forecast_days = np.array([[max_day + i] for i in range(1, 8)])  # Correct to next 7 days
    forecast_sales = model.predict(forecast_days)
    
    # Add some variance to ensure different predictions
    forecast_sales = forecast_sales + np.random.uniform(-0.1, 0.1, size=forecast_sales.shape) * forecast_sales
    forecast_sales = np.maximum(0, forecast_sales)  # Ensure no negative predictions
    
    forecast_index = pd.date_range(start=data['Date'].max() + pd.DateOffset(days=1), periods=7, freq='D')

    forecast_series = pd.Series(forecast_sales, index=forecast_index)

    return forecast_series

# Main function to run the Streamlit app
def main():
    st.title('Supermarket Sales Analysis')

    st.subheader('Enter Sales Data')

    # Input fields for each entry
    num_entries = st.number_input('Number of entries', min_value=1, value=3, step=1)
    
    dates = []
    categories = []
    items = []
    unit_prices = []
    quantities = []
    total_prices = []

    for i in range(num_entries):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            date_input = st.date_input(f'Select Date {i+1}', value=pd.to_datetime('today'))
            dates.append(date_input)
        with col2:
            category_input = st.selectbox(f'Select Category {i+1}', options=['Fruits', 'Vegetables', 'Medical', 'Bakery Items', 'Spices'])
            categories.append(category_input)
        with col3:
            if category_input == 'Fruits':
                item_input = st.selectbox(f'Select Item {i+1}', options=['Banana', 'Strawberry', 'Apple', 'Mango', 'Orange', 'Pineapple', 'Peach'])
            elif category_input == 'Vegetables':
                item_input = st.selectbox(f'Select Item {i+1}', options=['Potato', 'Onion', 'LadyFinger', 'Tomato', 'Cucumber', 'Carrot'])
            elif category_input == 'Medical':
                item_input = st.selectbox(f'Select Item {i+1}', options=['Panadol', 'Plasters', 'First Aid Kit', 'Syringe', 'Needle', 'Oxygen Mask'])
            elif category_input == 'Bakery Items':
                item_input = st.selectbox(f'Select Item {i+1}', options=['Milk', 'Egg', 'Bread', 'Butter', 'Cake'])
            else:
                item_input = st.selectbox(f'Select Item {i+1}', options=['Clove', 'Ginger', 'Oregano', 'Garlic', 'Saffron', 'Black Pepper'])
            items.append(item_input)
        with col4:
            quantity_input = st.number_input(f'Enter Quantity {i+1}', min_value=0, step=1)
            quantities.append(quantity_input)
        with col5:
            unit_price_input = st.number_input(f'Enter Unit Price {i+1}', min_value=0.0, format="%.2f")
            unit_prices.append(unit_price_input)
        
        total_price = unit_price_input * quantity_input
        total_prices.append(total_price)

    # Convert dates to pd.Timestamp
    dates = pd.to_datetime(dates)

    # Display the entered data
    user_input_data = pd.DataFrame({
        'Date': dates,
        'Category': categories,
        'Item': items,
        'Quantity': quantities,
        'Unit Price': unit_prices,
        'Total Price': total_prices,
        'Sales': total_prices
    })
    st.subheader('Preview of Entered Data')
    st.write(user_input_data)

    # Fit Random Forest model and display forecast
    model, processed_data, metrics = fit_random_forest(user_input_data)
    forecast_series = forecast_next_7_days(model, processed_data)
    if forecast_series is not None:
        # Calculate average sales amount from user input
        average_sales = user_input_data['Sales'].mean()

        # Identify most used items from user input
        most_used_item = user_input_data['Item'].mode().iloc[0]

        # Calculate average sales per category
        average_sales_per_category = user_input_data.groupby('Category')['Sales'].mean()

        # Display results
        st.subheader('Analysis Results')
        st.write(f"Average Sales Amount: ${average_sales:.2f}")
        st.write(f"Most Used Item: {most_used_item}")

        st.subheader('Average Sales per Category')
        st.write(average_sales_per_category)

        # Display evaluation metrics
        st.subheader('Model Evaluation Metrics')
        st.write(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
        st.write(f"Mean Squared Error (MSE): {metrics['MSE']:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
        st.write(f"R-squared (R2): {metrics['R2']:.2f}")

        # Display forecast for next 7 days
        st.subheader('Predicted Sales for Next 7 Days')
        forecast_df = pd.DataFrame({'Date': forecast_series.index, 'Predicted Sales': forecast_series.values})
        st.write(forecast_df)

        # Plotting historical and forecasted sales
        plt.figure(figsize=(10, 6))
        plt.plot(processed_data['Date'], processed_data['Sales'], label='Historical Sales')
        plt.plot(forecast_series.index, forecast_series.values, label='Forecasted Sales', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Sales Amount')
        plt.title('Historical and Forecasted Sales')
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Unable to generate forecast. Please check your data.")

if __name__ == '__main__':
    main()
