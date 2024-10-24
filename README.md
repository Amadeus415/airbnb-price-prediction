# Airbnb Price Estimator

This project contains an Airbnb price estimator for Marin County, California. It includes a data analysis notebook and a Streamlit web application for predicting Airbnb listing prices.


## Application Preview

Here are some screenshots of the Airbnb Price Estimator application:

### Price Estimator
![Price Estimator](./images/estimator.png)

### Airbnb Listings Map
![Airbnb Listings Map](./images/map.png)

These images showcase the main features of our Streamlit application: the price estimation tool and the interactive map of Airbnb listings in Marin County.



## Files in the Codebase

1. `airbnb_streamlit.py`: The main Streamlit application file that creates an interactive web interface for the Airbnb price estimator.

2. `airbnb_csv.ipynb`: A Jupyter notebook containing data analysis, preprocessing, and model training for the Airbnb dataset.

3. `datasets/airbnb_cleaned.csv`: The cleaned and preprocessed dataset used by the Streamlit application.

4. `datasets/dataset_airbnb.csv`: The original dataset used for analysis in the Jupyter notebook.

5. `images/`: A folder containing screenshots of the application for documentation purposes.

6. `requirements.txt`: A file listing all the Python dependencies required to run the Streamlit application.

7. `.gitignore`: A configuration file specifying which files and directories should be ignored by Git version control.



## Running the Streamlit Application

To run the Streamlit application, follow these steps:

1. Ensure you have Python installed on your system.

2. Create and activate a Virtual Environment

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Navigate to the project directory in your terminal.

5. Run the Streamlit app with the following command:
   ```
   streamlit run airbnb_streamlit.py
   ```

6. The application will start, and you should see a URL in your terminal. Open this URL in your web browser to interact with the Airbnb Price Estimator. Note: you may need to reload the page for the estimator to load in.

## Models Used in airbnb_csv.ipynb

The Jupyter notebook `airbnb_csv.ipynb` explores and compares several machine learning models for predicting Airbnb prices:

1. **Linear Regression**: A simple model that assumes a linear relationship between the features and the target variable (price).

2. **Random Forest**: An ensemble learning method that constructs multiple decision trees and merges them to get a more accurate and stable prediction.

3. **Gradient Boosting**: Another ensemble technique that builds trees sequentially, with each tree correcting the errors of the previous ones.

4. **Support Vector Regression (SVR)**: A model that tries to find the hyperplane that best fits the data in a high-dimensional space.

The notebook compares these models using metrics such as Mean Squared Error (MSE) and R-squared (R²) to evaluate their performance on the Airbnb dataset.

## Note

The Streamlit application uses a Linear Regression model for price prediction, which was found to perform well in the analysis conducted in the Jupyter notebook.