
<body>
    <h1>Car Price Prediction Model</h1>
    <p>This project aims to predict the selling price of cars using various machine learning models. The dataset includes various car attributes such as year, selling price, present price, kilometers driven, fuel type, seller type, transmission, and owner type. The machine learning models used include Linear Regression and Lasso Regression to predict car prices based on the features.</p>
    <h2>Dependencies</h2>
    <p>The following Python libraries are required to run the code:</p>
    <ul>
        <li><code>numpy</code></li>
        <li><code>pandas</code></li>
        <li><code>matplotlib</code></li>
        <li><code>seaborn</code></li>
        <li><code>sklearn</code></li>
    </ul>
    <p>You can install them using pip:</p>
    <pre><code>pip install numpy pandas matplotlib seaborn scikit-learn</code></pre>
    <h2>Dataset</h2>
    <p>The dataset used in this project is called "car data.csv", and it contains the following columns:</p>
    <ul>
        <li><strong>Car_Name</strong>: Name of the car</li>
        <li><strong>Year</strong>: Year of manufacture</li>
        <li><strong>Selling_Price</strong>: Selling price of the car (target variable)</li>
        <li><strong>Present_Price</strong>: Current price of the car</li>
        <li><strong>Kms_Driven</strong>: Total kilometers driven</li>
        <li><strong>Fuel_Type</strong>: Type of fuel (Petrol, Diesel, CNG)</li>
        <li><strong>Seller_Type</strong>: Type of seller (Dealer, Individual)</li>
        <li><strong>Transmission</strong>: Type of transmission (Manual, Automatic)</li>
        <li><strong>Owner</strong>: Number of previous owners</li>
    </ul>
    <h2>Data Processing</h2>
    <h3>1. Loading the Data</h3>
    <p>The dataset is loaded from a CSV file using pandas.</p>
    <pre><code>car_dataset = pd.read_csv("car data.csv")</code></pre>
    <h3>2. Data Inspection</h3>
    <p>The dataset is checked for missing values, data types, and basic statistics.</p>
    <pre><code>
    car_dataset.info()
    car_dataset.isnull().sum()
    </code></pre>
    <h3>3. Categorical Encoding</h3>
    <p>Categorical columns (<code>Fuel_Type</code>, <code>Seller_Type</code>, <code>Transmission</code>) are encoded into numerical values.</p>
    <pre><code>
    car_dataset.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)
    car_dataset.replace({'Seller_Type':{'Dealer':0, 'Individual':1}}, inplace=True)
    car_dataset.replace({'Transmission':{'Manual':0, 'Automatic':1}}, inplace=True)
    </code></pre>
    <h3>4. Feature Selection</h3>
    <p>The dataset is split into features (<code>X</code>) and target variable (<code>Y</code>).</p>
    <pre><code>
    X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
    Y = car_dataset['Selling_Price']
    </code></pre>
    <h2>Model Training</h2>
    <h3>Linear Regression</h3>
    <p>The Linear Regression model is trained using the training dataset and evaluated using the R-squared error metric.</p>
    <pre><code>
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, Y_train)
    </code></pre>
    <ul>
        <li><strong>Training R-squared Error</strong>: 0.8799</li>
        <li><strong>Test R-squared Error</strong>: 0.8366</li>
    </ul>
    <p>Visualizations of actual vs. predicted prices are generated for both training and testing datasets.</p>
    <h3>Lasso Regression</h3>
    <p>The Lasso Regression model is also trained and evaluated using similar steps as Linear Regression.</p>
    <pre><code>
    lass_reg_model = Lasso()
    lass_reg_model.fit(X_train, Y_train)
    </code></pre>
    <ul>
        <li><strong>Training R-squared Error</strong>: 0.8428</li>
        <li><strong>Test R-squared Error</strong>: 0.8709</li>
    </ul>
    <p>Visualizations of actual vs. predicted prices are generated for both training and testing datasets.</p>
    <h2>Model Evaluation</h2>
    <p>The models are evaluated using the R-squared error score, which measures how well the models explain the variance in the target variable (selling price).</p>
    <ul>
        <li><strong>Linear Regression</strong>: 
            <ul>
                <li>R-squared error for training: 0.8799</li>
                <li>R-squared error for testing: 0.8366</li>
            </ul>
        </li>
        <li><strong>Lasso Regression</strong>: 
            <ul>
                <li>R-squared error for training: 0.8428</li>
                <li>R-squared error for testing: 0.8709</li>
            </ul>
        </li>
    </ul>
    <h2>Visualizations</h2>
    <p>Scatter plots are used to compare actual prices vs. predicted prices for both training and testing datasets.</p>
    <pre><code>
    plt.scatter(Y_train, training_data_prediction)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual Prices vs Predicted Prices")
    plt.show()
    plt.scatter(Y_test, test_data_prediction)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual Prices vs Predicted Prices")
    plt.show()
    </code></pre>
    <h2>Conclusion</h2>
    <p>The models successfully predict car selling prices, with Lasso Regression performing slightly better on the test data compared to Linear Regression. Further improvements can be made by tuning the models or trying additional algorithms.</p>
    <h2>Files</h2>
    <ul>
        <li><strong>car data.csv</strong>: The dataset used for the prediction model.</li>
        <li><strong>car_price_prediction_model.ipynb</strong>: Jupyter notebook containing the implementation.</li>
    </ul>

</body>
</html>
