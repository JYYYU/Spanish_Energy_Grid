Accurate forecasting of energy demand is crucial, especially as we shift toward renewable energy in a market increasingly influenced by energy trading. Unlike traditional power sources, renewables like solar and wind are intermittent, meaning they don’t produce energy consistently throughout the day. Without a clear forecast of when and how much energy will be needed, utilities can't plan for periods when these clean sources fall short, forcing them to buy or sell energy at unfavorable prices. In energy trading, where prices fluctuate minute by minute, precise demand forecasts can determine the difference between profit and loss, enabling traders to optimize when to buy or sell electricity, hedge risks, and manage costs effectively. As energy markets become more complex with the addition of carbon credits, battery storage, and decentralized grids, mastering demand forecasting is essential for staying competitive and profitable.

In this project, I walk through how I preprocessed the data, performed feature engineering, and applied three different deep neural network models to forecast electricity demand.

**Data:**

The Data folder contains all the original datasets, including 'energy_dataset' and 'weather_features', along with the cleaned sets and engineered features.

- Hana, N. J. (2019). Energy consumption, generation, prices and weather. Kaggle. https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

**Data_Workflow:**

The Data_Workflow folder contains the three notebooks used for data cleaning and feature engineering. Additionally, the Python file 'DataHelper,' which is utilized in each notebook, holds all the complex code that I removed from the notebooks to present a cleaner and more streamlined view for the audience.

**Models:**

The Models folder contains the notebook where we evaluate the three deep neural network (DNN) models that were applied.
