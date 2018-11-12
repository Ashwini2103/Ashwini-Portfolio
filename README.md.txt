Airbnb's NewUser Booking Kaggle Dataset
	This dataset https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings was part of Airbnb's Kaggle Competition.

Data:-
	The data for Airbnb's dataset can be downloaded here:https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data

Description:-
	1. The main aim of dataset is to predict the destination country of new users. There are 12 possible destination countries
	   US, FR, CA, GB, ES, IT, PT, NL, DE, AU, NDF (no destination found) and other. Here, NDF means the user hasn't done the booking and 
	   other means user booked has booked country not mentioned in destination countries.Also, we make use of sessions dataset along with
	   train and test dataset. 
	2. The data in train dataset is upto 1/1/2014 and that in test starts from 7/1/2014.

Methodology:-
	1. The dataset contains lot of null values so replacing null values by grouping features in dataset.
	2. Feature Enginnering performed on features like device_type,secs_elapsed,timestamp_first_active,date_account_created,date_first_booking,
	   signup_flow.
	3. Usage of LDA(Linear Discrimiant Analysis) which helps in maximizing the seggration of categories.
	4. Used K-Fold cross-validation technique for validating data.
	4. Applied Logisitic Regression, Decision Tree and Random Forest Algorithm as part of model building.

Requirements:-
	Inorder to execute code we need Python packages such as :-
	1.Pandas
	2.Scikit-Learn

