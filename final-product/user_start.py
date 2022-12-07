import pandas as pd
import numpy as np
import sklearn as skl
import pickle
from sklearn.metrics import f1_score, recall_score, precision_score

#ignore warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

flight_registry_file = "data/airport_registry.parquet"
# flight_data_file = "data/testData_2022"
flight_data_file = "data/testDataBinaryTarget_2022"
# flight_data_file = "data/trainData_2018-2019"
airport_translation_file = "data/translation_Origin"
airline_translation_file = "data/translation_Airline"

# model_file = "models/final_models/model_logistic_regression.pkl"
model_file = "models/final_models/binary_model_naive_bayes.pkl"

test=False

# target = 'BinArrDelayMinutes'
target = 'BinaryArrDelayMinutes'

#load flight registry
flight_registry = pd.read_parquet(flight_registry_file, engine="fastparquet")

#load preprocessed 2022 dataset (only good from Dec 31, 2021 to Jul 30, 2022)
flight_data = pd.read_parquet(flight_data_file, engine="fastparquet")
airport_translation_data = pd.read_parquet(airport_translation_file, engine="fastparquet")
airline_translation_data = pd.read_parquet(airline_translation_file, engine="fastparquet")

#load model
model = pickle.load(open(model_file, 'rb'))

# columns to predict on
selectedFeatures = model.feature_names_in_

#### FUNCTIONS

## GetRelevantAirports
#retrieves relevant airport from flight registry with origin and destination provided
#possible inputs are airport names such as "SEA" or city name such as "Seattle"

# the user has provided the departure city or airport, arrival city or airport, and date
# now we find all relevant airports (date is not needed)
def GetRelevantAirports(departure, arrival, data):
    #translate departure and arrival input to markets
    #check if airport name contains input
    departure_search = data[data['AirportName']==departure]
    if not len(departure_search):
        #not found, search city name instead
        departure_search = data[data['CityName'].str.contains(departure)]
        if not len(departure_search):
            #error return nothing
            return None
    #get the market from the departure search result
    departure_market = departure_search.iloc[0]['CityMarketID']

    #do the same for arrival
    arrival_search = data[data['AirportName']==arrival]
    if not len(arrival_search):
        arrival_search = data[data['CityName'].str.contains(arrival)]
        if not len(arrival_search):
            #error return nothing
            return None
    arrival_market = arrival_search.iloc[0]['CityMarketID']

    #filter the data for our airports
    departures = data[data['CityMarketID']==departure_market]
    arrivals = data[data['CityMarketID']==arrival_market]

    #we only care about airport name
    filtered_data={
        "origins": departures['AirportName'],
        "dests": arrivals['AirportName']
    }
    return filtered_data

## EncodeAirport
# given translation data and list of airports 
def EncodeAirport(translation, airport_list):
    origin = translation[translation['Label'].isin(airport_list['origins'])]
    dest = translation[translation['Label'].isin(airport_list['dests'])]
    
    translations = {
        'origins': origin['Translation'],
        'dests': dest['Translation'],
    }
    return translations

## GetFlightsFromData
#gets the rest of the data for relevant flights
#returns filtered dataset or empty dataframe
#note that if you end up with an empty dataframe, you should ask the user to try another date
#date is an array where index 0 is month and index 1 is day of the month
def GetFlightsFromData(dataset, relevant_airports, date):
    #get flights in dataset given relevant airports and date
    #filter one at a time to avoid exception
    flights = dataset[dataset['Origin'].isin(relevant_airports['origins'])]
    flights = flights[flights['Dest'].isin(relevant_airports['dests'])]
    flights = flights[flights['Month'] == date[0]]
    flights = flights[flights['DayofMonth'] == date[1]]
    return flights
    
## PredictBestFlight
#runs model on relevant flights
#find best predict delay
#return flights with best delay
#date is an array where index 0 is month and index 1 is day of the month
def PredictBestFlight(model, features, flights):
    #run prediction
    X = flights[features]
    predictions = model.predict(X)

    predicted_data = flights
    predicted_data['PredArrDelayMinutes'] = predictions
    
    #get recommended flights   
    recommended_flights = predicted_data[predicted_data['PredArrDelayMinutes'] == predicted_data['PredArrDelayMinutes'].min()]

    #include actual delay in the return for analysis
    Y=flights[target]

    return {
        'recommended_flights': recommended_flights,
        'predicted_data': predicted_data,
        'predictions': predictions, 
        'actuals': Y
    }

## EncodeAirport
# given translation data and dataset of airline, origin, dest
def DecodeToAirlineAndAirports(airline_translation, airport_translation, data):
    airlines = list()
    origins = list()
    dests = list()
    for index, row in data.iterrows():
        airline = airline_translation[airline_translation['Translation']==row['Airline']].iloc[0]['Label']
        origin = airport_translation[airport_translation['Translation']==row['Origin']].iloc[0]['Label']
        dest = airport_translation[airport_translation['Translation']==row['Dest']].iloc[0]['Label']
        airlines.append(airline)
        origins.append(origin)
        dests.append(dest)

    translated_data = pd.DataFrame.from_dict({
        'Airline': airlines,
        'Origin': origins,
        'Dest': dests,
        'Pred': data['PredArrDelayMinutes'],
        'Actual': data[target]
    })
    return translated_data

## GetPerformance
def GetPerformance(Y_test, Y_predict):
    f1 = f1_score(Y_test, Y_predict, average='macro')
    recall = recall_score(Y_test, Y_predict, average='macro')
    precision = precision_score(Y_test, Y_predict, average='macro')
    score = {
        'f1': f1, 
        'recall': recall,
        'precision': precision
    }
    return score

## GetInput
#gets details from user
#date is an array where index 0 is month and index 1 is day of the month
def GetInput(test):
    user_input = {
            'departure_city': 'LAX',
            'arrival_city': 'San Francisco',
            'date': [4, 10]
        }
    ## 4/10 works pretty well for 2022 data
    #for quicker testing
    if test:
        return user_input
    
    #get user input
    departure_city = input("Please enter your Departure City or Airport:")
    arrival_city = input("Please enter your Arrival City or Airport:")
    month = int(input("Please enter the NUMBER month you are flying:"))
    day = int(input("Please enter the day of the month you are flying:"))
    user_input = {
            'departure_city': departure_city,
            'arrival_city': arrival_city,
            'date': [month, day]
        }    
    return user_input

#### MAIN
#introduction
print("\n======\nWelcome to our CSS581 Flight Delay Predictor!\n======\n")
state = 0
while True:
    #main menu
    if state == 0:
        print("\nThis is the Main Menu")
        print("1. Look for best flights")
        print("2. Exit")
        state = int(input("Please select an option:"))
        continue
    #request input and make the prediction
    elif state == 1:
        #request input from user
        user_input = GetInput(test)

        #translation input to relevant flights
        relevant_airports = GetRelevantAirports(
            user_input['departure_city'],
            user_input['arrival_city'], 
            flight_registry
        )
        relevant_airports = EncodeAirport(airport_translation_data, relevant_airports)
        flights = GetFlightsFromData(
            flight_data, 
            relevant_airports,
            user_input['date']
        )
        if(len(flights) == 0):
            #could not find any flights, try again
            print("\nCould not find any flights. Returning to Main Menu...\n")
            state = 0
            continue

        #predict best flight
        prediction = PredictBestFlight(
            model,
            selectedFeatures, 
            flights
        )

        print("\nLet's see how we performed on our predictions...")
        print(GetPerformance(prediction['actuals'], prediction['predictions']))
        print('\n\n')

        #if we randomize the results before picking top 5, we are more likely to get a good result
        top_5 = prediction['recommended_flights'].sample(frac=1).head(5)
        top_5 = top_5[['Airline', 'Origin', 'Dest', 'PredArrDelayMinutes', target]]
        top_5 = DecodeToAirlineAndAirports(airline_translation_data, airport_translation_data, top_5)

        print("There were " + str(len(flights)) + " available flights.")
        print("We found " + str(len(prediction['recommended_flights'])) + " flights with the least delay.")
        print("Here are the first 5:")
        print(top_5)

        input("\nPress ENTER to continue...")
        state = 0
        continue
    #loop or exit
    elif state == 2:
        print("Thank you for stopping by! Exiting...")
        break
    else:
        print("That is not a correct option. Please try again")
        state = 0

