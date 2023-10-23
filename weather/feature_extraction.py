
import csv


def featureize():
    
    features = []
    labels = []

   
    continent_mapping = {
        "Africa": [1, 0, 0, 0, 0, 0, 0],
        "Australia": [0, 1, 0, 0, 0, 0, 0],
        "North America": [0, 0, 1, 0, 0, 0, 0],
        "South America": [0, 0, 0, 1, 0, 0, 0],
        "Asia": [0, 0, 0, 0, 1, 0, 0],
        "Europe": [0, 0, 0, 0, 0, 1, 0],
        "Antarctica": [0, 0, 0, 0, 0, 0, 1]
    }

    season_mapping = {
        "Spring": [1, 0, 0, 0],
        "Summer": [0, 1, 0, 0],
        "Autumn": [0, 0, 1, 0],
        "Winter": [0, 0, 0, 1]
    }

    wind_speed_mapping = {
        "Low": [1, 0, 0],
        "Normal": [0, 1, 0],
        "High": [0, 0, 1]
    }

    location_mapping = {
        "Coast Line": [1, 0, 0],
        "Coastline": [0, 1, 0],
        "Inland": [0, 0, 1]
    }

    weather_mapping = {
        "Sunny": [1, 0, 0, 0],
        "Rainy": [0, 1, 0, 0],
        "Cloudy": [0, 0, 1, 0], 
        "Humid" : [0, 0, 0, 1]
    }

    temperature_mapping = {
        "Hot": [1, 0, 0],
        "Cold": [0, 1, 0],
        "Normal": [0, 0, 1]
    }

    with open('Prob/train.csv', 'r') as file :
        lines = csv.reader(file)

        for feature in lines:
        
            continent = continent_mapping[feature[0]]
            season = season_mapping[feature[1]]
            wind_speed = wind_speed_mapping[feature[2]]
            location = location_mapping[feature[3]]
            weather = weather_mapping[feature[4]]
            temperature = temperature_mapping[feature[5]]

            features_all = continent + season + wind_speed + location + weather

            features.append(features_all)
            labels.append(temperature)

        return features, labels


features, labels = featureize()
# print("features", features,"labels", labels )
