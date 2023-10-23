
import csv


def featureize():
    
    features = []
    labels = []

   
    continent_mapping = {
        "Africa": 0,
        "Australia": 1,
        "North America": 2,
        "South America": 3,
        "Asia": 4,
        "Europe": 5,
        "Antarctica": 6
    }

    season_mapping = {
        "Spring": 0,
        "Summer":1, 
        "Autumn": 2,
        "Winter": 3
    }

    wind_speed_mapping = {
        "Low": 0,
        "Normal": 1,
        "High": 2
    }

    location_mapping = {
        "Coast Line": 0,
        "Coastline": 1,
        "Inland": 2
    }

    weather_mapping = {
        "Sunny": 0,
        "Rainy": 1,
        "Cloudy": 2,
        "Humid" : 3
    }

    temperature_mapping = {
        "Hot": 0,
        "Cold": 1,
        "Normal": 2
    }

    with open('Prob/test.csv', 'r') as file :
        lines = csv.reader(file)

        for feature in lines:
        
            continent = continent_mapping[feature[0]]
            season = season_mapping[feature[1]]
            wind_speed = wind_speed_mapping[feature[2]]
            location = location_mapping[feature[3]]
            weather = weather_mapping[feature[4]]
            temperature = temperature_mapping[feature[5]]

            features_all = [continent] + [season] + [wind_speed] + [location] + [weather]

            features.append(features_all)
            labels.append(temperature)

        lens = [len(continent_mapping), len(season_mapping), len(wind_speed_mapping), len(location_mapping), len(weather_mapping)]

        return features, labels, lens


features, labels, lens = featureize()
# print("features", features,"labels", labels, "lens", lens )
