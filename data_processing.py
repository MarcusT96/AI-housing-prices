import pandas as pd

def load_data(file_path):
    housing = pd.read_csv(file_path)
    return housing

def preprocess_data(housing):
    # Skapa nya attribut
    housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
    housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
    housing['population_per_household'] = housing['population'] / housing['households']
    housing['capped'] = (housing['median_house_value'] == 500001).astype(int)

    # Omvandla kategoriska variabler till numeriska
    housing = pd.get_dummies(housing, columns=['ocean_proximity'])

    # Fyll saknade värden med medianvärden
    housing = housing.fillna(housing.median())

    return housing

def split_data(housing):
    X = housing.drop('median_house_value', axis=1)
    y = housing['median_house_value']
    return X, y
