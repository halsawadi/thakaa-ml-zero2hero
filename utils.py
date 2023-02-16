import pandas as pd
from pathlib import Path
import yaml

PROJECT_PATH = Path(__file__).parent

with open(PROJECT_PATH / "config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def read_config():
    with open(PROJECT_PATH / "config.yml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def join_file_name_with_format(filename, fileformat):
    return filename + '.' + fileformat

def read_saudi_arabia_used_cars_dataset_1():
    saudi_arabia_used_cars_dataset_config = config["saudi_arabia_used_cars_dataset"]
    data_dirname =  saudi_arabia_used_cars_dataset_config["data_dirname"]
    filename = saudi_arabia_used_cars_dataset_config["filename"]
    format = saudi_arabia_used_cars_dataset_config["format"]
    filename_with_format_extension = join_file_name_with_format(filename, format)
    usecols = saudi_arabia_used_cars_dataset_config["columns"]
    df_used_cars = pd.read_csv((PROJECT_PATH / data_dirname / filename_with_format_extension), usecols=usecols)
    df_used_cars_w_prices = df_used_cars[~df_used_cars["Negotiable"]].drop(columns=["Negotiable"])
    df_used_cars_w_prices['Price'] = df_used_cars_w_prices.Price.apply(int)
    return df_used_cars_w_prices

