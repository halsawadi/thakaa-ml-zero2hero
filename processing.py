from utils import *
from pathlib import Path
import numpy as np

# class SaudiArabiaUsedCarsDatasetProcessor:
#     def __init__(self, config) -> None:
#         self.data_dirname = config['data_dirname']
#         self.filename = config['filename']
#         self.format = config['format']
#         self.filepath = Path(PROJECT_PATH / self.data_dirname / (self.filename + '.' + self.format))

#         self.columns = config['columns']
#         self.negotiable_config = config["negotiable"]
#         self.fuel_type_config = config["fuel_type"]
#         self.make_config = config["make"]

#         self.pipeline = [
#             self.select_columns, 
#             self.filter_out_negotiable_cars,
#             self.convert_price_from_str_to_int,
#             self.lower_some_values_in_fuel_type_column,
#             self.compute_synthetic_age_column,
#             self.synthesize_price,
#             self.keep_only_specific_car_makers,
#             self.remove_ford_outliers,
#             self.add_above_50k_column
#             ]

#     def read(self):
#         df = pd.read_csv(self.filepath)
#         return df
    
#     def select_columns(self, df):
#         df = df[self.columns]
#         return df

#     def apply(self):
#         df = self.read()
#         for function in self.pipeline:
#             df = function(df)
#         return df

#     def reset_index(self, df):
#         return df.reset_index(drop=True)

#     def filter_out_negotiable_cars(self, df):
#         negotiable_column_name = self.negotiable_config['negotiable_column_name']
#         if negotiable_column_name not in df.columns:
#             raise KeyError("This dataframe doesn't have the negotiable column")
#         else:
#             df = df[~df[negotiable_column_name]].drop(columns=[negotiable_column_name]).reset_index(drop=True)
#             return df

#     def convert_price_from_str_to_int(self, df):
#         df['Price'] = df.Price.apply(int)
#         return df


#     def lower_some_values_in_fuel_type_column(self, df):
#         fuel_type_column_name = self.fuel_type_config['fuel_type_column_name']
#         lower_some_value_names = self.fuel_type_config['lower_some_value_names']
#         lower_some_values_portion = self.fuel_type_config['lower_some_values_portion']
#         number_of_samples = int(lower_some_values_portion * len(df))
#         indices = np.random.randint(0, len(df), number_of_samples)
#         for value_name in lower_some_value_names:
#             df.loc[indices, fuel_type_column_name] = df.loc[indices, fuel_type_column_name].str.replace(value_name, value_name.lower())
#         return df

#     def introduce_missing_values(self, df):
#         pass

#     def compute_synthetic_age_column(self, df):
#         month_margin = 3
#         df['Age'] = (2023 - df.Year + pd.Series(np.random.choice([-1, 1], len(df))) * np.random.randint(month_margin, size=len(df))  + np.random.randn(len(df))*1)*12
#         return df

#     def keep_only_specific_car_makers(self, df):
#         make_column_name = self.make_config['column_name']
#         car_makers = self.make_config['car_makers']
#         return df[df[make_column_name].isin(car_makers)]

#     def synthesize_price(self, df):
#         factor = 0.05
#         bias = 5000

#         df = df[df['Make'] == 'Ford'].reset_index(drop=True)

#         df_grouped = df.groupby(['Year'])['Price'].median().reset_index()
#         dict_year_mean_price = dict(zip(df_grouped.Year, df_grouped.Price))
#         df['Price'] = df.Year.apply(lambda x: dict_year_mean_price[x]) * (1-factor) + df.Price * pd.Series(np.random.choice([-1, 1], len(df))) * factor + bias
#         return df

#     def remove_ford_outliers(self, df):
#         df = df[(df['Price'] < 150000) & (df['Age'] < 500)]
#         return df


#     def add_above_50k_column(self, df):
#         df['>=70k'] = df.Price.apply(lambda x: x >= 70000)
#         return df


class SaudiArabiaUsedCarsDatasetProcessor:
    def __init__(self, config) -> None:
        self.data_dirname = config['data_dirname']
        self.filename = config['filename']
        self.format = config['format']
        self.filepath = Path(PROJECT_PATH / self.data_dirname / (self.filename + '.' + self.format))

        self.columns = config['columns']
        self.negotiable_config = config["negotiable"]
        self.fuel_type_config = config["fuel_type"]
        self.make_config = config["make"]

        self.pipeline = [
            self.select_columns, 
            self.filter_out_negotiable_cars,
            self.convert_price_from_str_to_int,
            self.lower_some_values_in_fuel_type_column,
            ]

    def read(self):
        df = pd.read_csv(self.filepath)
        return df
    
    def select_columns(self, df):
        df = df[self.columns]
        return df

    def apply(self):
        df = self.read()
        for function in self.pipeline:
            df = function(df)
        return df

    def reset_index(self, df):
        return df.reset_index(drop=True)

    def filter_out_negotiable_cars(self, df):
        negotiable_column_name = self.negotiable_config['negotiable_column_name']
        if negotiable_column_name not in df.columns:
            raise KeyError("This dataframe doesn't have the negotiable column")
        else:
            df = df[~df[negotiable_column_name]].drop(columns=[negotiable_column_name]).reset_index(drop=True)
            return df

    def convert_price_from_str_to_int(self, df):
        df['Price'] = df.Price.apply(int)
        df = df[df['Price'] < 700000].reset_index(drop=True)
        return df


    def lower_some_values_in_fuel_type_column(self, df):
        fuel_type_column_name = self.fuel_type_config['fuel_type_column_name']
        lower_some_value_names = self.fuel_type_config['lower_some_value_names']
        lower_some_values_portion = self.fuel_type_config['lower_some_values_portion']
        number_of_samples = int(lower_some_values_portion * len(df))
        indices = np.random.randint(0, len(df), number_of_samples)
        for value_name in lower_some_value_names:
            df.loc[indices, fuel_type_column_name] = df.loc[indices, fuel_type_column_name].str.replace(value_name, value_name.lower())
        return df

    def introduce_missing_values(self, df):
        pass


class FordDatasetProcessor(SaudiArabiaUsedCarsDatasetProcessor):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.pipeline = [
            self.select_columns, 
            self.filter_out_negotiable_cars,
            self.convert_price_from_str_to_int,
            self.lower_some_values_in_fuel_type_column,
            self.compute_synthetic_age_column,
            self.synthesize_price,
            self.keep_only_specific_car_makers,
            self.remove_ford_outliers,
            self.add_above_50k_column
            ]
    def compute_synthetic_age_column(self, df):
        month_margin = 3
        df['Age'] = (2023 - df.Year + pd.Series(np.random.choice([-1, 1], len(df))) * np.random.randint(month_margin, size=len(df))  + np.random.randn(len(df))*1)*12
        return df

    def keep_only_specific_car_makers(self, df):
        make_column_name = self.make_config['column_name']
        car_makers = self.make_config['car_makers']
        return df[df[make_column_name].isin(car_makers)]

    def synthesize_price(self, df):
        factor = 0.05
        bias = 5000

        df = df[df['Make'] == 'Ford'].reset_index(drop=True)

        df_grouped = df.groupby(['Year'])['Price'].median().reset_index()
        dict_year_mean_price = dict(zip(df_grouped.Year, df_grouped.Price))
        df['Price'] = df.Year.apply(lambda x: dict_year_mean_price[x]) * (1-factor) + df.Price * pd.Series(np.random.choice([-1, 1], len(df))) * factor + bias
        return df

    def remove_ford_outliers(self, df):
        df = df[(df['Price'] < 150000) & (df['Age'] < 500)]
        return df


    def add_above_50k_column(self, df):
        df['>=70k'] = df.Price.apply(lambda x: x >= 70000)
        return df




    
def read_saudi_arabia_used_cars_dataset():
    config = read_config()
    saudi_arabia_used_cars_dataset_config = config['saudi_arabia_used_cars_dataset']
    sa_used_cars_processor = SaudiArabiaUsedCarsDatasetProcessor(saudi_arabia_used_cars_dataset_config)
    df = sa_used_cars_processor.apply()
    return df


def read_ford_used_cars_dataset():
    config = read_config()
    saudi_arabia_used_cars_dataset_config = config['saudi_arabia_used_cars_dataset']
    ford_used_cars_processor = FordDatasetProcessor(saudi_arabia_used_cars_dataset_config)
    df = ford_used_cars_processor.apply()
    return df
    