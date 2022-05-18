# ================================================================================================
# Author:		Zuzana Knapekova
# Create date:  2022-05-18
# Description:  Script downloads gasoline prices data and merges it with database
# ===============================================================================================

from urllib.request import urlretrieve
import json
import openpyxl
import pandas as pd
import numpy as np
import shutil
from locale import setlocale, LC_NUMERIC, format_string

setlocale(LC_NUMERIC, 'en_EN')

# Load configuration file
try:
    config = open('configuration_file.json', encoding='utf8', errors='ignore')
    config_dict = json.load(config)
except FileNotFoundError:
    print('The file configuration_file.json was not found.')
except Exception as e:
    print('The file configuration_file.json could not be opened: '+str(e))

try:
    excel_file_path = config_dict['gasoline_proces_data_file_path']
    excel_sheet = config_dict['gasoline_prices_data_sheet_name']
    database_path = config_dict['database_path_weekly']
    database_backup_path = config_dict['database_backup_path_weekly']
except Exception as e:
    print("Failed to load: "+str(e))


# Download excel file
urlretrieve('https://ec.europa.eu/energy/observatory/reports/Oil_Bulletin_Prices_History.xlsx',
            excel_file_path)


df = pd.read_excel(excel_file_path, sheet_name=excel_sheet)

# Search for SK data and create table
sk_index = np.where(df[df.columns[0]] == 'SK')

sk_data = pd.DataFrame(df[sk_index[0][0]:df.shape[0]])
sk_data.dropna(axis=1, how='all', inplace=True)

sk_data.columns = ['country', 'Date', 'Exchange rate', 'Euro-super 95  (I) 1000L',
                   'Gas oil automobile Automotive gas oil Dieselkraftstoff (I) 1000L', 'Gas oil de chauffage Heating gas oil Heizöl (II) 1000L',
                   'Fuel oil - Schweres Heizöl (III) Soufre <= 1% Sulphur <= 1% Schwefel <= 1% t',
                   'Fuel oil -Schweres Heizöl (III) Soufre > 1% Sulphur > 1% Schwefel > 1% t',
                   'GPL pour moteur LPG motor fuel 1000L'
                   ]

sk_data['Date'] = pd.to_datetime(sk_data['Date'], errors='coerce')
sk_data = sk_data[sk_data['Date'].notna()]
sk_data_sorted = sk_data.sort_values('Date')

sk_data_sorted["Euro-super 95  (I) 1000L"] = [float(
    str(i).replace(",", "")) for i in sk_data_sorted["Euro-super 95  (I) 1000L"]]
sk_data_sorted["Gas oil automobile Automotive gas oil Dieselkraftstoff (I) 1000L"] = [float(
    str(i).replace(",", "")) for i in sk_data_sorted["Gas oil automobile Automotive gas oil Dieselkraftstoff (I) 1000L"]]
sk_data_sorted["Gas oil de chauffage Heating gas oil Heizöl (II) 1000L"] = [float(
    str(i).replace(",", "")) for i in sk_data_sorted["Gas oil de chauffage Heating gas oil Heizöl (II) 1000L"]]
sk_data_sorted["Fuel oil - Schweres Heizöl (III) Soufre <= 1% Sulphur <= 1% Schwefel <= 1% t"] = [float(
    str(i).replace(",", "")) for i in sk_data_sorted["Fuel oil - Schweres Heizöl (III) Soufre <= 1% Sulphur <= 1% Schwefel <= 1% t"]]
sk_data_sorted["Fuel oil -Schweres Heizöl (III) Soufre > 1% Sulphur > 1% Schwefel > 1% t"] = [float(
    str(i).replace(",", "")) for i in sk_data_sorted["Fuel oil -Schweres Heizöl (III) Soufre > 1% Sulphur > 1% Schwefel > 1% t"]]
sk_data_sorted["GPL pour moteur LPG motor fuel 1000L"] = [float(
    str(i).replace(",", "")) for i in sk_data_sorted["GPL pour moteur LPG motor fuel 1000L"]]


# Load database excel file
df_import = pd.read_excel(database_path)
max_date = df_import['Date'].max()

# Concat new data with data from database
df_final = pd.concat(
    [df_import, sk_data_sorted[sk_data_sorted['Date'] > max_date]], axis=0)


# Create backup
shutil.copyfile(database_path, database_backup_path)

# Update database excel file
datatoexcel = pd.ExcelWriter(database_path)

df_final.to_excel(datatoexcel, index=False)

datatoexcel.save()
