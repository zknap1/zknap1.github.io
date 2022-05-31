
from selenium import webdriver
import json
import pandas as pd
import os
import time

try:
    config = open('configuration_file.json', encoding='utf8', errors='ignore')
    config_dict = json.load(config)
except FileNotFoundError:
    print('The file configuration_file.json was not found.')
except Exception as e:
    print('The file configuration_file.json could not be opened: '+str(e))

try:
    chrome_webdriver_path = config_dict['chrome_webdriver_path']
    downloads = config_dict['downloads']
except Exception as e:
    print("Failed to load: "+str(e))


options = webdriver.ChromeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--disable-crash-reporter")
options.add_argument("--disable-extensions")
options.add_argument("--disable-in-process-stack-traces")
options.add_argument("--disable-logging")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--log-level=3")
options.add_argument("--output=/dev/null")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(chrome_webdriver_path, options=options)

URL = "http://www.bsse.sk/bcpben/Trading/Indices/SAXIndex/tabid/163/language/en-US/Default.aspx"
driver.get(URL)

# download excel file
export_data_xpath = '/html/body/form/dllimport/table/tbody/tr/td/table/tbody/tr/td/table/tbody/tr/td[2]/table/tbody/tr[3]/td[2]/table/tbody/tr[5]/td/table/tbody/tr[1]/td[1]/div[1]/div/div/table[2]/tbody/tr[2]/td[2]/div[2]/a'
driver.find_element_by_xpath(export_data_xpath).click()

time.sleep(2)
df = pd.read_csv(downloads+"SAXIndex.csv")
print(df)

if os.path.exists(downloads+"SAXIndex.csv"):
    os.remove(downloads+"SAXIndex.csv")

driver.close()
