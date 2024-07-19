# Crime Summary Natural Language Processing Bot


## crime_functions.py
Custom functions to download and parse PDFs from the official UC San Diego Crime Log [Website](https://www.police.ucsd.edu/docs/reports/CallsandArrests/Calls_and_Arrests.asp) can be found in crime_functions.py.

## main.py 
main.py is a script that runs every single day on a backend server to check the UC San Diego Crime Log [Website](https://www.police.ucsd.edu/docs/reports/CallsandArrests/Calls_and_Arrests.asp) for any updates. It runs every day at:
- 8 AM
- 12 PM
- 5 PM

If there are any new updates, the script will use the functions in crime_functions.py to download crime logs, parse crime logs, filter crime logs, predict "interesting" crime log summaries and upload them to a PostgreSQL database.

## SVM_model_trainining.ipynb
Simply a notebook denoting how I trained my Support Vector Machine Classifier to classify new and unseen crime log summaries. 

## pickle_models/
The actual TF-IDF vectorizer and SVM model that I am using on the backend to predict "interesting" crime log summaries. I will be creating better models as I gather more data. 