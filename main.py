import crime_functions as cf
import pandas as pd


crime_log_url = "https://www.police.ucsd.edu/docs/reports/CallsandArrests/Calls_and_Arrests.asp"
url_prefix = 'https://www.police.ucsd.edu/docs/reports/callsandarrests/'
last_date_file = "last_date.txt"

last_known_date = cf.get_last_known_date(last_date_file)
pdf_list = cf.get_latest_pdfs(crime_log_url, last_known_date)


if len(pdf_list) > 0:
    pdf_file_directories = cf.download_pdfs(pdf_list, url_prefix)
    cases_file_directories = cf.generate_text_files(pdf_file_directories)
            
    filtered_cases = cf.get_all_cases(cases_file_directories)
    clean_crime_df = cf.data_cleaning_fcn(pd.DataFrame(filtered_cases))
    predicted_y = cf.predict_labels(clean_crime_df)
    clean_crime_df = cf.prepare_for_psql(clean_crime_df, predicted_y)
    cf.write_to_psql(clean_crime_df)

    last_known_date = pdf_list[-1][0]
    with open(last_date_file, 'w') as file:
        file.write(last_known_date.strftime('%Y-%m-%d'))


else:
    print("No updates")






