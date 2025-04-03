import sqlite3
import pandas as pd

def query_climate_database(db_file, country, year_begin, year_end, month):
    """
    Query the climate database for temperature readings in a specified country, date range, and month.

    Parameters:
    - db_file (str): The file name of the SQLite database (i.e. temps.db).
    - country (str): Target country.
    - year_begin (int): Start year.
    - year_end (int): End year.
    - month (int): The month of the year for which data should be returned (1-12).

    Returns:
    - pd.DataFrame: A data frame containing the 
    station name, latitude, longitude, country, year, month, and temperature.
    """
    # Connect to the database
    conn = sqlite3.connect(db_file)
    
    # Writes the SQL query using f-strings
    query = f"""
        SELECT 
            S.name AS NAME,
            S.latitude AS LATITUDE,
            S.longitude AS LONGITUDE,
            C.name AS Country,
            T.year AS Year,
            T.month AS Month,
            T.temp AS Temp
        FROM 
            temperatures T
        JOIN 
            stations S ON T.id = S.id
        JOIN 
            countries C ON SUBSTR(S.id, 1, 2) = C."FIPS 10-4" 
        WHERE 
            C.name = '{country}' AND
            T.year BETWEEN '{year_begin}' AND '{year_end}' AND
            T.month = '{month}'
        ORDER BY
            S.name ASC
    """
    # Matches stations ID from the temps and stations tables
    # Matches using countries: FIPS 10-4 with first two letters of ID from Stations
    # Filters by country name, start/end year, and month
    # Orders the station names in alphabetical order 
    df = pd.read_sql_query(query, conn)
    # Close database
    conn.close()
    
    return df

def second_query_climate_database(db_file, country, year_begin, year_end):
    """
    Query the climate database for temperature data across months and latitudes.

    Parameters:
    - db_file (str): The file name of the SQLite database (i.e. temps.db).
    - country (str): Target country.
    - year_begin (int): Start year.
    - year_end (int): End year.

    Returns:
    - pd.DataFrame: A data frame containing month, latitude, and average temperature.
    """
    # Connect to the database
    conn = sqlite3.connect(db_file)
    
    # Write the SQL query using f-strings
    query = f"""
        SELECT 
            T.month AS Month,
            S.latitude AS LATITUDE,
            ROUND(AVG(T.temp),2) AS Avg_Temp
        FROM 
            temperatures T
        JOIN 
            stations S ON T.id = S.id
        JOIN 
            countries C ON SUBSTR(S.id, 1, 2) = C."FIPS 10-4"
        WHERE 
            C.name = '{country}' AND
            T.year BETWEEN {year_begin} AND {year_end}
        GROUP BY 
            T.month, S.latitude
        ORDER BY 
            T.month, S.latitude;
    """
    # Takes average temperature for each month and latitude
    # Execute the query and load the results into a DataFrame
    df = pd.read_sql_query(query, conn)
    
    # Close the database connection
    conn.close()
    
    return df

