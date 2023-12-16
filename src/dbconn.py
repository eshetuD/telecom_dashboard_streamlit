import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
import psycopg2

database_name = 'telecom'
table_name= 'xdr_data'

@st.cache_data
def connect_db():
    connection_params = { "host": "localhost", "user": "postgres", "password": "root",
                    "port": "5432", "database": "telecom"}

    engine = create_engine(f"postgresql+psycopg2://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}")

    # str or SQLAlchemy Selectable (select or text object)
    sql_query = 'SELECT * FROM xdr_data'

    df1 = pd.read_sql(sql_query, con= engine)
    return df1

df= pd.read_csv('my_db.csv')

df.head()