import re
from datetime import datetime
import streamlit as st
import pandas as pd
import os
import logging
from text_to_sql.entity_data import Entity_joins

def create_sql_prompt_with_examples(Entity_joins):
    examples = "\n".join([f"Query: {item['query']}\nSQL Query: {item['sql']}" for item in Entity_joins])
    return examples


def process_generated_sql(sql_query, question):
    return sql_query


def execute_sql_query(sql_query, db_engine):
    try:
        with db_engine.connect() as connection:
            result_df = pd.read_sql_query(sql_query, connection)
            return result_df
    except Exception as e:
        st.error(f"Error executing SQL query: {str(e)}")
        logging.error(f"Error executing SQL query: {e}")
        return None
