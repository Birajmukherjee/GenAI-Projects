import streamlit as st
from text_to_sql.llm import generate_layman_explanation, init_business_friendly_chain, init_llm_chain, init_layman_chain
from text_to_sql.db import init_database
from text_to_sql.sql_generation import create_sql_prompt_with_examples, execute_sql_query
from text_to_sql.entity_data import Entity_joins
import pandas as pd
from config import Config

def main():
    with st.sidebar:
        st.markdown('''
        ## Intelligent Operational Excellency Tracker:
        **The Operational Database Insights Bot empowers us with valuable data insights to optimize processes and enhance performance.**

        **:blue[Some prominent prompts]**
        - Retrieve the list of containers whose status are in visit state active.
        - Retrieve the list of containers whose status are in departed state.
        - Provide the list of containers with goods moving from port origin to destination.
        - Retrieve the list of Horizon departed from the yard and currently in on the move towards truck.
        - Retrieve the list of horizon which are already departed for the transit in vessels.
        - Provide the list of Ships carrying vessela are on the yard and not on move.
        - List out all the vessels and its names which are supposed to arrive in next 30 days in Apapa terminal.
        - List out all the vessels and its names which are already departed within last 60 days from Apapa terminal.
        - Fetch the count of carrier modes for each of the facility for Apapa terminal.
        ''')

    text_to_sql_question_input = st.text_input("What you love to know:")
    
    if text_to_sql_question_input:
        sql_examples = create_sql_prompt_with_examples(Entity_joins)
        context_explanation = """
                    Only generate a valid SQL query using the following tables and columns:
                    inv_unit:
                        GKEY: Unique identifier for the unit.
                        ID: Identifier for the unit.
                        VISIT_STATE: State of the visit.
                        CREATE_TIME: Time when the unit was created.
                        CATEGORY: Category of the unit.
                        FREIGHT_KIND: Type of freight.
                        TRUCKING_COMPANY: Trucking company associated with the unit.
                        GOODS: Goods associated with the unit.
                        ACTIVE_UFV: Active unit facility visit.
                        POL_GKEY: Foreign key to the port of loading.
                        CV_GKEY: Foreign key to the carrier visit.
                        POD1_GKEY: Foreign key to the first port of discharge.
                        POD2_GKEY: Foreign key to the second port of discharge.
                        IMPED_VESSEL: Impeding vessel.

                    inv_goods:
                        gkey: Unique identifier for the goods.
                        consignee: Consignee details.
                        shipper: Shipper details.
                        consignee_bzu: Foreign key to the consignee business unit.
                        shipper_bzu: Foreign key to the shipper business unit.
                        commodity_gkey: Foreign key to the commodity.
                        origin: Origin of the goods.
                        destination: Destination of the goods.
                    """
        try:
            llm_chain = init_llm_chain()
            response = llm_chain.invoke({
                "input_question": text_to_sql_question_input,
                "context_explanation": context_explanation,
                "examples": sql_examples
            })
            raw_response = response.get('text', '').strip() if isinstance(response, dict) else response.strip()
            if raw_response is None:
                st.error("Error invoking LLM. Check logs for details.")
                return
            raw_response = raw_response.replace("```sql\n", "")
            raw_response = raw_response.replace("```", "")
            raw_response = raw_response.replace("Sure,hereistheSQLqueryforthespecifiedinput:", "")
            raw_response = raw_response.replace("Certainly! Here is the SQL query based on the provided input:", "")
            st.info("Generated SQL Query:")
            st.write("### Generated SQL Query:")
            st.code(raw_response, language='sql')
            engine, session = init_database()
            result_df = execute_sql_query(raw_response, engine)

            if result_df is not None and not result_df.empty:
                st.write("### Query Result:")
                st.dataframe(result_df)
                explanation = init_business_friendly_chain(raw_response,result_df)

                if explanation:
                    with st.expander("**See Business Users Explanation:**"):
                        st.write(explanation)
                else:
                    st.write("No results found for the query.")
        except ValueError as ve:
            st.error(f"Error generating SQL query: {ve}")
        except Exception as e:
            st.error(f"An error occurred: {e}")


def run():
    main()