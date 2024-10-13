from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from config import Config

config = Config()

def init_database():
    db_uri = config.get_database_config()["url"]
    engine = create_engine(db_uri)
    session = sessionmaker(bind=engine)
    return engine, session()
