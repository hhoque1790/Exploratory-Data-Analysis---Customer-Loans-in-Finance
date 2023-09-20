import yaml
from sqlalchemy import create_engine
from sqlalchemy import URL
from sqlalchemy import text
import pandas as pd

with open('credentials.yaml', 'r') as file:
    credentials = yaml.safe_load(file)

class RDSDatabaseConnector:
    def __init__(self,credentials):
        self.credentials=credentials
    
    def start_engine(self):
        url_object = URL.create(
        "postgresql+psycopg2",
        username=self.credentials['RDS_USER'],
        password=self.credentials['RDS_PASSWORD'],  # plain (unescaped) text
        host=self.credentials['RDS_HOST'],
        database=self.credentials['RDS_DATABASE']
        )
        
        engine = create_engine(url_object)
        with engine.connect() as connection:
            result = connection.execute(text("select * from loan_payments"))
            result=result.fetchall()

        loan_payments_df=pd.DataFrame(result)
        loan_payments_df.to_csv('loan_payments_df.csv')
        print(loan_payments_df)


Connect = RDSDatabaseConnector(credentials)
Connect.start_engine()