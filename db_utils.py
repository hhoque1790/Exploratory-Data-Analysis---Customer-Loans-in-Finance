import yaml
from sqlalchemy import create_engine
from sqlalchemy import URL
from sqlalchemy import text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.stats import normaltest, yeojohnson
import plotly.express as px
import plotly


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
        return loan_payments_df
        # print(loan_payments_df)


# Connect = RDSDatabaseConnector(credentials)
# loan_payments_df=Connect.start_engine()

class DataFrameInfo:
    def __init__(self,loan_payments_df):
        self.loan_payments_df=loan_payments_df
    
    def types(self):
        return self.loan_payments_df.dtypes
    
    def mean(self):
        return self.loan_payments_df.describe().iloc[[1]]
    def median(self):
        medianlst=[]
        count=0
        for (columnName, columnData) in self.loan_payments_df.items():
            try:
                medianlst.append(np.median(columnData.values))
            except Exception as e:
                continue
            
            # if count==4:
            #     print(columnName)
            #     print(columnData)
            #     print(np.median(columnData.values))
            count += 1
        return medianlst
    def std(self):
        return self.loan_payments_df.describe().iloc[[2]]
    def distinct(self):
        self.loan_payments_df['term']
        # print(self.loan_payments_df.select_dtypes(include=['object']).columns)
        return self.loan_payments_df.select_dtypes(include=['object']).columns
    
    def unique(self):
        uniquevals=[]
        for col in self.loan_payments_df:
            if loan_payments_df[col].dtypes==object:
                uniquevals.append(len(loan_payments_df[col].unique()))
        return uniquevals
    
    def shape(self):
        return self.loan_payments_df.shape
    
    def nuls(self):
         percent= (self.loan_payments_df.isnull().sum() * 100 / len(self.loan_payments_df)).tolist()
         missing_value_df = pd.DataFrame({
             'Category':self.loan_payments_df.columns,
             'percent_missing': percent})
        #  print(missing_value_df)
         return missing_value_df[1:]
    
    def agostino(self,loan_payments_df):
        columns=[]
        stats=[]
        pvals=[]
        for column in loan_payments_df:
            try:
                stat, p = normaltest(loan_payments_df[column], nan_policy='omit')
                columns.append(column)
                pvals.append(p)
                stats.append(stat)
            except:
                continue
       
        agostinostats=pd.DataFrame({'Categories':columns,
                                    'agostinoStat':stats,
                                    'P-Values':pvals})
        return agostinostats
    def checkskew(self,loan_payments_df):
        columns=[]
        skewvals=[]
        for column in loan_payments_df:
            try:
                skewval=loan_payments_df[column].skew()
                columns.append(column)
                skewvals.append(skewval)
            except Exception as e:
                continue
        skewstats=pd.DataFrame({'Categories':columns,
                                'skewvals':skewvals,
                                })
            
        return skewstats

loan_payments_df = pd.read_csv ('loan_payments_df.csv')
Info=DataFrameInfo(loan_payments_df)

# print(Info.types())
# print(Info.mean())
# print(Info.std())
# print(Info.median())
# print(Info.distinct())
# print(type(Info.distinct()))
# print(Info.unique())
# print(Info.shape())
# print(Info.nuls())

class Plotter:
    # def __init__(self,missing_value_df):
    #     self.missing_value_df=missing_value_df

    def bargraph(self,missing_value_df):
        print(missing_value_df)
        bargraph = missing_value_df.plot.bar(x='Category',y = 'percent_missing', fontsize='5')
        plot.show()

    def scatter(self,loan_payments_df,category):
        fig = px.box(loan_payments_df, y=category)
        plotly.offline.plot(fig)
    
    def matrixcorr(self,loan_payments_df):
        fig=px.imshow(loan_payments_df.corr(), title="Correlation heatmap of loan_payments_df")
        plotly.offline.plot(fig)

# View all NULLs that need to be removed.
# missing_value_df=Info.nuls()
# vis=Plotter(missing_value_df)
# vis.bargraph()

class DataframeTransform:
    def __init__(self,loan_payments_df):
        self.loan_payments_df=loan_payments_df
    
    def drop(self,*args):
        return self.loan_payments_df.drop(columns=[args[0], args[1],args[2],args[3]])

    def impute(self,category,loan_payments_df):
        return loan_payments_df[category].fillna(loan_payments_df[category].median())
    
    def impute_mode(self,category,loan_payments_df):
        mode=loan_payments_df[category].mode().tolist()[0]
        loan_payments_df[category]=loan_payments_df[category].fillna(mode)
        return loan_payments_df[category]
    
    # def dropsrows(self):

    def logtransform(self,loan_payments_df,category):
        # print(loan_payments_df[category].skew())
        loan_payments_df[category] = loan_payments_df[category].map(lambda i: np.log(i) if i > 0 else 0)
        # print(loan_payments_df[category].skew())
        return loan_payments_df[category] 
    def boccoxtransform(self,loan_payments_df,category):
        from scipy import stats
        boxcox_population = loan_payments_df[category]
        boxcox_population= stats.boxcox(boxcox_population)
        boxcox_population= pd.Series(boxcox_population[0])
        loan_payments_df[category]=boxcox_population
        return loan_payments_df[category]
    def YeoJohnsontransform(self,loan_payments_df,category):
        yeojohnson_population = loan_payments_df[category]
        yeojohnson_population = yeojohnson(yeojohnson_population)
        yeojohnson_population= pd.Series(yeojohnson_population[0])
        loan_payments_df[category]=yeojohnson_population
        return loan_payments_df[category]

    # def removeOutliers():

Transform=DataframeTransform(loan_payments_df)


# Dropping colums with over 50% missing data
modloan_payments_df=Transform.drop('mths_since_last_record','mths_since_last_major_derog','next_payment_date','mths_since_last_delinq')
# modloan_payments_df=loan_payments_df.drop(columns=['mths_since_last_record', 'mths_since_last_major_derog','next_payment_date','mths_since_last_delinq'])


# Imputing values in columns with less than 10% missing data using median
modloan_payments_df['funded_amount']=Transform.impute('funded_amount',modloan_payments_df)
modloan_payments_df['int_rate']=Transform.impute('int_rate',modloan_payments_df)
modloan_payments_df['collections_12_mths_ex_med']=Transform.impute('collections_12_mths_ex_med',modloan_payments_df)

# Imputing values in columns using mode
modloan_payments_df['term']=Transform.impute_mode('term',modloan_payments_df)
modloan_payments_df['last_payment_date']=Transform.impute_mode('last_payment_date',modloan_payments_df)

# modloan_payments_df['term'] = modloan_payments_df['term'].fillna(modloan_payments_df['term'].median())
# modloan_payments_df['last_payment_date'] = modloan_payments_df['last_payment_date'].fillna(modloan_payments_df['last_payment_date'].median())

modloan_payments_df['last_credit_pull_date']=Transform.impute_mode('last_credit_pull_date',modloan_payments_df)
modloan_payments_df['employment_length']=Transform.impute_mode('employment_length',modloan_payments_df)

# Delete rows containg data/imput using mode as there is less than 5% missing data here
# modloan_payments_df['last_credit_pull_date'] = modloan_payments_df['last_credit_pull_date'].fillna(modloan_payments_df['last_credit_pull_date'].median())
# modloan_payments_df['employment_length'] = modloan_payments_df['employment_length'].fillna(modloan_payments_df['employment_length'].median())

modInfo=DataFrameInfo(modloan_payments_df)

#Check that all NULLs have been removed.
# missing_value_df=modInfo.nuls()
# vis=Plotter(missing_value_df)
# vis.bargraph()

#Check whether or not data is skewed
# print(modInfo.agostino(modloan_payments_df))
# print(list(modloan_payments_df))

#Check whether or not data is skewed
# skewstats=modInfo.checkskew(modloan_payments_df)
# print("BEFORE TRANSFORMATION:")
# print(skewstats.query('skewvals > 5'))

# Below categories have been successfuly transformed so that skew values are less than 5.
modloan_payments_df['annual_inc']=Transform.boccoxtransform(modloan_payments_df,'annual_inc')
modloan_payments_df['recoveries']=Transform.logtransform(modloan_payments_df,'recoveries')
modloan_payments_df['delinq_2yrs']=Transform.YeoJohnsontransform(modloan_payments_df,'delinq_2yrs')
modloan_payments_df['collection_recovery_fee']=Transform.YeoJohnsontransform(modloan_payments_df,'collection_recovery_fee')

# Below categories could not be transformed with skew values less than 5 regardles of transformation method used.
modloan_payments_df['collections_12_mths_ex_med']=Transform.YeoJohnsontransform(modloan_payments_df,'collections_12_mths_ex_med')
modloan_payments_df['total_rec_late_fee']=Transform.logtransform(modloan_payments_df,'total_rec_late_fee')

# skewstats=modInfo.checkskew(modloan_payments_df)
# print("AFTER TRANSFORMATION:")
# print(skewstats.query('skewvals > 5'))

# Plotting various categories to see data distribution
# modloan_payments_df['collection_recovery_fee'].hist(bins=40)
# modloan_payments_df['member_id'].hist(bins=40)
# modloan_payments_df['loan_amount'].hist(bins=40)
# modloan_payments_df['funded_amount'].hist(bins=40)
# modloan_payments_df['member_id'].hist(bins=100)
# plot.show()

# print(list(modloan_payments_df))
ordinal_cat=list(modloan_payments_df.select_dtypes(include = ['float','integer']))
graph=Plotter()
for count,i in enumerate(ordinal_cat):
    if count in [0,1,2]:
        continue
    stats=modloan_payments_df[i].describe()
    Q1=stats.iloc[4]
    Q3=stats.iloc[6]
    IQR=Q3-Q1
    upperfence=(Q3+(1.5*IQR)).item()

    lowerfence=Q1-(1.5*IQR).item()
    
    query="{category} > {upperfence}".format(category=i,upperfence=upperfence)
    toremove=modloan_payments_df.query(query)
    modloan_payments_df = modloan_payments_df[~modloan_payments_df.isin(toremove)].dropna(how="all")

    query="{category} < {lowerfence}".format(category=i,lowerfence=lowerfence)
    toremove=modloan_payments_df.query(query)
    modloan_payments_df = modloan_payments_df[~modloan_payments_df.isin(toremove)].dropna(how="all")
    
    # graph.scatter(modloan_payments_df[i],i)
    # user=input("Next Graph? Press Enter")

# Correlation Matrix
# graph.matrixcorr(modloan_payments_df.select_dtypes(include = ['float','integer']))

graph.scatter(modloan_payments_df['collection_recovery_fee'],'collection_recovery_fee')
# graph.scatter(modloan_payments_df['annual_inc'],'annual_inc')
# graph.scatter(modloan_payments_df['delinq_2yrs'],'delinq_2yrs')
