
#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


#%%

# READING THE CASE YAHOO TICKER SAMPLE LIST!
df = pd.read_excel("yahoo_ticker_sample_scraped.xlsx")
# df.info()


#%%

# LIST OF FIELDS SCRAPPED
list_of_fields = ['Market Cap', 'Enterprise Value', 'Trailing P/E', 'Forward P/E', 'PEG Ratio', 'Price/Sales', 'Price/Book', 'Enterprise Value/Revenue',
                  'Enterprise Value/EBITDA', 'Fiscal Year Ends', 'Most Recent Quarter', 'Profit Margin', 'Operating Margin', 'Return on Assets',
                  'Return on Equity', 'Revenue', 'Revenue Per Share', 'Quarterly Revenue Growth', 'Gross Profit', 'EBITDA',
                  'Net Income Avi to Common', 'Diluted EPS', 'Quarterly Earnings Growth', 'Total Cash', 'Total Cash Per Share', 'Total Debt',
                  'Total Debt/Equity', 'Current Ratio', 'Book Value Per Share', 'Operating Cash Flow', 'Levered Free Cash Flow', 'Beta', '52-Week Change',
                  'S&amp;P500 52-Week Change', '52 Week High', '52 Week Low', '50-Day Moving Average', '200-Day Moving Average', 'Avg Vol (3 month)',
                  'Avg Vol (10 day)', 'Shares Outstanding', 'Float', '% Held by Insiders', '% Held by Institutions', 'Shares Short', 'Short Ratio', 'Short % of Float',
                  'Shares Short (prior month)', 'Forward Annual Dividend Rate', 'Forward Annual Dividend Yield', 'Trailing Annual Dividend Rate',
                  'Trailing Annual Dividend Yield', '5 Year Average Dividend Yield', 'Payout Ratio', 'Dividend Date', 'Ex-Dividend Date',
                  'Last Split Factor', 'Last Split Date']
print("Number of columns of the dataset",len(list_of_fields))
#%%

#DOING SOME DATA CLEANSING

for fieldname in df.columns.values:
    df[fieldname]= df[fieldname].fillna(0)
print("Filling 0 , NA values: ")
print(df)


#MISSING DATA TO CHECK MISSING DATA

df.info()
sns.heatmap(df.isnull(), cbar=False)
msno.matrix(df)
# plt.show()

#%%

#LIST OF FIELDS, EXCLUDING THE DATES
list_of_fields_nodates = ['Market Cap', 'Enterprise Value', 'Trailing P/E', 'Forward P/E', 'PEG Ratio', 'Price/Sales', 'Price/Book', 'Enterprise Value/Revenue',
                          'Enterprise Value/EBITDA', 'Profit Margin', 'Operating Margin', 'Return on Assets', 'Return on Equity', 'Revenue', 'Revenue Per Share',
                          'Quarterly Revenue Growth', 'Gross Profit', 'EBITDA', 'Net Income Avi to Common', 'Diluted EPS', 'Quarterly Earnings Growth',
                          'Total Cash', 'Total Cash Per Share', 'Total Debt', 'Total Debt/Equity', 'Current Ratio', 'Book Value Per Share', 'Operating Cash Flow',
                          'Levered Free Cash Flow', '52-Week Change', 'S&amp;P500 52-Week Change', '52 Week High', '52 Week Low', '50-Day Moving Average',
                          '200-Day Moving Average', 'Avg Vol (3 month)', 'Avg Vol (10 day)', 'Shares Outstanding', 'Float', '% Held by Insiders', '% Held by Institutions',
                          'Shares Short', 'Short Ratio', 'Short % of Float', 'Shares Short (prior month)', 'Forward Annual Dividend Rate', 'Forward Annual Dividend Yield',
                          'Trailing Annual Dividend Rate', 'Trailing Annual Dividend Yield', '5 Year Average Dividend Yield', 'Payout Ratio']
print("Number of columns with nodates:", len(list_of_fields_nodates))



#FUNCTION TO TRANSFORM TEXT  INTO NUMBER 3.17B ---> 3,170,000,000).
d = {
     'k':  3,
     'M':  6,
     'B':  9,
     'T': 12,
     '%': -2
}
def text_to_num(text):
    try:
        if text[-1] in d:
            num, magnitude = text[:-1], text[-1]
            return float(num) * 10 ** d[magnitude]  #this case is when "text" has T, B, M, k or %
        else:
            return float(text) #this case is when "text" is string but look like a numeric
    except:
        try:
            return 1.0*text #this is when "text" is already numeric
        except:
            return 0.0 #it will reach this case when it is impossible to transform into numeric


#Apply the function in a optimized way using Numpy Vectorize - Example on how to call the function: text_to_num('3.17B')
for fieldname in list_of_fields_nodates:
    df[fieldname] = np.vectorize(text_to_num)(df[fieldname])

# print(df)
#
# #CREATING THE GROUPS - Bucketing Normalization
IN_MODEL = []
for fieldname in list_of_fields_nodates:
    try:
        df[fieldname+'_group'] = df.groupby(['sector'])[fieldname].transform(
                             lambda x: pd.qcut(x, [0.0714, 0.2143, .5, 0.7857, 0.9286, 1.], labels=range(1,6)))
        df[fieldname+'_group']= df[fieldname+'_group']
        IN_MODEL.append(fieldname+'_group')
    except ValueError:
        print("Could not be binned :", fieldname+'_group')
        df[fieldname+'_group'] = 0
        continue
#
print(IN_MODEL)




# DEVELOPING A SIMPLE MODEL
output_var = 'Beta'
X = df[list(IN_MODEL)]

X = X.apply(pd.to_numeric)
for fieldname in X.columns.values:
    X[fieldname]= X[fieldname].fillna(0)

y = df[output_var]

#Missing data for new columns:

print("----------------------------X---------------------------")
print(X)


try:
    model = sm.OLS(y.astype(float),X.astype(float))
    result = model.fit()
    print (result.summary())
    y_pred = result.predict(X)
except np.linalg.linalg.LinAlgError as err:
    if 'Singular matrix' in err.message:
        print ("MODEL-INVALID (Singular Matrix)")
    else:
        raise

df['pred'] = y_pred
#



# SAVING TO EXCEL ALL PREDICTED  DATA!
df.to_excel("yahoo_ticker_sample_scraped_grouped_predicted.xlsx")
print (df)
