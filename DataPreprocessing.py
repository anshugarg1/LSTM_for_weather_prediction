import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Data_Preprocess():
    def __init__(self, path: str, plot_graph: bool):
        self.path = path
        self.data = None
        self.df = None
        self.plot_graph = plot_graph  

    def list_funcc(self, s):
        return s.split('-')[0]

    def read_file(self):
        df = pd.read_csv(self.path, delimiter=';')
        df['MIN_TEMP[C]'] = (df['MIN_TEMP[C]'].str.lstrip()).replace(',','.',regex = True).astype(float, errors = 'raise')
        df['MAX_TEMP[C]'] = (df['MAX_TEMP[C]'].str.lstrip()).replace(',','.',regex = True).astype(float, errors = 'raise')
        df['AVG_TEMP[C]'] = (df['AVG_TEMP[C]'].str.lstrip()).replace(',','.',regex = True).astype(float, errors = 'raise')
        df['SUM_RAIN[mm]'] = (df['SUM_RAIN[mm]'].str.lstrip()).replace(',','.',regex = True).astype(float, errors = 'raise')
        df['AVG_HUMI[%]'] = (df['AVG_HUMI[%]'].str.lstrip()).replace(',','.',regex = True).astype(float, errors = 'raise')
        df['SUM_ET[mm]'] = (df['SUM_ET[mm]'].str.lstrip()).replace(',','.',regex = True).astype(float, errors = 'raise')
        df['HOUR'] = df['HOUR'].apply(self.list_funcc)
    #     df['HOUR'] = pd.to_datetime(df['HOUR'])
        df = df.groupby(df['HOUR']).mean() #.reset_index()
        self.df = df
        self.data = df.to_numpy()
        if self.plot_graph:
            _plot_graph()
        self.data = self.data/self.data.max(axis = 0)   #normalize data column wise
        return self.data

    #to visualise the data
    def _plot_graph(self):
        print(self.df.columns)
        
        plt.figure(figsize=(10,5))
        self.df['MAX_TEMP[C]'].plot(label = 'test', title = 'Hour vs MAX_TEMP[C]')    
        plt.show()
        
        plt.figure(figsize=(10,5))
        self.df['MAX_TEMP[C]'].plot(label = 'test', title = 'Hour vs MAX_TEMP[C]')    
        plt.show()
        
        plt.figure(figsize=(10,5))
        self.df['AVG_TEMP[C]'].plot(label = 'test', title = 'Hour vs AVG_TEMP[C]')
        plt.show()
        
        plt.figure(figsize=(10,5))
        self.df['SUM_RAIN[mm]'].plot(label = 'test', title = 'Hour vs SUM_RAIN[mm]')    
        plt.show()
        
        plt.figure(figsize=(10,5))
        self.df['AVG_HUMI[%]'].plot(label = 'test', title = 'Hour vs AVG_HUMI[%]')    
        plt.show()

        plt.figure(figsize=(10,5))
        self.df['SUM_ET[mm]'].plot(label = 'test', title = 'Hour vs SUM_ET[mm]')
        plt.show()    