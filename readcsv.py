import numpy as np
import pandas as pd

def get_data(file='file_names.txt', start='1970-01-01', end='2020-12-31', 
             cols=['Close', 'Open', 'High', 'Low'], 
             method='ffill', init='bfill', verbose=False):   

    """ 
    Get all daily data, padding NaNs with appropriate values.
    
    - file: .txt file containing file names to include.
    - start: start day of timeseries.  
    - end: end day of timeseries.
    - cols: columns to obtain.
    - method: how to fill missing data points (see pd.reindex documentation).
    - init: how to fill leading NaNs (data history too short).
        'bfill': fill with first valid data point.
        x: custom replacement value.
    - verbose: print columns included in data.
    """

    file_names = get_file_names(file)
    series = []
    ix = pd.date_range(start=start, end=end, freq='D')
    count = 0

    for name in file_names:
        
        df = pd.read_csv(name, sep=',', header=2, index_col=0, parse_dates=['Date'])

        mean = None
        std = None

        for col in cols:

            s = df[col]

            ## if column NaN, don't use
            if pd.isna(s[0]):
                continue

            if verbose:
                print(count, name, col)

            ## resample to have daily data points
            s = s.reindex(ix, method=method)

#            ## normalize to Close
#            ticker = df['Ticker'][0]
#            if normalize and ticker != 'USDEUR':
#                if col == 'Close':
#                    mean = s.mean()
#                    std = s.std(ddof=0)
#                s = (s - mean)/std

            ## handle leading NaNs
            if init == 'bfill':
                s.bfill(inplace=True)
            else:
                s.fillna(init, inplace=True)

            series.append(s)
            count += 1

    return pd.concat(series, axis=1).to_numpy()



def get_file_names(file_name):
    with open(file_name) as f:
        lines = ['Data/'+s.strip('\n') for s in f.readlines()]
    return lines



