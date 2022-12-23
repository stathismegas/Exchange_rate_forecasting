import numpy as np


def generate_batches_regression(data, batch_size, learn_window, predict_window, 
                                start_days=[], seed=None):
    """
    Randomly generate batches for regression. Returns Generator that 
    yields tuples (X, y), where X contains the data to learn and y
    contains the data to predict.
    
    - data: shape = (N, C)
    - batch_size: integer
    - learn_window: array of indices to learn from.
    - predict_window: array of indices to predict.
    - start_days: array of indices that we can start a learn window on. If empty, then uses entire data.
    - seed: seed for RNG, if any.
    
    e.g. if learn_window = [0, 1, 2, ..., 9] and predict_window = [0, 1, 2, 3, 4],
    then learn on 10 days and try to predict next 5 days.
    
    returns: generator that yields a tuple (X, y)
    - X: shape = (batch_size, len(learn_window), C)
    - y: shape = (batch_size, len(predict_window))
    """
    
    if seed != None:
        np.random.seed(seed)
        
    learn_window = np.array(learn_window)
    predict_window = np.array(predict_window)
    
    N, C = data.shape
    L, P = len(learn_window), len(predict_window)
    Lcap, Pcap = int(np.max(learn_window))+1, int(np.max(predict_window))+1
    
    M = 0 # size of start_days
    if len(start_days) == 0:
    
        M = N - Lcap - Pcap 
        start_days = np.arange(M)
        
    else:
        
        M = len(start_days)
        
    X = np.zeros((batch_size, L, C))
    y = np.zeros((batch_size, P))
    
    i = 0 # tracks index
    k = 0 # tracks batch size
    while True:
            
        i = start_days[np.random.randint(0, M)]
        X[k] = data[i+learn_window]
        y[k] = data[i+Lcap+predict_window, 0] ## USD-EUR Close is index 0
        
        k += 1
        
        if k == batch_size:
            k = 0
            yield X, y
    

def generate_all_regression(data, batch_size, learn_window, predict_window, start_days=[]):
    """
    Deterministic version of generate_batches_regression. 
    """
    
    learn_window = np.array(learn_window)
    predict_window = np.array(predict_window)
    
    N, C = data.shape
    L, P = len(learn_window), len(predict_window)
    Lcap, Pcap = int(np.max(learn_window))+1, int(np.max(predict_window))+1
    
    M = 0 # size of start_days
    if len(start_days) == 0:
    
        M = N - Lcap - Pcap 
        start_days = np.arange(M)
        
    else:
        
        M = len(start_days)
        
    X = np.zeros((batch_size, L, C))
    y = np.zeros((batch_size, P))
    
    k = 0 # tracks batch size
    for j in range(M):
            
        i = start_days[j]
        X[k] = data[i+learn_window]
        y[k] = data[i+Lcap+predict_window, 0] ## USD-EUR Close is index 0
        
        k += 1
        
        if k == batch_size:
            k = 0
            yield X, y
    
        
def generate_batches_accuracy(data, batch_size, learn_window, seed=None):
    """
    Randomly generate batches for accuracy. Here y contains 1 if next close
    value is higher, or 0 if next close value is lower.
    
    returns: generator that yields a tuple (X, y)
    - X: shape = (batch_size, len(learn_window), C)
    - y: shape = (batch_size, 1)
    """
    
    if seed != None:
        np.random.seed(seed)
        
    learn_window = np.array(learn_window)
    
    N, C = data.shape
    L, P = len(learn_window), 1
    Lcap, Pcap = int(np.max(learn_window))+1, 1
    
    X = np.zeros((batch_size, L, C))
    y = np.zeros((batch_size, P))
    
    i = 0 # tracks index
    k = 0 # tracks batch size
    M = N - Lcap - Pcap # ceiling for i
    while True:
        
        if k == batch_size:
            k = 0
            yield X, y
            
        i = np.random.randint(0, M)
        X[k] = data[i+learn_window]
        y[k, 0] = data[i+Lcap-1, 0] < data[i+Lcap, 0] ## USD-EUR Close is index 0
        
        k += 1
        
def test_rnn_helper():
    
    data = np.arange(1000).reshape(-1, 1)
        
    print("generate_batches_regression...")
    G = generate_batches_regression(data, 3, np.arange(10), np.arange(5))
    X, y = next(G)
    for i in range(3):
        print(i, "X:", X[i, :, 0])
        print(i, "y:", y[i, :])
    
    print("get_all_regression...")
    X, y = get_all_regression(data, np.arange(10), np.arange(5), [0, 100, 200])
    for i in range(3):
        print(i, "X:", X[i, :, 0])
        print(i, "y:", y[i, :])