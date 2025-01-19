from sklearn.preprocessing import MinMaxScaler  
from sklearn.metrics import euclidean_distances  
import pandas as pd  
  
def knn_predict(data, test_data, k):  
    scaler = MinMaxScaler()  
    normalized_data = pd.DataFrame(scaler.fit_transform(data[['Terjual', 'Harga', 'Stok']]), columns=['Terjual', 'Harga', 'Stok'])  
    normalized_test_data = scaler.transform(test_data)  
      
    distances = euclidean_distances(normalized_data, normalized_test_data)  
    data['Distance'] = distances[:, 0]  
    nearest_neighbors = data.nsmallest(k, 'Distance')  
      
    prediction = nearest_neighbors['Label'].mode()[0]  
    total_sales = nearest_neighbors['Terjual'].sum()  
    avg_sales = nearest_neighbors['Terjual'].mean()  
      
    return prediction, total_sales, avg_sales, nearest_neighbors  
