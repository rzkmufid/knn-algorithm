import streamlit as st  
import pandas as pd  
from sklearn.preprocessing import MinMaxScaler  
  
def read_file(file):  
    try:  
        if file.type == "text/csv":  
            return pd.read_csv(file)  
        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":  
            return pd.read_excel(file)  
        else:  
            return None  
    except Exception as e:  
        st.error(f"Terjadi kesalahan saat membaca file: {e}")  
        return None  
  
def clean_data(data):  
    try:  
        data['Harga'] = data['Harga'].replace('[\$,]', '', regex=True).astype(float)  
        return data  
    except Exception as e:  
        st.error(f"Terjadi kesalahan saat membersihkan data: {e}")  
        return None  
  
def display_data_with_pagination(data, page_size=50, key_prefix=None):  
    # Jika tidak ada key_prefix yang diberikan, buat satu yang unik  
    if key_prefix is None:  
        key_prefix = str(uuid.uuid4())  
  
    total_rows = data.shape[0]  
    num_pages = total_rows // page_size + (1 if total_rows % page_size > 0 else 0)  
    page_number = st.number_input(f'{key_prefix}Page Number', min_value=1, max_value=num_pages, step=1, value=1, key=f"{key_prefix}_page_number")  
    start_row = (page_number - 1) * page_size  
    end_row = min(start_row + page_size, total_rows)  
    st.write(f"Menampilkan baris {start_row} hingga {end_row} dari total {total_rows}")  
    st.dataframe(data.iloc[start_row:end_row], use_container_width=True)  
