import streamlit as st  
from streamlit_option_menu import option_menu  
from utils.data_utils import read_file, clean_data, display_data_with_pagination  
from utils.knn_utils import knn_predict  
import uuid  
import pandas as pd  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.metrics import euclidean_distances  
from io import BytesIO  
  
st.set_page_config(layout="wide", page_title="Analisis KNN", page_icon="🔍")  
  
# Add logo in the sidebar  
st.sidebar.image('assets/Logo Klinik Gunuang.png', width=200, use_container_width=True)  
  
# Sidebar with option menu  
with st.sidebar:  
    selected_page = option_menu(  
        menu_title="Klinik Gunuang",  
        options=["Dashboard", "Data Training", "Data Uji", "Hitung Jarak Euclidean", "Perangkingan", "Hasil Prediksi"],  
        icons=["house", "file-earmark-spreadsheet", "file-earmark-text", "calculator", "graph-up-arrow", "check-circle"],  
        menu_icon="cast",  
        default_index=0,  
    )  
  
# Inisialisasi session state  
if 'data_train' not in st.session_state:  
    st.session_state['data_train'] = None  
if 'data_test' not in st.session_state:  
    st.session_state['data_test'] = None  
if 'train_file_name' not in st.session_state:  
    st.session_state['train_file_name'] = None  
if 'test_file_name' not in st.session_state:  
    st.session_state['test_file_name'] = None  
if 'features' not in st.session_state:  
    st.session_state['features'] = ['Terjual', 'Harga', 'Stok']  
if 'normalized_train_data' not in st.session_state:  
    st.session_state['normalized_train_data'] = None  
if 'normalized_test_data' not in st.session_state:  
    st.session_state['normalized_test_data'] = None  
if 'k' not in st.session_state:  
    st.session_state['k'] = 4  
if 'distances' not in st.session_state:  
    st.session_state['distances'] = None  
if 'neighbors' not in st.session_state:  
    st.session_state['neighbors'] = None  
if 'original_neighbors' not in st.session_state:  
    st.session_state['original_neighbors'] = None  # Menyimpan salinan asli data neighbors  
  
# Dashboard Page  
if selected_page == "Dashboard":  
    st.title("Dashboard")  
  
    st.write("Selamat datang di aplikasi Analisis KNN. Ikuti langkah-langkah berikut untuk melakukan prediksi penjualan:")  
    st.markdown("""  
    1. **Data Training**: Unggah dan normalisasi data training.  
    2. **Data Uji**: Tambahkan data uji secara manual atau unggah file. Normalisasi data, hitung jarak Euclidean, lakukan perangkingan, dan lihat hasil prediksi.  
    """)  
  
# Data Training Page  
elif selected_page == "Data Training":  
    st.title("Data Training")  
    if st.session_state.get('train_file_name'):  
        st.write(f"File yang telah diunggah: {st.session_state['train_file_name']}")  
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"], key="train_upload")  
    if uploaded_file is not None:  
        data_train = read_file(uploaded_file)  
        if data_train is not None:  
            st.session_state['train_file_name'] = uploaded_file.name  
            st.session_state['data_train'] = clean_data(data_train)  
            if st.session_state['data_train'] is not None:  
                st.write("Data Training:")  
                display_data_with_pagination(st.session_state['data_train'], key_prefix="train")  
                if st.button("Normalisasi Data Training"):  
                    scaler = MinMaxScaler()  
                    st.session_state['normalized_train_data'] = pd.DataFrame(scaler.fit_transform(st.session_state['data_train'][st.session_state['features']]), columns=st.session_state['features'])  
                    st.write("Data Training yang dinormalisasi:")  
                    display_data_with_pagination(st.session_state['normalized_train_data'], key_prefix="normalized_train")  
        else:  
            st.write("Format file tidak didukung. Harap unggah file CSV atau Excel.")  
    elif st.session_state.get('data_train') is not None:  
        st.write("Data Training:")  
        display_data_with_pagination(st.session_state['data_train'], key_prefix="train")  
        if st.session_state.get('normalized_train_data') is not None:  
            st.write("Data Training yang dinormalisasi:")  
            display_data_with_pagination(st.session_state['normalized_train_data'], key_prefix="normalized_train")  
  
# Data Uji Page  
elif selected_page == "Data Uji":  
    st.title("Data Uji")  
  
    # Upload file untuk data uji (Tetap di atas)  
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel untuk Data Uji atau tambahkan data secara manual", type=["csv", "xlsx"])  
      
    # Proses file yang diunggah  
    if uploaded_file is not None:  
        data_test = read_file(uploaded_file)  
        if data_test is not None:  
            st.session_state['uploaded_data_test'] = clean_data(data_test)  # Simpan data upload pertama kali  
  
            # Set 'data_test' jika belum ada, atau hanya update 'uploaded_data_test'  
            if 'data_test' not in st.session_state or st.session_state['data_test'] is None:  
                st.session_state['data_test'] = st.session_state['uploaded_data_test'].copy()  
            else:  
                # Jika data sudah ada, gabungkan data baru dari file jika tidak ada duplikasi  
                if st.session_state['uploaded_data_test'] is not None and st.session_state['data_test'] is not None:  
                    st.session_state['data_test'] = pd.concat([st.session_state['uploaded_data_test'], st.session_state['data_test']], ignore_index=True).drop_duplicates()  
  
            st.session_state['test_file_name'] = uploaded_file.name  
            st.success("Data berhasil diunggah!")  
  
    # Tampilkan form untuk pengisian manual data uji (Di bawah Select file)  
    st.subheader("Tambah Data Uji Secara Manual")  
    with st.form(key='manual_data_input'):  
        nama_obat = st.text_input("Nama Obat")  
        stok = st.number_input("Stok", min_value=0)  
        harga = st.number_input("Harga", min_value=0.0, format="%.2f")  
        terjual = st.number_input("Terjual", min_value=0)  
        submit_button = st.form_submit_button(label='Tambahkan Data Uji')  
  
    # Menyimpan data uji manual ke session state  
    if submit_button:  
        manual_data = pd.DataFrame({  
            'Nama Obat': [nama_obat],  
            'Stok': [stok],  
            'Harga': [harga],  
            'Terjual': [terjual]  
        })  
  
        # Tambahkan data manual tanpa mengulang data dari file  
        st.session_state['data_test'] = pd.concat([st.session_state['data_test'], manual_data], ignore_index=True).drop_duplicates()  
  
        st.success("Data berhasil ditambahkan!")  
  
    # Tampilkan data uji (Di bawah form)  
    if st.session_state.get('data_test') is not None:  
        st.write("Data Uji:")  
        display_data_with_pagination(st.session_state['data_test'], key_prefix=str(uuid.uuid4()))  
  
    # Button Normalisasi (Di bawah tabel)  
    if st.session_state.get('normalized_train_data') is not None:  
        if st.button("Normalisasi Data Uji"):  
            if st.session_state['data_test'] is not None and not st.session_state['data_test'].empty:  
                scaler = MinMaxScaler()  
                scaler.fit(st.session_state['data_train'][st.session_state['features']])  # Fit dengan data latih  
                # Normalisasi semua data yang ada di session state 'data_test'  
                st.session_state['normalized_test_data'] = pd.DataFrame(scaler.transform(st.session_state['data_test'][st.session_state['features']]), columns=st.session_state['features'])  
  
                # Setelah normalisasi, tampilkan tabel yang sudah dinormalisasi (Di bawah tombol normalisasi)  
                st.write("Data Uji yang dinormalisasi:")  
                display_data_with_pagination(st.session_state['normalized_test_data'], key_prefix=str(uuid.uuid4()))  
            else:  
                st.warning("Data Uji kosong, tidak bisa dinormalisasi.")  
    else:  
        st.write("Harap normalisasi data training terlebih dahulu.")  
  
# Hitung Jarak Euclidean Page  
elif selected_page == "Hitung Jarak Euclidean":  
    st.title("Hitung Jarak Euclidean")  
      
    if st.session_state.get('normalized_train_data') is not None and st.session_state.get('normalized_test_data') is not None:  
        data_for_knn = st.session_state['normalized_train_data']  
        test_data = st.session_state['normalized_test_data']  
          
        # Hitung semua jarak Euclidean  
        distances = euclidean_distances(data_for_knn, test_data)  
          
        # Simpan jarak Euclidean di data_train  
        st.session_state['distances'] = st.session_state['data_train'].copy()  
  
        # Dropdown untuk memilih data uji  
        selected_index = st.selectbox("Pilih data uji untuk melihat perhitungan jarak Euclidean:", range(len(st.session_state['data_test'])), format_func=lambda x: st.session_state['data_test'].iloc[x]['Nama Obat'])  
          
        # Update kolom Distance berdasarkan data yang dipilih  
        st.session_state['distances']['Distance'] = distances[:, selected_index]  
          
        # Tampilkan tabel dengan hasil jarak Euclidean untuk data yang dipilih  
        st.write(f"Hasil perhitungan jarak Euclidean untuk data uji ke-{selected_index + 1} ({st.session_state['data_test'].iloc[selected_index]['Nama Obat']}):")  
        display_data_with_pagination(st.session_state['distances'], key_prefix=str(uuid.uuid4()))  
    else:  
        st.write("Harap unggah dan normalisasi data training serta data uji terlebih dahulu.")  
  
# Perangkingan Page  
elif selected_page == "Perangkingan":  
    st.title("Perangkingan")  
      
    if st.session_state.get('distances') is not None:  
        data = st.session_state['distances']  
  
        # Dropdown untuk memilih data uji  
        selected_index = st.selectbox("Pilih data uji untuk melihat perangkingan jarak Euclidean:", range(len(st.session_state['data_test'])), format_func=lambda x: st.session_state['data_test'].iloc[x]['Nama Obat'])  
  
        # Perbarui jarak Euclidean berdasarkan data uji yang dipilih  
        data['Distance'] = euclidean_distances(st.session_state['normalized_train_data'], [st.session_state['normalized_test_data'].iloc[selected_index]])[:, 0]  
  
        k_input = st.number_input("Masukkan nilai k (jumlah tetangga terdekat)", min_value=1, max_value=len(data), value=4, step=1, key="k_value")  
        st.session_state['k'] = k_input  
          
        nearest_neighbors = data.nsmallest(st.session_state['k'], 'Distance')  
        st.session_state['neighbors'] = nearest_neighbors  
        st.session_state['original_neighbors'] = nearest_neighbors.copy()  # Simpan salinan asli untuk keamanan  
          
        st.write(f"{st.session_state['k']} Data Teratas Berdasarkan Jarak Euclidean untuk data uji ke-{selected_index + 1} ({st.session_state['data_test'].iloc[selected_index]['Nama Obat']}):")  
        display_data_with_pagination(nearest_neighbors, key_prefix=str(uuid.uuid4()))  
    else:  
        st.write("Harap hitung jarak Euclidean terlebih dahulu.")  
  
# Hasil Prediksi Page  
elif selected_page == "Hasil Prediksi":  
    st.title("Hasil Prediksi")  
      
    if st.session_state.get('normalized_test_data') is not None and st.session_state.get('normalized_train_data') is not None:  
        results = []  
        k = st.session_state['k']  # Gunakan nilai k yang sudah diatur di halaman perangkingan  
  
        # Lakukan perhitungan prediksi untuk setiap data uji  
        for idx in range(len(st.session_state['data_test'])):  
            # Ambil data uji yang sedang diproses  
            test_data = st.session_state['normalized_test_data'].iloc[[idx]]  
  
            # Hitung jarak Euclidean antara data uji dan data latih  
            distances = euclidean_distances(st.session_state['normalized_train_data'], test_data)  
  
            # Simpan jarak Euclidean dalam data_train  
            st.session_state['data_train']['Distance'] = distances[:, 0]  
  
            # Ambil k tetangga terdekat berdasarkan jarak Euclidean  
            nearest_neighbors = st.session_state['data_train'].nsmallest(k, 'Distance')  
  
            # Hitung rata-rata penjualan dari k tetangga terdekat  
            avg_sales = nearest_neighbors['Terjual'].mean()  
  
            # Prediksi label berdasarkan tetangga terdekat  
            label_status = nearest_neighbors['Label'].mode()[0]  
  
            # Simpan hasil untuk data uji ini  
            results.append({  
                'Nama Obat': st.session_state['data_test'].iloc[idx]['Nama Obat'],  
                'Penjualan Rata-rata': avg_sales,  
                'Label/Status': label_status  
            })  
  
        result_df = pd.DataFrame(results)  
        st.write("Hasil Prediksi:")  
        display_data_with_pagination(result_df, key_prefix=str(uuid.uuid4()))  
  
        # Tombol unduh hasil sebagai CSV  
        csv = result_df.to_csv(index=False)  
        st.download_button(label="Unduh Hasil sebagai CSV", data=csv, file_name='hasil_prediksi.csv', mime='text/csv')  
  
        # Tombol unduh hasil sebagai Excel  
        excel_buffer = BytesIO()  
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:  
            result_df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')  
          
        st.download_button(label="Unduh Hasil sebagai Excel", data=excel_buffer.getvalue(), file_name='hasil_prediksi.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')  
    else:  
        st.write("Harap lakukan perangkingan terlebih dahulu.")  
