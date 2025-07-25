import logging
import os
import pandas as pd
from werkzeug.utils import secure_filename

UPLOAD_FOLDER_SALES = 'uploads/sales'

os.makedirs(UPLOAD_FOLDER_SALES, exist_ok=True)

# Fungsi untuk menyimpan file ke folder
def save_file(file, folder):
    filename = secure_filename(file.filename)
    file_path = os.path.join(folder, filename)
    file.save(file_path)
    return file_path

def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"File {file_path} berhasil dihapus")
    except Exception as e:
        logging.error(f"Gagal menghapus file {file_path}: {str(e)}")
        
def load_excel(file_path):
    return pd.read_excel(file_path)


class Preprocessing:
    @staticmethod
    def clean_data(df, column_mapping):
        # Rename columns based on column mapping
        renamed_columns = {v: k for k, v in column_mapping.items()}
        df_renamed = df[list(renamed_columns.keys())].rename(columns=renamed_columns)

        # Fill missing values with mean (for numerical columns) or mode (for categorical columns)
        for column in df_renamed.columns:
            if df_renamed[column].dtype in ['float64', 'int64']:
                df_renamed[column].fillna(df_renamed[column].mean(), inplace=True)
            else:
                df_renamed[column].fillna(df_renamed[column].mode()[0], inplace=True)
        return df_renamed

class EDA:
    def rata_rata_penjualan_per_hari(df_last_3_months):
        # Count the unique days in the dataset
        unique_days = df_last_3_months['Tanggal Transaksi'].dt.date.nunique()

        # Calculate total sales
        total_sales = (df_last_3_months['Kuantitas Produk'] * df_last_3_months['Harga Produk']).sum()

        # Calculate average sales per day
        average_sales_per_day = total_sales / unique_days
        return average_sales_per_day

    def rata_rata_nilai_penjualan_per_transaksi(df_last_3_months):
        # Count the number of transactions
        num_transactions = df_last_3_months.shape[0]

        # Calculate total sales
        total_sales = (df_last_3_months['Kuantitas Produk'] * df_last_3_months['Harga Produk']).sum()

        # Calculate average sales per transaction
        average_sales_per_transaction = total_sales / num_transactions
        return average_sales_per_transaction

    def hari_paling_ramai(df_last_3_months):
        # Group by day and count the number of transactions
        daily_transactions = df_last_3_months.groupby(df_last_3_months['Tanggal Transaksi'].dt.date).size()

        # Find the day with the most transactions
        most_crowded_day = daily_transactions.idxmax()
        return most_crowded_day
    
    def jam_paling_ramai(df_last_3_months):
        # Extract the hour from the transaction time
        df_last_3_months['hour'] = df_last_3_months['Tanggal Transaksi'].dt.hour

        # Group by hour and count the number of transactions
        hourly_transactions = df_last_3_months.groupby('hour').size()

        # Find the hour with the most transactions
        most_crowded_hour = hourly_transactions.idxmax()
        return most_crowded_hour

    def analyze_sales(df):
        required_columns = [
            'Tanggal Transaksi',
            'Kuantitas Produk',
            'Harga Produk',
            'Nama Produk'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        df['Tanggal Transaksi'] = pd.to_datetime(df['Tanggal Transaksi'], errors='coerce')
        df['Kuantitas Produk'] = pd.to_numeric(df['Kuantitas Produk'], errors='coerce')
        df['Harga Produk'] = pd.to_numeric(df['Harga Produk'], errors='coerce')

        # Filter last one months
        latest_date = df['Tanggal Transaksi'].max()
        if latest_date.day >= 15:
            one_months_prior = (latest_date - pd.DateOffset(months=2)).replace(day=1)
        else:
            one_months_prior = (latest_date - pd.DateOffset(months=3)).replace(day=1)

        df_last_3_months = df[df['Tanggal Transaksi'] >= one_months_prior]

        total_produk_terjual = int(df_last_3_months['Kuantitas Produk'].sum())
        total_omzet = int((df_last_3_months['Kuantitas Produk'] * df_last_3_months['Harga Produk']).sum())

        list_produk_kontribusi_omzet = (
            (df_last_3_months['Kuantitas Produk'] * df_last_3_months['Harga Produk'])
            .groupby(df_last_3_months['Nama Produk'])
            .sum()
        ).to_dict()

        sorted_produk_list_kontribusi_omzet = sorted(list_produk_kontribusi_omzet.items(), key=lambda item: item[1], reverse=True)
        top_7_sorted_produk_list = [
            {
                "Nama Produk": item[0], 
                "kontribusi": item[1]
            } 
            for item in sorted_produk_list_kontribusi_omzet[:7]
        ]
        
        latest_date = df_last_3_months['Tanggal Transaksi'].max()
        if latest_date.day >= 15:
            # Jika tanggalnya lebih dari 15, hitung bulan ini
            previous_month = (latest_date - pd.DateOffset(months=0)).replace(day=1)
        else:
            # Jika tanggalnya sebelum 15, hitung bulan sebelumnya
            previous_month = (latest_date - pd.DateOffset(months=1)).replace(day=1)
        
        df_last_month = df[df['Tanggal Transaksi'].dt.month == previous_month.month-1]
        # Menghitung total omzet untuk bulan terakhir yang sudah disesuaikan
        total_omzet_last_month = (df_last_month['Kuantitas Produk'] * df_last_month['Harga Produk']).sum()
        total_omzet_last_month = int(total_omzet_last_month)

        df_last_3_months = df_last_3_months.copy()  # atau gunakan .loc jika tidak ingin membuat salinan
        df_last_3_months['bulan'] = df_last_3_months['Tanggal Transaksi'].dt.month
        df_last_3_months['tahun'] = df_last_3_months['Tanggal Transaksi'].dt.year
        
        list_omzet_per_bulan = df_last_3_months.groupby(['tahun', 'bulan']).apply(
            lambda x: (x['Kuantitas Produk'] * x['Harga Produk']).sum()
        ).reset_index(name='omzet')

        nama_bulan_indonesia = {
            1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
            7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember"
        }
        bulan_last_month = nama_bulan_indonesia[previous_month.month-1]
        
        total_omzet_last_month = {
            'omzet': total_omzet_last_month,
            'bulan': bulan_last_month
        }
        
        list_omzet_per_bulan['bulan'] = list_omzet_per_bulan['bulan'].apply(lambda x: nama_bulan_indonesia[x])

        top_3_omzet_per_bulan = list_omzet_per_bulan.to_dict(orient='records')

        list_total_terjual_per_produk = df_last_3_months.groupby('Nama Produk')['Kuantitas Produk'].sum().reset_index(
            name='total_terjual').sort_values(by='total_terjual', ascending=False).to_dict(orient='records')

        top_10_highest_total_terjual = list_total_terjual_per_produk[:10]
        top_10_lowest_total_terjual = list_total_terjual_per_produk[-10:]

        list_total_terjual_per_produk_exclude_bonus = df_last_3_months[
            df_last_3_months['Harga Produk'] > 0
        ].groupby('Nama Produk')['Kuantitas Produk'].sum().reset_index(
            name='total_terjual').sort_values(by='total_terjual', ascending=False).to_dict(orient='records')

        top_10_highest_total_terjual_exclude_bonus = list_total_terjual_per_produk_exclude_bonus[:10]
        
        processed_file_path = "uploads/sales/processed_sales.xlsx"  # atau path lain sesuai kebutuhan
        with pd.ExcelWriter(processed_file_path, engine='openpyxl') as writer:
            # Sheet kontribusi omzet
            df_sorted_produk_kontribusi_omzet = pd.DataFrame(sorted_produk_list_kontribusi_omzet, columns=['Nama Produk', 'kontribusi_omzet'])
            df_sorted_produk_kontribusi_omzet.to_excel(writer, sheet_name='Kontribusi Omzet', index=False)
            
            # Sheet omzet per bulan
            df_list_omzet_per_bulan = pd.DataFrame(list_omzet_per_bulan)
            df_list_omzet_per_bulan.to_excel(writer, sheet_name='Omzet Per Bulan', index=False)
            
            # Sheet total terjual per produk
            df_list_total_terjual_per_produk = pd.DataFrame(list_total_terjual_per_produk)
            df_list_total_terjual_per_produk.to_excel(writer, sheet_name='Total Terjual Per Produk', index=False)


        # Calculate Average Sales Per Day
        rata_rata_penjualan_per_hari = EDA.rata_rata_penjualan_per_hari(df_last_3_months)
        
        # Calculate Average Sales Per Transaction
        rata_rata_nilai_penjualan_per_transaksi = EDA.rata_rata_nilai_penjualan_per_transaksi(df_last_3_months)
        
        # Calculate Most Crowded Day
        hari_paling_ramai = EDA.hari_paling_ramai(df_last_3_months)
        
        # Calculate Most Crowded Hour
        jam_paling_ramai = EDA.jam_paling_ramai(df_last_3_months)

        return {
            'total_produk_terjual': int(total_produk_terjual),
            'total_omzet': int(total_omzet),
            'top_10_highest_total_terjual': top_10_highest_total_terjual,
            'top_10_lowest_total_terjual': top_10_lowest_total_terjual,
            'top_10_highest_total_terjual_exclude_bonus': top_10_highest_total_terjual_exclude_bonus,
            'top_3_omzet_per_bulan': top_3_omzet_per_bulan,
            'rata_rata_penjualan_per_hari': int(rata_rata_penjualan_per_hari),
            'rata_rata_nilai_penjualan_per_transaksi': int(rata_rata_nilai_penjualan_per_transaksi),
            'hari_paling_ramai': str(hari_paling_ramai),
            'jam_paling_ramai': str(jam_paling_ramai)
        }


