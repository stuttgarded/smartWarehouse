import logging
from flask import jsonify

from app.models.insight_model import EDA, Preprocessing, load_excel, save_file


class InsightController: 
    @staticmethod
    def process(request): 
        try: 
            if 'file_penjualan' not in request.files not in request.files:
                return jsonify({'success': False, 'error': 'File penjualan tidak ditemukan'}), 400

            file_penjualan = request.files['file_penjualan']

            if file_penjualan.filename == '':
                return jsonify({'success': False, 'error': 'File tidak dipilih'}), 400

            # Simpan File
            file_path_penjualan = save_file(file_penjualan, 'uploads/sales')

            # Load file Excel 
            df_penjualan = load_excel(file_path_penjualan)

            # Preprocess data penjualan
            sales_result = EDA.analyze_sales(df_penjualan)

            return jsonify(sales_result)

        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return jsonify({'success': False, 'error': 'Server sedang tidak dapat diakses'})
