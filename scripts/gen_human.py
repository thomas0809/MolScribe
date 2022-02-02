import os
import numpy as np
import pandas as pd
import xlsxwriter
import cv2

pred_path = 'output/uspto/swin_base_aux_1m_new/prediction_valko.csv'
data_path = 'data/molbank/Img2Mol/valko.csv'

pred_df = pd.read_csv(pred_path)
data_df = pd.read_csv(data_path)

workbook = xlsxwriter.Workbook(pred_path.replace('.csv', '.xlsx'))
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'image_id')
worksheet.write(0, 1, 'image')
worksheet.write(0, 2, 'post_SMILES')
worksheet.write(0, 3, 'molblock')

os.makedirs('tmp', exist_ok=True)
worksheet.set_column_pixels(1, 1, 520)
worksheet.set_column_pixels(2, 3, 320)
cell_format = workbook.add_format()
cell_format.set_text_wrap()

for i, row in pred_df.iterrows():
    worksheet.set_row_pixels(i+1, 520)
    worksheet.write(i+1, 0, row['image_id'])
    img_path = 'data/molbank/' + data_df.loc[i, 'file_path']
    tmp_path = 'tmp/' + row['image_id'] + '.png'
    img = cv2.imread(img_path)
    cv2.imwrite(tmp_path, img)
    worksheet.insert_image(i+1, 1, tmp_path, {'x_scale': 2, 'y_scale': 2, 'x_offset': 3, 'y_offset': 3})
    worksheet.write(i+1, 2, row['post_SMILES'])
    if type(row['molblock']) is str:
        worksheet.write(i+1, 3, row['molblock'], cell_format)

workbook.close()
