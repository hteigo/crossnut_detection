import ultralytics
import os
import sys
import time
import cv2
import numpy as np
import traceback
import pandas as pd

from utils.zip_folder import zip_folder
from utils.find_cross import find_cross
from utils.find_label import find_label
from utils.find_cross_point import find_cross_point
from utils.find_anchor import find_label_anchor, find_number_anchor
from PIL import Image, ImageDraw, ImageFont
from tkinter import Tk, filedialog

if __name__ == '__main__':
    cross_model = ultralytics.YOLO('model/cross_best.pt')
    label_model = ultralytics.YOLO('model/find_label_best.pt')
    number_model = ultralytics.YOLO('model/number_detection_best.pt')

    #選擇圖片資料夾
    root = Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title="請選擇圖片資料夾")
    print("選擇的資料夾是：", input_dir)

    # 使用拖曳的方式取得圖片資料夾
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = './pics/colored_1' # default input image directory
    # input_dir = './pics/data'   # input image directory, change as your wish
    # input_dir = './pics/1140430-gov2'   # input image directory, change as your wish
    # input_dir = './pics/colored_1'   # input image directory, change as your wish
    output_dir = './output/colored_1' # output image directory
    result_dir = './result' # result image directory
    successed_dir = os.path.join(result_dir, "successed")
    failed_dir = os.path.join(result_dir, "failed")
    warning_dir = os.path.join(result_dir, "warning")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(successed_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    os.makedirs(warning_dir, exist_ok=True)



    # Caculate inference time
    results = []  # 存每張圖片的處理結果
    start_time = time.time()

    for image_file in os.listdir(input_dir):
        try:
            if not image_file.endswith(('.jpg', '.png', '.jpeg', '.webp')):
                continue
            
            print(f"Processing image: {image_file}")
            image_name = image_file.split('.')[0]
            
            ### find cross image from original image
            img_path = os.path.join(input_dir, image_file)
            img = cv2.imread(img_path)
            cross_img = find_cross(img, img_name=image_name, output_dir=output_dir, model=cross_model)

            # no cross found
            if cross_img is None:
                fail_save_path = os.path.join(failed_dir, image_file)
                cv2.imwrite(fail_save_path, img)
                results.append({
                "filename": image_file,
                "result(cm)": "failed"
                })

                print(f"No cross found in image {image_file}.")
                continue

            # cross_img, cross_point, straight_line = temp
            # print(f"Image: {image_file}")
            # print(f"new Cross point: ({cross_point[0]}, {cross_point[1]})")
            
            ### find label nums and positions
            temp = find_label(cross_img, img_name=image_name, output_dir=output_dir, model=label_model)
            label_count, label_centers, label_confs, label_imgs = temp
            if label_count > 5:
                warning = 1
            else:
                warning = 0
            
            print(f"Number of labels: {label_count}")

            ### find cross point
            cross_x, cross_y = find_cross_point(cross_img, image_name, output_dir)
            print(f"Cross point: ({cross_x}, {cross_y})")
                
            labels_dir = os.path.join(output_dir, 'labels', image_name)
            anchors = find_label_anchor(label_imgs, label_centers, label_confs, labels_dir=labels_dir)
                
            if anchors.shape[0] <= 2: # can't use interpolation (插值)
                number_anchor = find_number_anchor(cross_img, output_dir, image_name, number_model)
                anchors = np.concatenate((anchors, number_anchor), axis=0)
                warning = 1
                
            ### 建立擬合模型
            poly = np.poly1d(np.polyfit(anchors[:, 1], anchors[:, 0], 2))
            rpoly = round(poly(cross_y),1)
            # 要顯示的文字
            text = f" {rpoly}"

            # 將 cross_img 轉換為 PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font_path = "C:/Windows/Fonts/msjh.ttc" 
            font_size = 160  
            font = ImageFont.truetype(font_path, font_size)

            # 計算文字的邊界框
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            img_width, img_height = pil_img.size
            x = img_width - text_width - 160
            y = img_height - text_height - 160
            # 如果文字超出圖片範圍，則調整位置
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            # 如果文字超出圖片範圍，則調整位置
            if x + text_width > img_width:
                x = img_width - text_width
            if y + text_height > img_height:
                y = img_height - text_height
            # 畫文字
            margin = 60  # 黑底與文字的邊距
            rect_x0 = x - margin
            rect_y0 = y - margin + 80
            rect_x1 = x + text_width + margin
            rect_y1 = y + text_height + margin

            draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill="black")
            draw.text((x, y), text, font=font, fill="white")
            if warning == 1:
                save_path = os.path.join(warning_dir, f"{image_name}.jpg")
                pil_img.save(save_path)
                results.append({
                "filename": image_file,
                "result(cm)": f"{rpoly} (Warning)"
                })
            else:
                save_path = os.path.join(successed_dir, f"{image_name}.jpg")
                pil_img.save(save_path)
                results.append({
                "filename": image_file,
                "result(cm)": rpoly
                })
            # 將 PIL Image 轉回 OpenCV 格式
            print("Estimate height: ", rpoly)
            print()

            

        except Exception as e:
            fail_save_path = os.path.join(failed_dir, image_file)
            cv2.imwrite(fail_save_path, img)
            results.append({
            "filename": image_file,
            "result(cm)": "failed"
            })

            print(f"Error processing image {image_file}: {e}")
            print(traceback.print_exc())
            continue

    # Caculate inference time
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.2f} seconds")
    df = pd.DataFrame(results)
    csv_path = os.path.join(result_dir, "results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig") 

# 壓縮成功圖片
success_zip_path = os.path.join(result_dir, "successed.zip")
zip_folder(successed_dir, success_zip_path)

# 壓縮失敗圖片
failed_zip_path = os.path.join(result_dir, "failed.zip")
zip_folder(failed_dir, failed_zip_path)

# 壓縮警告圖片
warning_zip_path = os.path.join(result_dir, "warning.zip")
zip_folder(warning_dir, warning_zip_path)

