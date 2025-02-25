import os
import glob
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import os
from assets.config import PERSON_DET , SOURCE_DIR, PRED_DIR, LABEL_DIR

def evaluate_images(model = PERSON_DET):
    """
    이미지 평가를 수행하고 예측 결과를 저장하는 함수
    """
    model = YOLO(model)
    os.makedirs(PRED_DIR, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(LABEL_DIR, "*.csv"))
    print(f"{len(csv_files)}개의 CSV 파일을 찾았습니다.")
    
    for csv_file in csv_files:
        folder_name = os.path.basename(csv_file).replace('.csv', '')
        try:
            df = pd.read_csv(csv_file)
            print(f"  CSV 파일 로드됨: {csv_file} (행 수: {len(df)})")
        except Exception as e:
            print(f"  CSV 파일 로드 오류: {e}")
            continue
        
        image_folder = os.path.join(SOURCE_DIR , folder_name)
        if not os.path.exists(image_folder):
            # VL_로 시작하면 VS_로 변환
            if folder_name.startswith('VL_'):
                alt_folder_name = 'VS_' + folder_name[3:]
                alt_image_folder = os.path.join(SOURCE_DIR, alt_folder_name)
                if os.path.exists(alt_image_folder):
                    image_folder = alt_image_folder
                    print(f"  VL_을 VS_로 변환하여 폴더 찾음: {alt_folder_name}")
            # VS_로 시작하면 VL_로 변환
            elif folder_name.startswith('VS_'):
                alt_folder_name = 'VL_' + folder_name[3:]
                alt_image_folder = os.path.join(SOURCE_DIR, alt_folder_name)
                if os.path.exists(alt_image_folder):
                    image_folder = alt_image_folder
                    print(f"  VS_를 VL_로 변환하여 폴더 찾음: {alt_folder_name}")
        
        if not os.path.exists(image_folder):
            print(f"  이미지 폴더를 찾을 수 없음: {image_folder}")
            continue
        
        df['pred'] = np.nan
        
        processed_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="이미지 처리 중"):
            image_name = row['image_name']
            image_path = os.path.join(image_folder, image_name)
            
            if not os.path.exists(image_path):
                print(f"  이미지를 찾을 수 없음: {image_path}")
                continue
            try:

                results = model.predict(image_path, conf=0.15, classes=[0])
                for result in results:
                    count = result.boxes.shape[0]
                    df.at[idx, 'pred'] = count
                processed_count += 1
            except Exception as e:
                print(f"  이미지 처리 오류 ({image_name}): {e}")
        
        output_file = os.path.join(PRED_DIR, os.path.basename(csv_file))
        df.to_csv(output_file, index=False)
        print(f"  처리 완료: {processed_count}/{len(df)} 이미지, 결과 저장됨: {output_file}")
    print("\n모든 이미지 처리 및 예측이 완료되었습니다.")

