import os
import json
import pandas as pd
import glob
from assets.config import LABEL_EXTRACT_DIR, LABEL_DIR 

def setting_extract_label():
    os.makedirs(LABEL_DIR , exist_ok=True)
    json_folders = [folder for folder in os.listdir(LABEL_EXTRACT_DIR) if folder.endswith("_json")]

    for folder in json_folders:
        data = []
        folder_path = os.path.join(LABEL_EXTRACT_DIR, folder)
        json_files = glob.glob(os.path.join(folder_path, "*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8-sig') as f:
                    json_data = json.load(f)
                
                # image가 리스트인 경우
                image_data = json_data.get('image', [])
                if isinstance(image_data, list):
                    for img in image_data:
                        image_name = img.get('imagename', '')
                        counting = img.get('crowdinfo', {}).get('counting', 0)
                        
                        data.append({
                            'image_name': image_name,
                            'crowd_counting': counting
                        })
                else:
                    # 기존 구조인 경우
                    image_name = json_data.get('image', {}).get('imagename', '')
                    counting = json_data.get('image', {}).get('crowdinfo', {}).get('counting', 0)
                    
                    data.append({
                        'image_name': image_name,
                        'crowd_counting': counting
                    })
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

        df = pd.DataFrame(data)
        csv_filename = folder.replace("_json", "") + ".csv"
        output_path = os.path.join(LABEL_DIR , csv_filename)
        
        df.to_csv(output_path, index=False)
        
        print(f"Processed {len(data)} files from {folder} and saved to {output_path}")

    print("All JSON folders have been processed and CSV files created in the 'label' directory.")



    