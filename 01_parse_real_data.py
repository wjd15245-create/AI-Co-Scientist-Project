import json
import os
import pandas as pd
import glob
import re

# ---------------------------------------------------------
# [설정] 경로
# ---------------------------------------------------------
BASE_DIR = os.getcwd()
JSON_DIR = os.path.join(BASE_DIR, '02_JSON_Data')
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Model_Input')

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def clean_paper_id(paper_id, filename):
    """
    파일명이 난해할 경우(예: d3tc01379k.pdf), 깔끔한 ID로 변환
    """
    # 1. 기본 정리 (확장자 제거)
    clean_name = str(paper_id).replace('.pdf', '').replace('.json', '')
    
    # 2. 의미 없는 해시값/코드인 경우 (길이가 짧거나 숫자가 너무 많음)
    # 예: d3tc01379k -> Ref_Batch05_d3tc
    if len(clean_name) < 5 or not re.search(r'[a-zA-Z]', clean_name):
        clean_name = f"Ref_{filename.replace('.json','')}_{clean_name[:6]}"
    
    # 3. 특수문자 제거 (언더바 제외)
    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", clean_name)
    
    return clean_name

def parse_composition(comp_str):
    elements = {'In': 0.0, 'Ga': 0.0, 'Zn': 0.0, 'Sn': 0.0}
    if not comp_str or comp_str == 'null': return elements
    
    try:
        # 비율 숫자 추출
        ratios = re.findall(r"[\d\.]+", str(comp_str))
        ratios = [float(x) for x in ratios]
        
        # 원소 확인
        comp_upper = str(comp_str).upper()
        active = []
        if 'IN' in comp_upper: active.append('In')
        if 'GA' in comp_upper: active.append('Ga')
        if 'ZN' in comp_upper: active.append('Zn')
        if 'SN' in comp_upper: active.append('Sn')
        
        # 비율 할당 (정규화)
        total = sum(ratios)
        if total > 0 and len(ratios) >= len(active) and active:
            for i, el in enumerate(active):
                if i < len(ratios): elements[el] = ratios[i] / total
        elif active:
            # 비율 없으면 균등 분배
            for el in active: elements[el] = 1.0 / len(active)
    except: pass
    
    return elements

def load_and_parse():
    print("🔄 [ETL] 데이터 클리닝 및 파싱 시작...")
    all_data = []
    
    json_files = glob.glob(os.path.join(JSON_DIR, '**', '*.json'), recursive=True)
    print(f"📄 발견된 파일: {len(json_files)}개")

    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                items = content.get('fullContent', content) if isinstance(content, dict) else content
                if not isinstance(items, list): items = [items]

                for item in items:
                    # [핵심] ID 클리닝 적용
                    raw_id = item.get('Paper_ID', 'Unknown')
                    fname = os.path.basename(filepath)
                    clean_id = clean_paper_id(raw_id, fname)
                    
                    # 데이터 파싱
                    mat = item.get('Material_Data', {})
                    comp_vals = parse_composition(mat.get('Composition_Ratio'))
                    
                    proc = item.get('Process_Data', {})
                    temp = proc.get('Anneal_Temp_C')
                    try: temp = float(temp) if temp and temp != 'null' else 300.0
                    except: temp = 300.0
                    
                    perf = item.get('Performance_Data', {})
                    mob = perf.get('Mobility')
                    try: mob = float(mob) if mob and mob != 'null' else 0.0
                    except: mob = 0.0
                    
                    stab_str = str(perf.get('PBTS_Shift', '0.0'))
                    try: stab = float(re.findall(r"[-+]?\d*\.\d+|\d+", stab_str)[0])
                    except: stab = 1.0
                    
                    logic = item.get('Physics_Logic', {}).get('Mechanism', 'No mechanism info.')
                    
                    if mob > 0:
                        all_data.append({
                            'Paper_ID': clean_id,
                            'In': comp_vals['In'], 'Ga': comp_vals['Ga'], 
                            'Zn': comp_vals['Zn'], 'Sn': comp_vals['Sn'],
                            'Temp': temp,
                            'Mobility': mob, 'Stability': abs(stab),
                            'Mechanism': logic
                        })
        except Exception as e:
            print(f"⚠️ Skip: {os.path.basename(filepath)} ({e})")

    if all_data:
        df = pd.DataFrame(all_data)
        # 결측치 0으로 채우기
        df = df.fillna(0)
        
        save_path = os.path.join(OUTPUT_DIR, 'real_paper_db.csv')
        df.to_csv(save_path, index=False)
        print(f"✅ [Success] 총 {len(df)}개 데이터 정제 완료. (저장: {save_path})")
    else:
        print("❌ 유효한 데이터가 없습니다.")

if __name__ == "__main__":
    load_and_parse()