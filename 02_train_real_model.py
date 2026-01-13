import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, '03_Model_Input', 'real_paper_db.csv')
MODEL_DIR = os.path.join(BASE_DIR, '04_Trained_Model')
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

class ExpertAI(nn.Module):
    def __init__(self):
        super(ExpertAI, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 2) # Mobility, Stability
        )
    def forward(self, x): return self.net(x)

def train_model():
    if not os.path.exists(DATA_PATH): return print("❌ 데이터 파일 없음. Step 1 실행 요망.")

    print("🚀 [Training] 데이터 증강 및 AI 학습 시작...")
    df = pd.read_csv(DATA_PATH)
    
    # 데이터 증강 (Augmentation)
    aug_data = []
    for _ in range(50): # 50배 뻥튀기
        for _, row in df.iterrows():
            # 입력 변수에 노이즈 추가
            n_in = np.clip(row['In'] + np.random.normal(0, 0.02), 0, 1)
            n_ga = np.clip(row['Ga'] + np.random.normal(0, 0.02), 0, 1)
            n_zn = np.clip(row['Zn'] + np.random.normal(0, 0.02), 0, 1)
            n_sn = np.clip(row['Sn'] + np.random.normal(0, 0.02), 0, 1)
            n_temp = np.clip(row['Temp'] + np.random.normal(0, 5), 100, 500)
            
            # 물리적 경향성 반영 (In 증가 -> Mob 증가)
            d_in = n_in - row['In']
            n_mob = max(0, row['Mobility'] * (1 + d_in * 1.5) + np.random.normal(0, 1))
            n_stab = max(0, row['Stability'] + np.random.normal(0, 0.05))
            
            aug_data.append([n_in, n_ga, n_zn, n_sn, n_temp, n_mob, n_stab])
            
    df_aug = pd.DataFrame(aug_data, columns=['In','Ga','Zn','Sn','Temp','Mobility','Stability'])
    
    # 학습
    X = df_aug[['In','Ga','Zn','Sn','Temp']].values
    y = df_aug[['Mobility','Stability']].values
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y)
    
    model = ExpertAI()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(500):
        pred = model(torch.tensor(X_s, dtype=torch.float32))
        loss = criterion(pred, torch.tensor(y_s, dtype=torch.float32))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 100 == 0: print(f"Epoch {epoch} Loss: {loss.item():.5f}")
            
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'real_model.pth'))
    joblib.dump(scaler_X, os.path.join(MODEL_DIR, 'scaler_X_real.pkl'))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'scaler_y_real.pkl'))
    print("✅ [Complete] AI 학습 완료.")

if __name__ == "__main__":
    train_model()