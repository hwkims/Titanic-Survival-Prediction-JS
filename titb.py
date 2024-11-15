import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset, DataLoader

# 데이터 로드
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# 데이터 전처리 함수
def preprocess_data(df):
    # 결측값 처리: 나이(Age), 요금(Fare), 탑승항구(Embarked)
    imputer = SimpleImputer(strategy='median')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Fare'] = imputer.fit_transform(df[['Fare']])
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 'Cabin' 정보를 첫 문자만 사용하여 처리
    df['Cabin'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')

    # 범주형 변수 인코딩
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    df['Cabin'] = le.fit_transform(df['Cabin'])

    # Title 추출
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 4, 'Rev': 5, 'Col': 6, 'Mlle': 7, 'Countess': 8,
                 'Lady': 8, 'Don': 9, 'Sir': 9, 'Jonkheer': 10, 'Dona': 10, 'Mme': 2, 'Capt': 6, 'Major': 6, 'Ms': 1,
                 'Prof': 4}
    df['Title'] = df['Title'].map(title_map).fillna(0)

    # FamilySize와 IsAlone
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 새로운 특성 추가: 가족 규모와 혼자 여부
    df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

    # Feature 선택
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'Cabin']
    return df[features]


# 데이터 전처리
X = preprocess_data(train_df)
y = train_df['Survived']
X_test = preprocess_data(test_df)

# 3. StandardScaler로 데이터 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# 4. 학습 데이터와 검증 데이터로 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 5. PyTorch Dataset 클래스 정의
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 데이터 로더 생성
train_dataset = TitanicDataset(X_train, y_train)
val_dataset = TitanicDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 6. 모델 정의 (Deep Feedforward NN)
class TitanicModel(nn.Module):
    def __init__(self, input_dim):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.batch_norm3(self.fc3(x)))
        x = self.fc4(x)
        return self.sigmoid(x).squeeze()


# 7. 모델 생성
model = TitanicModel(X_train.shape[1])

# 8. 손실 함수와 최적화 함수 설정
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 9. 학습 및 검증 함수
def train_model(model, train_loader, val_loader, epochs=50):
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_preds.extend(outputs.numpy().flatten())
                val_labels.extend(labels.numpy().flatten())

        # 정확도 계산
        val_preds = np.array(val_preds) > 0.5
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(
            f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # 성능이 가장 좋은 모델 저장
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training complete.")
    return model


# 10. 모델 훈련
trained_model = train_model(model, train_loader, val_loader, epochs=50)

# 11. 테스트 데이터 예측
trained_model.eval()
with torch.no_grad():
    test_preds = trained_model(torch.tensor(X_test, dtype=torch.float32))
test_preds = (test_preds.numpy().flatten() > 0.5).astype(int)

# 12. 제출 파일 생성
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': test_preds})
submission.to_csv('submission_pytorch_improved.csv', index=False)
print("최종 제출 파일 'submission_pytorch_improved.csv' 생성 완료.")
