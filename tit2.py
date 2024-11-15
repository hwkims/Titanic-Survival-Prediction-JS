import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.model_selection import ParameterGrid


# 데이터 전처리 함수 개선
def preprocess_data_improved(df):
    # 1. 결측값 처리
    # 'Age' 결측값을 Pclass와 Sex 그룹별 중앙값으로 채우기
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

    # 'Fare' 결측값을 Pclass별 중앙값으로 채우기
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

    # 'Embarked' 결측값을 Pclass별 최빈값으로 채우기
    df['Embarked'] = df.groupby('Pclass')['Embarked'].transform(lambda x: x.fillna(x.mode()[0]))

    # 2. 'Cabin' 처리 (첫 문자만 사용, 없으면 'U'로 대체)
    df['Cabin'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')

    # 3. 성별 인코딩
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])

    # 4. 'Embarked' 인코딩
    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

    # 5. 'Title' 처리 (승객 이름에서 직위를 추출하고, 여러 직위를 하나로 묶기)
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    title_map = {
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 4, 'Rev': 5, 'Col': 6, 'Mlle': 7, 'Countess': 8,
        'Lady': 8, 'Don': 9, 'Sir': 9, 'Jonkheer': 10, 'Dona': 10, 'Mme': 2, 'Capt': 6, 'Major': 6, 'Ms': 1,
        'Prof': 4, 'Miss': 1, 'Mrs': 2, 'Sir': 9
    }
    df['Title'] = df['Title'].map(title_map).fillna(0)

    # 6. 가족 크기 및 혼자 여부 특성 추가
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 7. 특성 선택
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    return df[features]


# 데이터 로드 및 전처리
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 트레인과 테스트 데이터를 각각 전처리
X_train = preprocess_data_improved(train)
y_train = train['Survived']
X_test = preprocess_data_improved(test)

# 특성 표준화 (정규화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 데이터 텐서로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)


# 개선된 TitanicNN 모델 정의
class ImprovedTitanicNN(nn.Module):
    def __init__(self, input_dim, hidden_units=256, hidden_layers=3, dropout_rate=0.3):
        super(ImprovedTitanicNN, self).__init__()
        layers = []
        in_units = input_dim

        # 숨겨진 레이어 추가
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_units, hidden_units))
            layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            in_units = hidden_units

        layers.append(nn.Linear(in_units, 2))  # 출력 레이어 (Survived: 0 or 1)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 하이퍼파라미터 후보들 설정
param_grid = {
    'epochs': [100, 150],
    'batch_size': [64, 128],
    'learning_rate': [0.001, 0.0005, 0.01],
    'dropout_rate': [0.3, 0.5],
    'hidden_units': [256, 512],  # FC 레이어의 유닛 수를 변경
    'hidden_layers': [3, 4],  # 숨겨진 레이어 수
}

# 파라미터 조합 생성
param_combinations = ParameterGrid(param_grid)

best_model = None
best_accuracy = 0.0
best_params = None

# 각 파라미터 조합에 대해 실험
for params in param_combinations:
    # 모델 초기화
    model = ImprovedTitanicNN(input_dim=X_train.shape[1], hidden_units=params['hidden_units'],
                              hidden_layers=params['hidden_layers'], dropout_rate=params['dropout_rate'])

    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # 손실 함수 설정
    criterion = nn.CrossEntropyLoss()

    # 데이터 로더 설정
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)

    # 훈련 루프
    for epoch in range(params['epochs']):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            # Gradients 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 역전파
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 예측 결과
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / total_preds

        print(f"Epoch [{epoch + 1}/{params['epochs']}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # 최적 모델 저장 (accuracy 기준)
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model = model
            best_params = params

# 최적 모델로 평가
model = best_model  # 최적 모델 선택
model.eval()

# 테스트 데이터로 예측
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# 결과 저장
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predicted.numpy()})
submission.to_csv('submission.csv', index=False)

# 최적 모델 저장
torch.save(model, 'best_model.pth')

print(f"최적 모델 저장 완료, 정확도: {best_accuracy:.4f} (하이퍼파라미터: {best_params})")