import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 데이터 로드
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 데이터 전처리 함수
def preprocess_data(df):
    # Age 결측값 처리 (Pclass와 Sex에 따른 중앙값으로 채우기)
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
    df['Embarked'] = df.groupby('Pclass')['Embarked'].transform(lambda x: x.fillna(x.mode()[0]))

    # Cabin 결측값 처리 및 첫 문자로 대체 (U는 Unknown)
    df['Cabin'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')

    # Sex, Embarked, Cabin 인코딩
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    df['Cabin'] = le.fit_transform(df['Cabin'])

    # Title 추출 및 인코딩
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 4, 'Rev': 5, 'Col': 6, 'Mlle': 7, 'Countess': 8,
                 'Lady': 8, 'Don': 9, 'Sir': 9, 'Jonkheer': 10, 'Dona': 10, 'Mme': 2, 'Capt': 6, 'Major': 6, 'Ms': 1,
                 'Prof': 4}
    df['Title'] = df['Title'].map(title_map).fillna(0)

    # FamilySize와 IsAlone
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FamilyType'] = df['FamilySize'].apply(lambda x: 'Solo' if x == 1 else ('Small' if x <= 4 else 'Large'))

    # FamilyType 인코딩
    df['FamilyType'] = le.fit_transform(df['FamilyType'])

    # Feature 선택
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'FamilyType']
    return df[features]

# 데이터 전처리
X = preprocess_data(train_df)
y = train_df['Survived']
X_test = preprocess_data(test_df)

# StandardScaler로 데이터 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# 학습 데이터와 검증 데이터로 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 텐서플로 모델 정의 함수 (하이퍼파라미터 튜닝 적용)
def create_model(input_dim, layers=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # 이진 분류
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 모델 생성 (하이퍼파라미터 튜닝)
model = create_model(X_train.shape[1], layers=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001)

# EarlyStopping 설정 (검증 정확도가 더 이상 향상되지 않으면 훈련을 중단)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 훈련
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# 모델 평가
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# 앙상블을 위한 다른 모델들 생성 (모델 튜닝)
model_2 = create_model(X_train.shape[1], layers=[256, 128, 64], dropout_rate=0.4, learning_rate=0.0005)
model_3 = create_model(X_train.shape[1], layers=[64, 32], dropout_rate=0.5, learning_rate=0.001)

# 모델 훈련
model_2.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
model_3.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# 앙상블 예측 (각 모델의 예측을 평균)
y_pred_1 = model.predict(X_test)
y_pred_2 = model_2.predict(X_test)
y_pred_3 = model_3.predict(X_test)

# 예측값 평균내기 (앙상블 기법)
y_pred_ensemble = (y_pred_1 + y_pred_2 + y_pred_3) / 3
y_pred_ensemble = (y_pred_ensemble > 0.5).astype(int)  # 이진 분류 예측 후 0, 1로 변환

# 제출 파일 생성
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred_ensemble.flatten()})
submission.to_csv('submission_ensemble.csv', index=False)

print(f"최종 제출 파일 'submission_ensemble.csv' 생성 완료.")
