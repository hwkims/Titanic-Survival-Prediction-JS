import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('train.csv')

# 데이터의 첫 몇 줄을 확인
print(data.head())

# 데이터 요약 통계 확인
print(data.describe())

# 데이터의 결측값 확인
print(data.isnull().sum())

# 기본적인 시각화 설정
sns.set(style="whitegrid")

# 1. 성별에 따른 생존율 시각화
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Survived', hue='Sex')
plt.title('Survival Counts by Gender')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()

# 2. 승객 등급(Pclass)과 생존율의 관계 시각화
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Pclass', hue='Survived')
plt.title('Survival Counts by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# 3. 나이(Age) 구간에 따른 생존율 시각화
# 나이를 구간으로 나누기
data['Age_bin'] = pd.cut(data['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], right=False)

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Age_bin', hue='Survived')
plt.title('Survival Counts by Age Range')
plt.xlabel('Age Range')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 4. 생존 여부에 따른 승객의 요금(Fare) 분포 시각화
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='Survived', y='Fare')
plt.title('Fare Distribution by Survival Status')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Fare')
plt.show()

# 5. 가족 관계에 따른 생존율 분석 (SibSp + Parch = 가족 크기)
data['FamilySize'] = data['SibSp'] + data['Parch']

plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='FamilySize', hue='Survived')
plt.title('Survival Counts by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Count')
plt.show()

# 6. 승선한 항구(Embarked)와 생존율 시각화
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Embarked', hue='Survived')
plt.title('Survival Counts by Embarked Port')
plt.xlabel('Embarked Port')
plt.ylabel('Count')
plt.show()

# 7. Age와 Fare의 관계를 시각화하여 생존 여부에 따른 차이를 비교
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Age', y='Fare', hue='Survived', palette='coolwarm', alpha=0.6)
plt.title('Age vs Fare by Survival Status')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()
