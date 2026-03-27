# network-traffic.ipynb 네트워크 부하 분석 


## 한눈에 보는 결론

이 노트북은 8개의 ISCX CSV 파일을 합쳐서 공통으로 등장하는 `Destination Port`만 남긴 뒤, 5개의 트래픽 특성을 기준으로 네트워크 부하를 해석하려는 분석 노트북이다.

분석은 크게 3가지 축으로 나뉜다.

- CCA로 트래픽 특성과 목적지 포트 간의 구조적 관계를 본다.
- XGBoost로 `Destination Port == 443` 여부를 `load`로 간주해 이진 분류를 수행한다.
- TableOne으로 `load=0`과 `load=1` 그룹의 기술통계를 비교한다.

즉, 이 노트북은 "트래픽 수치들만 보고 특정 포트(특히 443)를 높은 부하 상태로 볼 수 있는가"를 상관분석, 분류모델, 통계요약으로 동시에 확인하려는 구성이다.

## 1차 파악: 데이터 구성과 전처리

### 사용 파일

노트북은 아래 8개 CSV를 입력으로 사용한다.

- `Mon_ISCX.csv`
- `Tue_ISCX.csv`
- `Wed_ISCX.csv`
- `Thu(1)_ISCX.csv`
- `Thu(2)_ISCX.csv`
- `Fri(1)_ISCX.csv`
- `Fri(2)_ISCX.csv`
- `Fri(3)_ISCX.csv`

### 공통 처리 방식

- 각 CSV를 `pandas.read_csv()`로 읽는다.
- 컬럼명의 앞뒤 공백을 `str.strip()`으로 제거한다.
- 각 파일의 `Destination Port` 집합을 구한다.
- 8개 파일 전체에 공통으로 존재하는 포트만 남긴다.
- 필요한 컬럼만 선택하고 결측치를 제거한다.

노트북에 저장된 출력 기준으로, 8개 파일에서 공통으로 잡힌 `Destination Port`의 개수는 `2178개`다.

### 실제로 사용하는 특징량

노트북은 매우 많은 네트워크 컬럼을 쓰지 않고, 아래 5개 특징만 핵심 입력으로 고정한다.

- `Flow Duration`
- `Total Fwd Packets`
- `Total Length of Fwd Packets`
- `Total Length of Bwd Packets`
- `Fwd Packet Length Mean`

이 선택은 전체 노트북에서 일관되게 유지된다. 즉, 분석 방향이 "복잡한 전체 피처 엔지니어링"보다는 "몇 개의 대표 트래픽 지표로 포트/부하를 설명"하는 쪽에 가깝다.

## 2차 파악: CCA 분석이 하는 일

### 목적

CCA 셀은 트래픽 특성 집합 `X`와 목적지 포트 `Y` 사이의 관계를 1개의 정준 변수로 압축해서 본다.

- `X`: 위의 5개 트래픽 특징
- `Y`: `Destination Port`

코드는 `StandardScaler`로 `X`와 `Y`를 각각 표준화한 뒤, `CCA(n_components=1)`를 적용한다.

### 처리 흐름

이 셀의 작업 순서는 아래와 같다.

1. 8개 CSV를 읽고 공통 포트를 계산한다.
2. 모든 데이터를 합친다.
3. 공통 포트에 해당하는 행만 남긴다.
4. 5개 특징과 `Destination Port`만 선택한다.
5. 결측치를 제거한다.
6. `X`와 `Y`를 각각 표준화한다.
7. CCA로 정준 변수 `X_c`, `Y_c`를 만든다.
8. 정준 변수 간 상관과 회귀선을 산점도로 시각화한다.
9. Pearson, Spearman, Kendall 상관분석 결과를 CSV로 저장한다.

### 저장 산출물

이 셀은 아래 결과물을 디스크에 저장하도록 작성돼 있다.

- `./CCA/CCA_Scatter_Traffic_vs_Port.png`
- `./CCA/correlation.csv`

### 노트북에 저장된 실제 결과

출력에 저장된 상관계수는 아래와 같다.

- Pearson correlation: `0.124261618487111`
- Spearman correlation: `-0.3169122563327456`
- Kendall correlation: `-0.23871847109708755`
- 각 p-value: 모두 `0.0`으로 출력됨

### 해석

이 결과는 다음처럼 읽는 것이 적절하다.

- 선형 상관(Pearson)은 약한 양의 관계만 보인다.
- 순위 기반 상관(Spearman, Kendall)은 음의 방향으로 나타난다.
- 즉, 단순 직선 관계 하나로 설명되는 구조는 아니고, 포트와 트래픽 특징 사이 관계가 비선형적이거나 단조 관계가 복합적으로 섞여 있을 가능성이 있다.

다만 중요한 주의점이 있다.

- `Destination Port`는 본질적으로 범주형 의미가 강한 값인데, 여기서는 숫자형 변수처럼 표준화해서 CCA에 넣고 있다.
- 따라서 "포트 번호의 크기 자체"를 해석하는 분석이라기보다, "포트 번호를 숫자로 간주했을 때 트래픽 특징과 어떤 수치적 연관이 보이는가" 정도로 보는 것이 안전하다.

## 3차 파악: XGBoost + SHAP + LIME 분석이 하는 일

### 목적

두 번째 핵심 셀은 분류 문제를 만든다.

여기서 `Destination Port == 443`이면 `load = 1`, 아니면 `load = 0`으로 두고, 5개 트래픽 특징만으로 이 값을 예측하는 XGBoost 모델을 학습한다.

즉, 이 노트북은 "443 포트 트래픽을 높은 부하 상태의 대리 지표(proxy)로 보고, 이를 트래픽 특징으로 분류할 수 있는가"를 실험한다.

### 처리 흐름

1. 8개 CSV를 읽고 공통 포트를 구한다.
2. 데이터를 모두 합친 뒤 공통 포트만 남긴다.
3. 5개 특징과 `Destination Port`를 선택하고 결측치를 제거한다.
4. `Destination Port == 443`를 `load = 1`로 라벨링한다.
5. `train_test_split(test_size=0.2, stratify=y, random_state=42)`로 분할한다.
6. `XGBClassifier`를 학습한다.
7. `classification_report`로 성능을 평가한다.
8. SHAP으로 전역 중요도와 샘플별 기여를 해석한다.
9. LIME으로 첫 번째 테스트 샘플을 국소 해석한다.

### 노트북에 저장된 실제 분류 성능

`classification_report` 출력은 아래와 같다.

- 클래스 `0`: precision `0.97`, recall `0.97`, f1-score `0.97`, support `347203`
- 클래스 `1`: precision `0.90`, recall `0.89`, f1-score `0.90`, support `101142`
- 전체 accuracy: `0.95`
- macro avg f1-score: `0.93`
- weighted avg f1-score: `0.95`

이 테스트셋 크기는 `448345`건이며, `test_size=0.2` 기준으로 보면 전체 데이터는 약 `2241722`건 수준으로 해석된다. 이 값은 TableOne 출력의 전체 표본 수와 사실상 일치한다.

### 해석

이 결과만 놓고 보면, 선택한 5개 트래픽 특징만으로도 `443 포트 여부`를 꽤 높은 정확도로 구분하고 있다.

특히 볼 수 있는 점은 아래와 같다.

- 클래스 불균형이 있지만 `stratify=y`로 분할해 비율은 유지했다.
- 다수 클래스(0) 성능은 매우 높다.
- 소수 클래스(1)도 f1-score가 `0.90`으로 꽤 높은 편이다.
- 따라서 `443 포트 트래픽`은 이 5개 특징에서 어느 정도 분리 가능한 패턴을 가진다고 볼 수 있다.

### SHAP/LIME 해석 정보

SHAP은 아래 3가지 시각화를 만든다.

- 중요도 막대 그래프
- 전체 분포 요약 그래프
- 0번째 샘플의 waterfall plot

LIME은 첫 번째 테스트 샘플(`i = 0`)을 설명한다. 노트북 HTML 출력에 저장된 내용 기준으로, 이 샘플에 대해 사용된 값과 기여 방향은 아래와 같다.

- `Fwd Packet Length Mean = 41.78` 는 음의 기여 `-0.14235601146425628`
- `Flow Duration = 117399319.00` 는 양의 기여 `0.1404992120787622`
- `Total Fwd Packets = 23.00` 는 음의 기여 `-0.07694276529824326`
- `Total Length of Fwd Packets = 961.00` 는 양의 기여 `0.06282708247704262`
- `Total Length of Bwd Packets = 5232.00` 는 음의 기여 `-0.036292402433256585`

이 샘플 설명은 "예측이 한두 개 변수로만 결정되지 않고, 양의 기여와 음의 기여가 동시에 작용한 합성 결과"라는 점을 보여준다.

## 4차 파악: TableOne 요약 통계가 하는 일

### 목적

세 번째 핵심 셀은 `load=0`과 `load=1` 그룹의 기술통계 차이를 요약한다.

여기서도 `load`는 동일하게 `Destination Port == 443` 여부로 생성된다.

### 처리 흐름

1. 8개 CSV를 읽는다.
2. `Destination Port`를 정수형으로 변환한다.
3. 공통 포트를 계산한다.
4. 5개 특징과 `Destination Port`만 남기고 결측치를 제거한다.
5. 모든 날짜 데이터를 합친다.
6. 공통 포트 기준으로 다시 필터링한다.
7. `load` 변수를 만든다.
8. `TableOne(..., categorical=['load'], groupby='load', pval=True)`로 그룹 비교표를 만든다.
9. 결과를 텍스트 파일로 저장한다.

### 저장 산출물

- `./TableOne_Summary/AllDays_tableone_summary.txt`

### 노트북에 저장된 실제 통계 결과

전체 표본 수는 `2241722`건이다.

- `load=0`: `1736012`건, `77.4%`
- `load=1`: `505710`건, `22.6%`

주요 평균(표준편차)은 아래와 같다.

- `Flow Duration`
  Overall: `18194508.8 (36565761.1)`
  load=0: `15169852.3 (33733479.9)`
  load=1: `28577613.5 (43374456.7)`
  p-value: `<0.001`

- `Total Fwd Packets`
  Overall: `11.3 (842.3)`
  load=0: `9.6 (954.5)`
  load=1: `17.2 (132.0)`
  p-value: `<0.001`

- `Total Length of Fwd Packets`
  Overall: `545.9 (6132.8)`
  load=0: `304.2 (5532.5)`
  load=1: `1375.8 (7794.9)`
  p-value: `<0.001`

- `Total Length of Bwd Packets`
  Overall: `20370.6 (2543013.8)`
  load=0: `19531.8 (2882614.9)`
  load=1: `23250.0 (376521.3)`
  p-value: `0.099`

- `Fwd Packet Length Mean`
  Overall: `53.4 (110.0)`
  load=0: `47.4 (106.4)`
  load=1: `73.7 (119.4)`
  p-value: `<0.001`

### 해석

이 통계표가 말해주는 핵심은 아래와 같다.

- `load=1(443 포트)` 그룹은 전체적으로 더 긴 `Flow Duration`을 가진다.
- `load=1` 그룹은 전방향 패킷 수와 전방향 바이트 길이가 더 크다.
- `Fwd Packet Length Mean`도 `load=1` 그룹이 더 높다.
- 반면 `Total Length of Bwd Packets`는 평균 차이가 있어 보여도 p-value가 `0.099`라서, 이 노트북 기준에서는 통계적으로 유의하다고 보기 어렵다.

즉, 이 노트북은 "443 포트로 정의한 부하 상태는 특히 전방향 트래픽 특성에서 차이가 크다"는 메시지를 주고 있다.

## 노트북 전체 흐름을 한 문장씩 요약하면

- 첫 번째 코드 셀은 8개 파일에서 공통으로 나타나는 포트가 얼마나 되는지 확인한다.
- 두 번째 코드 셀은 CCA로 트래픽 특징과 목적지 포트의 구조적 관계를 본다.
- 세 번째 코드 셀은 XGBoost로 443 포트 여부를 분류하고 SHAP/LIME으로 해석한다.
- 네 번째 코드 셀은 TableOne으로 두 그룹의 통계 차이를 정리한다.

## 이 노트북의 핵심 메시지

이 노트북은 단순한 시각화 노트북이 아니라, 아래 세 층위를 한 번에 다룬다.

- 상관 구조 분석: CCA
- 예측 모델 분석: XGBoost
- 그룹 차이 통계 분석: TableOne

그리고 세 분석이 공통적으로 가리키는 방향은 다음과 같다.

- 선택된 5개 트래픽 특징은 443 포트 트래픽과 일반 포트 트래픽을 어느 정도 구분한다.
- 특히 `Flow Duration`, `Total Fwd Packets`, `Total Length of Fwd Packets`, `Fwd Packet Length Mean`가 중요한 구분 축으로 보인다.
- 즉, 네트워크 부하를 포트 기반으로 정의했을 때, 전방향 트래픽 특성이 주요 설명 변수로 작동한다.

## 주의할 점과 한계

README에 남겨둘 만한 중요한 한계도 분명하다.

- `load`의 정의가 실제 측정된 시스템 부하가 아니라 `Destination Port == 443` 여부다.
- 따라서 이 노트북은 "실제 서버 부하 예측"이라기보다 "443 포트 트래픽 구분"에 더 가깝다.
- `Destination Port`는 범주형 성격이 강한데 CCA에서는 연속형 숫자처럼 다뤄진다.
- 공통 포트만 남기는 방식은 날짜별로 특이하게 등장하는 포트를 제거하므로, 실제 운영환경의 희귀 트래픽은 놓칠 수 있다.
- SHAP 결과는 시각화로 저장돼 있지만, 코드상 별도 파일로 저장하지는 않는다.
- LIME 출력은 첫 번째 테스트 샘플 하나만 설명하므로 전체 데이터 해석으로 일반화하면 안 된다.
- 실행 출력에는 `xgboost` 파라미터 경고, `lime`의 `FutureWarning`, `tableone`의 `SettingWithCopyWarning`가 남아 있어 패키지 버전 의존성이 있다.

## 재실행 시 필요한 환경

코드 기준으로 필요한 주요 라이브러리는 아래와 같다.

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
- `xgboost`
- `shap`
- `lime`
- `tableone`

또한 노트북은 CSV 파일들이 노트북 실행 위치 기준 같은 디렉터리에 있다고 가정하고 상대경로로 읽는다.

현재 저장소 폴더에는 `README.md`와 `network-traffic.ipynb`만 보였고, CSV 원본은 확인되지 않았다. 따라서 재실행하려면 위 8개 CSV가 동일 경로에 준비돼 있어야 한다.
