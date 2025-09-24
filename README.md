# FlaskCryptoTrader

**Flask, Celery, Docker 기반의 End-to-End 암호화폐 자동매매 백엔드 시스템**

이 프로젝트는 실시간 시세 데이터 수집부터 ML 모델 기반의 예측, 자동 주문 실행, 리스크 관리에 이르는 전체 자동매매 사이클을 구현한 백엔드 시스템입니다. 안정적인 운영과 확정성을 목표로 설계되었으며, MLOps와 백엔드 엔지니어링의 모범 사례를 적용했습니다.

## 주요 기능 (Features)

### 데이터 파이프라인
- **주기적 데이터 수집**: Celery Beat를 이용해 15분/1시간 단위의 시세 데이터를 자동으로 수집합니다. ([`app/tasks/__init__.py`](app/tasks/__init__.py))
- **데이터 무결성 보장**: PostgreSQL의 `ON CONFLICT DO NOTHING`과 복합 기본 키(`symbol`, `timestamp`, `timeframe`)를 활용하여 데이터의 중복 저장을 방지하고 멱등성을 확보합니다. ([`app/tasks/data_task.py`](app/tasks/data_task.py), [`app/models.py`](app/models.py))
- **지능적인 결측치 관리**: 누락된 데이터 구간을 자동으로 탐지([`find_data_gaps`](app/tasks/data_task.py))하고, API 호출 비용을 줄이기 위해 데이터가 없는 것으로 확인된 구간을 Redis에 캐싱하여 불필요한 요청을 방지합니다. ([`_add_empty_gap_to_cache`](app/tasks/data_task.py))
- **대용량 데이터 처리**: 과거 데이터의 대량 수집을 위한 비동기 작업을 제공하며, API를 통해 진행 상황을 모니터링할 수 있습니다. ([`collect_bulk_historical_data`](app/tasks/data_task.py))

### ML & 트레이딩 로직
- **피쳐 엔지니어링**: TA-Lib을 활용하여 RSI, MACD, 볼린저 밴드 등 수십 개의 기술적 지표와 다중 시간대(Multi-Timeframe) 데이터를 결합한 피쳐를 생성합니다. ([`add_technical_indicators`](app/trading/feature_engineering.py))
- **시장 상황 분석 (Market Regime)**: ADX, ATR, BBW 등의 지표를 종합하여 현재 시장을 '추세(Trending)' 또는 '횡보(Ranging)' 국면으로 판단하고, 이에 맞는 모델과 전략을 적용합니다. ([`update_market_regime`](app/tasks/market_analysis_task.py))
- **앙상블 모델 서빙**: 시장 국면에 따라 LightGBM, XGBoost, CatBoost 모델의 예측을 앙상블하여 예측 안정성을 높입니다. ([`ModelHandler`](app/ml/model_handler.py))
- **뉴스 감성 분석**: Gemini LLM을 활용해 최신 뉴스의 감성 점수를 추출하고, 이를 트레이딩 결정의 보조 지표로 활용합니다. ([`update_news_sentiment`](app/tasks/sentiment_analysis_task.py))
- **자동화된 모델 재학습**: Celery Beat를 통해 주기적으로 최신 데이터로 모델을 자동 재학습하는 파이프라인을 구축했습니다. ([`trigger_model_retraining`](app/tasks/ml_task.py))

### 리스크 관리 및 주문 실행
- **자동 주문 실행**: AI 예측 신호와 여러 단계의 필터링 로직([`filters.py`](app/trading/filters.py))을 통과하면, ATR 기반의 동적 손절 라인과 포지션 사이즈를 계산하여 자동으로 브라켓 주문(Bracket Order)을 실행합니다. ([`run_trading_cycle_for_symbol`](app/trading/execution_engine.py))
- **동적 포지션 관리**: 1R 수익 도달 시 본절(Break-even) 이동, 부분 익절, 트레일링 스탑(Trailing Stop) 등 진입 이후의 포지션 관리를 자동화합니다. ([`monitor_positions_and_risk`](app/tasks/risk_management_task.py))
- **서킷 브레이커**: 일일 최대 손실률(-3%) 도달 시 모든 포지션을 청산하고 신규 거래를 중단하여 계좌를 보호합니다. ([`monitor_positions_and_risk`](app/tasks/risk_management_task.py))

### 아키텍처 및 운영
- **컨테이너 기반 환경**: Docker와 Docker Compose를 통해 개발 및 배포 환경을 표준화하여 "내 컴퓨터에선 됐는데" 문제를 원천 차단합니다. ([`Dockerfile`](Dockerfile), [`docker-compose.yml`](docker-compose.yml))
- **비동기 작업 처리**: 데이터 수집, 모델 학습 등 시간이 오래 걸리는 작업을 Celery와 Redis를 통해 비동기적으로 처리하여 API 응답성을 확보합니다. ([`celery_app.py`](app/tasks/celery_app.py))
- **관측 가능성(Observability)**: Prometheus를 통한 핵심 지표(AI 예측 수, 데이터 처리량 등) 모니터링, JSON 포맷의 구조화된 로깅, Grafana 대시보드를 통해 시스템 상태를 실시간으로 파악합니다. ([`metrics.py`](app/metrics.py), [`logging_config.py`](app/logging_config.py))
- **API 기반 제어**: 시스템의 주요 기능을 제어하고 상태를 조회할 수 있는 RESTful API를 제공합니다. ([`app/api/__init__.py`](app/api/__init__.py))

## 아키텍처



## API 엔드포인트
- `POST /api/collect-historical-data`: 과거 데이터 대량 수집
- `POST /api/train`: 모델 재학습 파이프라인 실행
- `POST /api/trigger-trade-cycle`: AI 트레이딩 사이클 수동 실행
- `GET /api/system-status`: 시스템 전반의 상태 조회
- `GET /api/health`, `/api/ready`: 서비스 헬스 체크
