#!/bin/bash

# 파이썬 버전 및 가상 환경 경로 설정
PYTHON_VERSION="python3.8"
VIRTUALENV_PATH="./"

# 가상 환경 생성
$PYTHON_VERSION -m pip install virtualenv
$PYTHON_VERSION -m virtualenv $VIRTUALENV_PATH

# 가상 환경 활성화
source $VIRTUALENV_PATH/bin/activate

# 필요한 패키지 설치
pip install -r requirements.txt

# 스크립트 실행
python3 your_script.py

# 가상 환경 비활성화
deactivate
