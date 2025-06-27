"""
환경 테스트를 위한 간단한 스크립트입니다.
이 스크립트는 Python 환경이 제대로 설정되어 있는지 확인합니다.
"""

import sys
import platform
import pkg_resources

def check_python_version():
    """Python 버전을 확인합니다."""
    print("\n=== Python 버전 확인 ===")
    print(f"Python 버전: {platform.python_version()}")
    print(f"실행 경로: {sys.executable}")

def check_installed_packages():
    """필요한 패키지가 설치되어 있는지 확인합니다."""
    print("\n=== 설치된 패키지 확인 ===")
    required = ['pandas', 'numpy', 'backtrader', 'matplotlib']
    
    for package in required:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"{package}: 설치됨 (버전: {version})")
        except pkg_resources.DistributionNotFound:
            print(f"{package}: 설치되지 않음")

def main():
    print("=== 환경 테스트 시작 ===")
    check_python_version()
    check_installed_packages()
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    main()
