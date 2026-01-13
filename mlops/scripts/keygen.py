"""
ClearML Access Key Generator
=============================
ClearML Agent 인증에 사용할 랜덤 키를 생성합니다.

사용법:
    python keygen.py

생성된 키를 .env 파일에 복사하세요.
"""

import secrets
import string

def generate_key(length: int = 32) -> str:
    """안전한 랜덤 키 생성"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def main():
    print("=" * 60)
    print("🔐 ClearML Access Key Generator")
    print("=" * 60)
    print()
    
    # Access Key 생성 (짧은 식별자)
    access_key = generate_key(24)
    
    # Secret Key 생성 (긴 비밀 키)
    secret_key = generate_key(48)
    
    print("📋 아래 값들을 .env 파일에 복사하세요:")
    print()
    print(f"CLEARML_AGENT_ACCESS_KEY={access_key}")
    print(f"CLEARML_AGENT_SECRET_KEY={secret_key}")
    print()
    print("=" * 60)
    print()
    
    # .env 파일 직접 업데이트 옵션
    update = input("🔧 .env 파일을 자동으로 업데이트할까요? (y/n): ").strip().lower()
    
    if update == 'y':
        try:
            with open('.env', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 기존 키 교체
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if line.startswith('CLEARML_AGENT_ACCESS_KEY='):
                    new_lines.append(f'CLEARML_AGENT_ACCESS_KEY={access_key}')
                elif line.startswith('CLEARML_AGENT_SECRET_KEY='):
                    new_lines.append(f'CLEARML_AGENT_SECRET_KEY={secret_key}')
                else:
                    new_lines.append(line)
            
            with open('.env', 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
            
            print("✅ .env 파일이 업데이트되었습니다!")
            print()
            print("⚠️  변경사항을 적용하려면 Docker 컨테이너를 재시작하세요:")
            print("    docker-compose down && docker-compose up -d")
            
        except FileNotFoundError:
            print("❌ .env 파일을 찾을 수 없습니다. 수동으로 복사해주세요.")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    else:
        print("위의 키 값들을 수동으로 .env 파일에 복사해주세요.")


if __name__ == "__main__":
    main()
