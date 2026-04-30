import subprocess
import sys
import os

def run_training(phase):
    """Запуск обучения для указанной фазы (comodgan или migan)"""
    experiment_name = f"{phase}_mirflickr512"
    
    print(f"\n{'='*60}")
    print(f"Запуск обучения: {phase.upper()}")
    print(f"Эксперимент: {experiment_name}")
    print(f"{'='*60}\n")
    
    cmd = [sys.executable, "main.py", "--experiment", experiment_name]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nОшибка при обучении {phase}!")
        return False
    
    return True

if __name__ == "__main__":
    # Этап 1: Обучение ComodGAN
    if not run_training("comodgan"):
        print("\nОбучение ComodGAN завершилось с ошибкой. Остановка.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("ComodGAN обучение завершено!")
    print("="*60 + "\n")
    
    # Этап 2: Обучение MI-GAN
    if not run_training("migan"):
        print("\nОбучение MI-GAN завершилось с ошибкой. Остановка.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("MI-GAN обучение завершено!")
    print("Все этапы обучения успешно выполнены!")
    print("="*60 + "\n")
