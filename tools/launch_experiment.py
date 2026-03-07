import os
import sys
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path

"""
Скрипт запуска экспериментов по оптимизации промтов
Использование: python tools/launch_experiment.py --config=experiments/cluster0_ga.yaml
"""

# Добавляем корневую директорию проекта в путь для импортов
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_config(config_path):
    """Загружаем конфиг из YAML файла"""
    if not Path(config_path).is_absolute():
        if not config_path.startswith('configs/'):
            config_path = Path("configs") / config_path # Проверяем, начинается ли путь с configs/
        else:
            config_path = Path(config_path)
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config['config_file'] = str(config_path)
    exp_name = Path(config['config_file']).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['experiment_id'] = f"{exp_name}_{timestamp}"
    
    return config

def setup_results_dir(config):
    """Создаем папку для результатов эксперимента"""
    experiment_id = config['experiment_id']
    
    results_dir = project_root / "results_experiments" / experiment_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем полный конфиг в директорию результатов
    with open(results_dir / "config.json", "w", encoding='utf-8') as f:
        json.dump(config, f, indent=2, default=str, ensure_ascii=False)
    
    config['results_dir'] = str(results_dir)
    return config

def main():
    """Основная функция запуска эксперимента"""
    parser = argparse.ArgumentParser(
        description="Запуск экспериментов по оптимизации промтов"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Путь к конфигурационному файлу (относительно configs/ или полный путь)'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Загрузка конфигурации из: {args.config}")
        config = load_config(args.config)
        
        print("Создание директории для результатов...")
        config = setup_results_dir(config)
        print(f"Директория результатов: {config['results_dir']}")
        
        # Импортируем и запускаем эксперимент по режиму
        mode = str((config.get('experiment') or {}).get('mode', 'ga')).lower()
        if mode == "hype":
            from src.simulator.person_opt_hype import run_experiment
        else:
            from src.simulator.person_type_opt import run_experiment

        print("Запуск эксперимента")
        run_experiment(config)
        
        print("Эксперимент завершен успешно!")
        
    except Exception as e:
        print(f"Ошибка при выполнении эксперимента: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
