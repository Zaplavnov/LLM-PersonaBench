import json
import os
from datetime import datetime
from pathlib import Path

def save_log(log_data, results_dir, name_file):
    """Сохраняет лог в файл"""
    log_file = results_dir / name_file
    # Создаем директорию, если её нет
    os.makedirs(results_dir, exist_ok=True)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)


def save_compact_result(
    result_payload: dict,
    results_dir: str | Path,
    exp_name: str,
    *,
    save_raw: bool = False,
    raw_payload: dict | None = None,
) -> Path:
    """Save compact experiment output and optional raw artifacts."""
    results_dir = Path(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = exp_name.replace(" ", "_")
    compact_path = results_dir / f"{timestamp}_{safe_name}.json"

    with open(compact_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False, default=str)

    if save_raw and raw_payload is not None:
        raw_path = results_dir / f"{timestamp}_{safe_name}_raw.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_payload, f, indent=2, ensure_ascii=False, default=str)

    return compact_path
