from __future__ import annotations

import json
import random
import re
import subprocess
import time
import warnings
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.prompt.hype_persona_optimizer import optimize_persona_prompt_hype
from src.utils.config_loader import load_config
from src.utils.parse import parse_response
from src.utils.prompt import get_modifier_bisect, get_modifier_by_match
from src.utils.save_result import save_compact_result

SECTION_ORDER = ("SYSTEM", "INSTRUCTION", "OUTPUT_FORMAT")
SECTION_BLOCK_RE = re.compile(
    r"\[(SYSTEM|INSTRUCTION|OUTPUT_FORMAT)\]\s*(.*?)\s*\[/\1\]",
    flags=re.DOTALL,
)
PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")

PLACEHOLDERS = {
    "role": "{{ROLE_SECTION}}",
    "traits": "{{TRAITS_SECTION}}",
    "facets": "{{FACETS_SECTION}}",
    "critic": "{{CRITIC_SECTION}}",
    "questions": "{{QUESTIONS_SECTION}}",
}

# CoolPrompt внутри использует pydantic/openai-tooling и может шуметь нефатальными
# serializer warning'ами. Для чистого лога эксперимента скрываем только их.
warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic serializer warnings.*",
    category=UserWarning,
)

_CHAT_PROMPT_TEMPLATE = None
_PERSONALITY_HELPERS = None
_LC_MESSAGES = None


def _get_chat_prompt_template_cls():
    """Lazy import to avoid heavy langchain import at module load time."""
    global _CHAT_PROMPT_TEMPLATE
    if _CHAT_PROMPT_TEMPLATE is None:
        from langchain_core.prompts import ChatPromptTemplate

        _CHAT_PROMPT_TEMPLATE = ChatPromptTemplate
    return _CHAT_PROMPT_TEMPLATE


def _get_langchain_message_classes():
    """Lazy import plain message classes to bypass f-string template parsing."""
    global _LC_MESSAGES
    if _LC_MESSAGES is None:
        from langchain_core.messages import HumanMessage, SystemMessage

        _LC_MESSAGES = (SystemMessage, HumanMessage)
    return _LC_MESSAGES


def _get_personality_helpers():
    """Lazy import five-factor helpers used in person_type_opt/personality_match."""
    global _PERSONALITY_HELPERS
    if _PERSONALITY_HELPERS is None:
        from src.utils.personality_match import (
            OCEAN_AND_FACET_ORDER,
            aggregate_cluster_five_factor_metrics,
            compute_five_factor_metrics,
        )
        from src.utils import five_factor

        _PERSONALITY_HELPERS = {
            "OCEAN_AND_FACET_ORDER": OCEAN_AND_FACET_ORDER,
            "aggregate_cluster_five_factor_metrics": aggregate_cluster_five_factor_metrics,
            "compute_five_factor_metrics": compute_five_factor_metrics,
            "five_factor": five_factor,
        }
    return _PERSONALITY_HELPERS


class PromptFormatError(ValueError):
    pass


def _get_project_root() -> Path:
    file_path = Path(__file__).resolve()
    candidate = file_path.parent.parent.parent
    if (candidate / "src").exists():
        return candidate
    cwd = Path.cwd()
    current = cwd
    while current != current.parent:
        if (current / "src").exists():
            return current
        current = current.parent
    return cwd


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _get_project_root() / p


def _load_json(path: str) -> dict:
    with open(_resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_traits(config: dict) -> dict[int, dict[str, str]]:
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("traits_path")
    if path:
        data = _load_json(path)
        return {int(k): v for k, v in data.items() if str(k).isdigit()}
    from src.prompt.traits import traits

    return traits


def _load_facets(config: dict) -> dict[int, dict[str, str]]:
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("facets_path")
    if path:
        data = _load_json(path)
        return {int(k): v for k, v in data.items() if str(k).isdigit()}
    from src.prompt.facets import facets

    return facets


def _load_system(config: dict) -> dict:
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("system_path")
    if path:
        return _load_json(path)
    from src.prompt.system import system

    return system


def _load_trait_target_values(config: dict) -> dict[int, dict[str, float]]:
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("traits_path")
    if path:
        data = _load_json(path)
        raw = data.get("trait_target_values", {})
        return {int(k): v for k, v in raw.items() if str(k).isdigit()}
    from src.prompt.traits import trait_target_values

    return trait_target_values


def _load_facet_target_values(config: dict) -> dict[int, dict[str, float]]:
    prompt_cfg = config.get("prompt") or {}
    path = prompt_cfg.get("facets_path")
    if path:
        data = _load_json(path)
        raw = data.get("facet_target_values", {})
        return {int(k): v for k, v in raw.items() if str(k).isdigit()}
    from src.prompt.facets import facet_target_values

    return facet_target_values


def serialize_prompt(template_obj: dict[str, str]) -> str:
    parts = []
    for section in SECTION_ORDER:
        value = (template_obj.get(section.lower()) or "").strip()
        parts.append(f"[{section}]\n{value}\n[/{section}]")
    return "\n\n".join(parts)


def parse_prompt(prompt: str) -> dict[str, str]:
    matches = SECTION_BLOCK_RE.findall(prompt or "")
    parsed = {name.lower(): content.strip() for name, content in matches}
    if len(parsed) != len(SECTION_ORDER):
        missing = [s for s in SECTION_ORDER if s.lower() not in parsed]
        raise PromptFormatError(f"Missing prompt sections: {missing}")
    return parsed


def validate_prompt_format(
    prompt: str,
    *,
    required_placeholders: set[str] | None = None,
    preserve_sections: bool = True,
) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise PromptFormatError("Prompt must be a non-empty string")
    parsed = parse_prompt(prompt)
    if preserve_sections:
        for key, value in parsed.items():
            if not value.strip():
                raise PromptFormatError(f"Section {key!r} must be non-empty")
    if required_placeholders:
        found = {m.group(0) for m in PLACEHOLDER_RE.finditer(prompt)}
        missing = sorted(required_placeholders - found)
        if missing:
            raise PromptFormatError(f"Missing placeholders: {missing}")


def _build_prompt_template(system_cfg: dict) -> str:
    template_obj = {
        "system": (
            f"{PLACEHOLDERS['role']}\n\n"
            f"Your traits:\n{PLACEHOLDERS['traits']}\n\n"
            f"Your specific behavioral aspects:\n{PLACEHOLDERS['facets']}\n\n"
            f"Internal reflection guideline:\n{PLACEHOLDERS['critic']}"
        ),
        "instruction": f"{system_cfg['task']}\n\nQuestions:\n{PLACEHOLDERS['questions']}",
        "output_format": system_cfg["response_format"],
    }
    return serialize_prompt(template_obj)


def _build_cluster_genotype(
    cluster: int,
    *,
    system_cfg: dict,
    traits: dict[int, dict[str, str]],
    facets: dict[int, dict[str, str]],
    trait_targets: dict[int, dict[str, float]],
    facet_targets: dict[int, dict[str, float]],
) -> dict:
    trait_formulations = traits[cluster]
    facet_formulations = facets[cluster]
    cluster_trait_targets = trait_targets.get(cluster, {})
    cluster_facet_targets = facet_targets.get(cluster, {})
    return {
        "role_definition": system_cfg["role"],
        "trait_formulations": trait_formulations,
        "facet_formulations": facet_formulations,
        "intensity_modifiers": system_cfg["intensity_modifiers"],
        "critic_formulations": system_cfg["critic_internal"],
        "trait_targets": {
            k: cluster_trait_targets[k] for k in trait_formulations if k in cluster_trait_targets
        },
        "facet_targets": {
            k: cluster_facet_targets[k] for k in facet_formulations if k in cluster_facet_targets
        },
    }


def _build_traits_facets_text(genotype: dict, participant: pd.Series) -> tuple[str, str]:
    trait_targets = genotype.get("trait_targets") or {}
    facet_targets = genotype.get("facet_targets") or {}
    modifiers_cfg = genotype["intensity_modifiers"]

    traits_text = []
    for trait, description in genotype["trait_formulations"].items():
        value = participant.get(trait, participant.get(str(trait).lower()))
        if value is None:
            continue
        target = trait_targets.get(trait, trait_targets.get(str(trait).lower()))
        modifier = (
            get_modifier_by_match(value, target, modifiers_cfg)
            if target is not None
            else get_modifier_bisect(value, modifiers_cfg)
        )
        traits_text.append(f"- This trait ({trait}) describes you {modifier}: {description}")

    facets_text = []
    for facet, description in genotype["facet_formulations"].items():
        value = participant.get(facet, participant.get(str(facet).lower()))
        if value is None:
            continue
        target = facet_targets.get(facet, facet_targets.get(str(facet).lower()))
        modifier = (
            get_modifier_by_match(value, target, modifiers_cfg)
            if target is not None
            else get_modifier_bisect(value, modifiers_cfg)
        )
        facets_text.append(f"- This facet ({facet}) describes you {modifier}: {description}")

    return "\n".join(traits_text), "\n".join(facets_text)


def _render_prompt_for_participant(
    prompt_str: str,
    participant: pd.Series,
    task: dict,
    genotype_by_cluster: dict[int, dict],
) -> dict[str, str]:
    parsed = parse_prompt(prompt_str)
    cluster = int(participant["clusters"])
    genotype = genotype_by_cluster[cluster]
    traits_text, facets_text = _build_traits_facets_text(genotype, participant)
    questions_text = "\n".join(f"{q['id']}. {q['text']}" for q in task["ipip_neo"])

    system_text = parsed["system"]
    system_text = system_text.replace(PLACEHOLDERS["role"], genotype["role_definition"])
    system_text = system_text.replace(PLACEHOLDERS["traits"], traits_text)
    system_text = system_text.replace(PLACEHOLDERS["facets"], facets_text)
    system_text = system_text.replace(PLACEHOLDERS["critic"], genotype["critic_formulations"])

    instruction_text = parsed["instruction"].replace(PLACEHOLDERS["questions"], questions_text)
    human_text = f"{instruction_text}\n\n{parsed['output_format']}"

    return {"system": system_text, "human": human_text}


def _build_compact_prompt_export(genotype: dict | None) -> dict | None:
    if genotype is None:
        return None
    return {
        "role": genotype.get("role_definition"),
        "traits": deepcopy(genotype.get("trait_formulations") or {}),
        "facets": deepcopy(genotype.get("facet_formulations") or {}),
        "internal": genotype.get("critic_formulations"),
    }


def _score_single_participant(
    participant: pd.Series,
    prompt_str: str,
    task: dict,
    model,
    genotype_by_cluster: dict[int, dict],
) -> dict:
    ChatPromptTemplate = _get_chat_prompt_template_cls()
    helpers = _get_personality_helpers()
    OCEAN_AND_FACET_ORDER = helpers["OCEAN_AND_FACET_ORDER"]
    compute_five_factor_metrics = helpers["compute_five_factor_metrics"]
    five_factor = helpers["five_factor"]

    prompt = _render_prompt_for_participant(prompt_str, participant, task, genotype_by_cluster)
    # Prefer direct message invoke to avoid PromptTemplate f-string parsing of `{...}`.
    if hasattr(model, "llm") and model.llm is not None:
        SystemMessage, HumanMessage = _get_langchain_message_classes()
        response = model.llm.invoke(
            [
                SystemMessage(content=prompt["system"]),
                HumanMessage(content=prompt["human"]),
            ]
        )
    else:
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompt["system"]), ("human", prompt["human"])]
        )
        response = model.generate(prompt_template)
    model_answers = parse_response(response.content)
    if not model_answers:
        return {
            "similarity": None,
            "avg_diff": None,
            "pearson_corr": None,
            "mae_35": None,
            "mae_per_dim": None,
            "similarity_35": None,
            "pearson_35": None,
            "kappa_35": None,
            "mean_similarity_facets": None,
            "mean_similarity_traits": None,
            "parsed_ok": False,
            "model_answers": None,
            "simulated_ocean": None,
        }

    similarities = []
    abs_diffs = []
    lsit_model_ans = []
    lsit_human_ans = []
    for q_id, model_ans in model_answers.items():
        human_ans = participant.get(f"i{q_id}")
        if human_ans is None or (isinstance(human_ans, float) and np.isnan(human_ans)):
            continue
        lsit_model_ans.append(model_ans)
        lsit_human_ans.append(human_ans)
        similarities.append(1 - abs(model_ans - human_ans) / 4)
        abs_diffs.append(abs(model_ans - human_ans))

    similarity = float(np.mean(similarities)) if similarities else 0.0
    avg_diff = float(np.mean(abs_diffs)) if abs_diffs else 0.0
    if len(lsit_model_ans) >= 2 and len(lsit_human_ans) >= 2:
        try:
            p = np.corrcoef(lsit_model_ans, lsit_human_ans)[0, 1]
            pearson_corr = 0.0 if np.isnan(p) else float(p)
        except Exception:
            pearson_corr = 0.0
    else:
        pearson_corr = 0.0

    score = {
        "similarity": similarity,
        "avg_diff": avg_diff,
        "pearson_corr": pearson_corr,
        "mae_35": None,
        "mae_per_dim": None,
        "similarity_35": None,
        "pearson_35": None,
        "kappa_35": None,
        "mean_similarity_facets": None,
        "mean_similarity_traits": None,
        "parsed_ok": True,
        "model_answers": model_answers,
        "simulated_ocean": None,
    }

    simulated_ocean = five_factor.compute_ocean_facets(
        model_answers,
        participant.get("sex"),
        participant.get("age", 30),
        question=120,
    )
    score["simulated_ocean"] = simulated_ocean

    if simulated_ocean is not None and len(model_answers) >= 120:
        real_flat = {}
        for k in OCEAN_AND_FACET_ORDER:
            if k not in participant.index:
                continue
            v = participant[k]
            if v is None or (isinstance(v, float) and (np.isnan(v) or v != v)):
                continue
            try:
                real_flat[k] = float(v)
            except (TypeError, ValueError):
                pass
        common = [k for k in OCEAN_AND_FACET_ORDER if k in real_flat and k in simulated_ocean]
        if len(common) >= 30:
            m = compute_five_factor_metrics(real_flat, simulated_ocean, keys=common)
            score["mae_35"] = m["mae_35"]
            score["mae_per_dim"] = m["mae_per_dim"]
            score["similarity_35"] = m["similarity_35"]
            score["pearson_35"] = m["pearson_35"]
            score["kappa_35"] = m["kappa_35"]
            score["mean_similarity_facets"] = m["mean_similarity_facets"]
            score["mean_similarity_traits"] = m["mean_similarity_traits"]

    return score


def _evaluate_prompt(
    prompt_str: str,
    participants: pd.DataFrame,
    task: dict,
    model,
    genotype_by_cluster: dict[int, dict],
    batch_size: int = 1,
) -> dict[str, float | int]:
    helpers = _get_personality_helpers()
    aggregate_cluster_five_factor_metrics = helpers["aggregate_cluster_five_factor_metrics"]

    bs = int(batch_size or 0)
    if bs <= 1:
        scores = [
            _score_single_participant(p, prompt_str, task, model, genotype_by_cluster)
            for _, p in participants.iterrows()
        ]
    else:
        items = [(idx, p) for idx, p in participants.iterrows()]

        def _run_one(item):
            _idx, participant = item
            return _score_single_participant(participant, prompt_str, task, model, genotype_by_cluster)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=bs) as ex:
            scores = list(ex.map(_run_one, items))
        wall_s = time.perf_counter() - t0
        print(f"  [hype-batch] {len(items)} participants, max_concurrent={bs}, wall_time={wall_s:.1f}s")

    if not scores:
        return {
            "n": 0,
            "n_parsed": 0,
            "n_unparsed": 0,
            "n_answer_metrics": 0,
            "parse_success_rate": 0.0,
            "mean_similarity_ans": 0.0,
            "std_similarity": 0.0,
            "mean_avg_diff": 0.0,
            "mean_pearson_corr": 0.0,
            "mean_mae_35": 0.0,
            "mean_similarity_35": 0.0,
            "mean_similarity_facets": 0.0,
            "mean_similarity_traits": 0.0,
            "mean_pearson_35": 0.0,
            "mean_kappa_35": 0.0,
        }

    similarity_values = [float(s["similarity"]) for s in scores if s.get("similarity") is not None]
    avg_diff_values = [float(s["avg_diff"]) for s in scores if s.get("avg_diff") is not None]
    pearson_values = [float(s["pearson_corr"]) for s in scores if s.get("pearson_corr") is not None]
    n_total = len(scores)
    n_answer_metrics = len(similarity_values)
    n_unparsed = n_total - n_answer_metrics

    mean_similarity = float(np.mean(similarity_values)) if similarity_values else 0.0
    std_similarity = float(np.std(similarity_values)) if similarity_values else 0.0
    mean_avg_diff = float(np.mean(avg_diff_values)) if avg_diff_values else 0.0
    mean_pearson_corr = float(np.mean(pearson_values)) if pearson_values else 0.0
    agg_ff = aggregate_cluster_five_factor_metrics(scores)

    return {
        "n": n_total,
        "n_parsed": n_answer_metrics,
        "n_unparsed": n_unparsed,
        "n_answer_metrics": n_answer_metrics,
        "parse_success_rate": (float(n_answer_metrics) / float(n_total)) if n_total else 0.0,
        "mean_similarity_ans": mean_similarity,
        "std_similarity": std_similarity,
        "mean_avg_diff": mean_avg_diff,
        "mean_pearson_corr": mean_pearson_corr,
        "mean_mae_35": agg_ff.get("mean_mae_35", 0.0),
        "mean_similarity_35": agg_ff.get("mean_similarity_35", 0.0),
        "mean_similarity_facets": agg_ff.get("mean_similarity_facets", 0.0),
        "mean_similarity_traits": agg_ff.get("mean_similarity_traits", 0.0),
        "mean_pearson_35": agg_ff.get("mean_pearson_35", 0.0),
        "mean_kappa_35": agg_ff.get("mean_kappa_35", 0.0),
    }


def _build_persona_optimizer_input(genotype: dict) -> dict[str, object]:
    return {
        "role": genotype["role_definition"],
        "traits": deepcopy(genotype["trait_formulations"]),
        "facets": deepcopy(genotype["facet_formulations"]),
        "critic_internal": genotype["critic_formulations"],
    }


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_get_project_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def run_experiment(config: dict) -> dict:
    from src.models.registry import get_model

    experiment_cfg = config.get("experiment") or {}
    hype_cfg = config.get("hype") or {}

    seed = int(experiment_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    print("📦 Загрузка модели...")
    model = get_model(config["model"])
    print(f"✅ Модель загружена: {config['model'].get('model_name', 'unknown')}")

    if hasattr(model, "llm") and hype_cfg.get("temperature") is not None:
        model.llm.temperature = float(hype_cfg["temperature"])

    print("📂 Загрузка данных участников...")
    data_path = _resolve_path(config["data"]["file_path"])
    participants = pd.read_csv(data_path)

    clusters = config["data"].get("clusters") or sorted(participants["clusters"].dropna().unique().tolist())
    participants = participants.loc[participants["clusters"].isin(clusters)]

    n_subjects = int(
        experiment_cfg.get("n_subjects")
        or config["data"].get("num_participants")
        or len(participants)
    )
    n_subjects = min(n_subjects, len(participants))
    participants = participants.sample(n=n_subjects, random_state=seed)
    print(f"✅ Выбрано участников: {n_subjects} (clusters={clusters})")

    # В HYPE-режиме используем один и тот же набор участников.
    eval_df = participants
    participant_batch_size = int(
        hype_cfg.get("participant_batch_size")
        or (config.get("simulation") or {}).get("participant_batch_size")
        or 1
    )
    print(f"⚙️  participant_batch_size={participant_batch_size}")

    print("📋 Загрузка вопросов IPIP-NEO...")
    with open(_resolve_path("data/IPIP-NEO/120/questions.json"), "r", encoding="utf-8") as f:
        ipip_neo_questions = json.load(f).get("questions", [])
    print(f"✅ Вопросов загружено: {len(ipip_neo_questions)}")

    system_cfg = _load_system(config)
    traits = _load_traits(config)
    facets = _load_facets(config)
    trait_targets = _load_trait_target_values(config)
    facet_targets = _load_facet_target_values(config)

    task = {
        "task": system_cfg["task"],
        "ipip_neo": ipip_neo_questions,
        "response_format": system_cfg["response_format"],
    }

    used_clusters = sorted({int(c) for c in participants["clusters"].dropna().unique().tolist()})
    genotype_by_cluster = {
        c: _build_cluster_genotype(
            c,
            system_cfg=system_cfg,
            traits=traits,
            facets=facets,
            trait_targets=trait_targets,
            facet_targets=facet_targets,
        )
        for c in used_clusters
        if c in traits and c in facets
    }
    if not genotype_by_cluster:
        raise ValueError("No valid clusters found for building prompt genotype")

    base_prompt = _build_prompt_template(system_cfg)
    required_ph = {
        PLACEHOLDERS["traits"],
        PLACEHOLDERS["facets"],
        PLACEHOLDERS["critic"],
        PLACEHOLDERS["questions"],
    }
    validate_prompt_format(base_prompt, required_placeholders=required_ph, preserve_sections=True)

    baseline_genotype_by_cluster = deepcopy(genotype_by_cluster)
    llm = getattr(model, "llm", None)
    problem_description = hype_cfg.get("problem_description")
    has_problem_description = bool(str(problem_description).strip()) if problem_description else False

    print("\n" + "=" * 70)
    print("🧪 ПОКЛАСТЕРНОЕ ТЕСТИРОВАНИЕ: BASELINE + HYPE OPTIMIZED")
    print("=" * 70)
    cluster_results: dict[str, dict] = {}
    successful_clusters: list[int] = []
    skipped_clusters: list[int] = []

    for cluster in sorted(baseline_genotype_by_cluster):
        cluster_df = eval_df.loc[eval_df["clusters"] == cluster]
        if cluster_df.empty:
            continue

        print(f"\n--- Cluster {cluster} ---")
        baseline_genotype = baseline_genotype_by_cluster[cluster]
        timing = {
            "baseline_eval_s": None,
            "optimization_s": None,
            "optimized_eval_s": None,
        }

        t0 = time.perf_counter()
        baseline_metrics = _evaluate_prompt(
            base_prompt,
            cluster_df,
            task,
            model,
            {cluster: baseline_genotype},
            batch_size=participant_batch_size,
        )
        timing["baseline_eval_s"] = round(time.perf_counter() - t0, 6)
        print(
            "baseline mean_similarity_ans="
            f"{baseline_metrics['mean_similarity_ans']:.4f}"
        )

        t1 = time.perf_counter()
        optimized_genotype = None
        if llm is None:
            cluster_hype_meta = {
                "status": "skipped_due_to_optimization_error",
                "error": "hype_cfg['llm'] is required",
                "optimized_applied": False,
                "evaluation_skipped": True,
            }
        else:
            try:
                kwargs = {
                    "base_prompt": _build_persona_optimizer_input(baseline_genotype),
                    "llm": llm,
                }
                if has_problem_description:
                    kwargs["problem_description"] = str(problem_description).strip()
                opt_result = optimize_persona_prompt_hype(**kwargs)
                opt_prompt = opt_result.optimized_prompt
                optimized_genotype = deepcopy(baseline_genotype)
                optimized_genotype["role_definition"] = opt_prompt["role"]
                optimized_genotype["trait_formulations"] = opt_prompt["traits"]
                optimized_genotype["facet_formulations"] = opt_prompt["facets"]
                optimized_genotype["critic_formulations"] = opt_prompt["critic_internal"]
                cluster_hype_meta = {
                    "status": "ok",
                    "error": None,
                    "optimized_applied": True,
                    "evaluation_skipped": False,
                }
            except Exception as e:
                cluster_hype_meta = {
                    "status": "skipped_due_to_optimization_error",
                    "error": str(e),
                    "optimized_applied": False,
                    "evaluation_skipped": True,
                }
        timing["optimization_s"] = round(time.perf_counter() - t1, 6)

        if optimized_genotype is not None:
            t2 = time.perf_counter()
            hype_metrics = _evaluate_prompt(
                base_prompt,
                cluster_df,
                task,
                model,
                {cluster: optimized_genotype},
                batch_size=participant_batch_size,
            )
            timing["optimized_eval_s"] = round(time.perf_counter() - t2, 6)
            successful_clusters.append(cluster)
            print(
                "optimized mean_similarity_ans="
                f"{hype_metrics['mean_similarity_ans']:.4f}"
            )
        else:
            hype_metrics = None
            skipped_clusters.append(cluster)
            print("optimized evaluation skipped due to optimization error")

        cluster_subject_ids = []
        for idx, row in cluster_df.iterrows():
            if "case" in cluster_df.columns:
                cluster_subject_ids.append(row.get("case"))
            else:
                cluster_subject_ids.append(idx)

        cluster_results[str(cluster)] = {
            "subject_ids": cluster_subject_ids,
            "prompts": {
                "baseline": _build_compact_prompt_export(baseline_genotype),
                "hype_optimized": _build_compact_prompt_export(optimized_genotype),
            },
            "metrics": {
                "baseline": baseline_metrics,
                "hype_optimized": hype_metrics,
            },
            "hype": cluster_hype_meta,
            "timing": timing,
        }

    timestamp = datetime.now(timezone.utc).isoformat()
    result_payload = {
        "run_info": {
            "timestamp": timestamp,
            "git_commit": _git_commit(),
            "seed": seed,
            "config_path": config.get("config_file", "unknown"),
        },
        "data": {
            "n_subjects": n_subjects,
            "split": "train_only",
            "n_clusters": len(cluster_results),
        },
        "sources": {
            "traits_path": (config.get("prompt") or {}).get("traits_path"),
            "facets_path": (config.get("prompt") or {}).get("facets_path"),
            "system_path": (config.get("prompt") or {}).get("system_path"),
            "used_clusters": sorted(list(baseline_genotype_by_cluster.keys())),
        },
        "clusters": cluster_results,
        "hype": {
            "status": (
                "ok"
                if successful_clusters and not skipped_clusters
                else ("partial" if successful_clusters else "error")
            ),
            "problem_description_from_config": has_problem_description,
            "successful_clusters": successful_clusters,
            "skipped_clusters": skipped_clusters,
            "clusters": {k: v["hype"] for k, v in cluster_results.items()},
            "optimized_eval_skipped": not bool(successful_clusters),
            "participant_batch_size": participant_batch_size,
        },
    }

    save_raw = bool(hype_cfg.get("save_raw", False))
    raw_payload = None
    if save_raw:
        raw_payload = {
            "cluster_ids": sorted([int(k) for k in cluster_results.keys()]),
            "hype_meta": {k: v["hype"] for k, v in cluster_results.items()},
        }

    results_dir = Path(config["results_dir"])
    exp_name = config.get("name", "hype_experiment")
    output_path = save_compact_result(
        result_payload=result_payload,
        results_dir=results_dir,
        exp_name=exp_name,
        save_raw=save_raw,
        raw_payload=raw_payload,
    )

    print("\n" + "=" * 70)
    print("📈 ИТОГОВЫЕ УСРЕДНЁННЫЕ МЕТРИКИ")
    print("=" * 70)
    baseline_values = [
        v["metrics"]["baseline"]["mean_similarity_ans"]
        for v in cluster_results.values()
        if v.get("metrics", {}).get("baseline")
    ]
    hype_values = [
        v["metrics"]["hype_optimized"]["mean_similarity_ans"]
        for v in cluster_results.values()
        if v.get("metrics", {}).get("hype_optimized")
    ]
    baseline_mean = float(np.mean(baseline_values)) if baseline_values else 0.0
    hype_mean = float(np.mean(hype_values)) if hype_values else None
    print(f"Baseline: mean_similarity_ans={baseline_mean:.4f}")
    if hype_mean is not None:
        print(f"HypeOpt : mean_similarity_ans={hype_mean:.4f}")
    else:
        print("HypeOpt : skipped (no clusters passed HyPE optimization)")
    print("=" * 70)
    print(f"Saved hype result: {output_path}")
    return result_payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run hype-based prompt optimization experiment")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)

    project_root = _get_project_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.setdefault("experiment_id", f"hype_{timestamp}")
    cfg.setdefault("results_dir", str(project_root / "results_experiments" / cfg["experiment_id"]))
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    run_experiment(cfg)
