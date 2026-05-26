from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from agent_core import WeiboBotDetectorAgent
from config import AppConfig
from GeminiAPI import AsyncGeminiMultimodalClient
from multimodal_checker import MultimodalConsistencyChecker
from weibo_scraper import WeiboLoginRequiredError, WeiboScraper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微博机器人账号检测 LLM-agent")
    parser.add_argument(
        "account",
        nargs="?",
        default=None,
        help="微博 uid、完整主页 URL，或包含 uid 字典的 JSON 文件路径（--evaluate 模式下可省略）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认读取 OUTPUT_DIR 或 weibo-accounts-outputs",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="评测模式：依次检测测试集和 Agent 账号，并报告 acc/f1/precision/recall",
    )
    parser.add_argument(
        "--reference-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="数据截止日期，超过该日期的博文将被过滤（默认不过滤）",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        metavar="N",
        help="评测模式下重复实验次数（默认 3）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("正在加载配置...", file=sys.stderr, flush=True)
    config = AppConfig.from_env()
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        # 默认输出目录改为 weibo-accounts-outputs
        config.output_dir = "weibo-accounts-outputs"
    if args.reference_date:
        config.reference_date = args.reference_date
    try:
        config.validate()
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    if args.evaluate:
        return run_evaluation(config, args.num_runs)

    if args.account is None:
        print("错误: 请提供账号参数，或使用 --evaluate 进入评测模式", file=sys.stderr)
        return 2

    # 解析输入：判断是 JSON 文件还是单个账号
    uid_dict = load_uid_dict(args.account)
    
    if uid_dict is not None:
        # 批量检测模式
        return run_batch_detection(uid_dict, config)
    else:
        # 单个账号检测模式
        return run_single_detection(args.account, config)

def load_uid_dict(account_input: str) -> dict[str, int] | None:
    """尝试将输入解析为 UID 字典（JSON 文件）。
    
    如果输入是 JSON 文件且内容为字典，返回该字典；否则返回 None。
    """
    path = Path(account_input)
    if not path.is_file() or path.suffix.lower() != ".json":
        return None
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 验证是字典且 key 都是字符串
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        pass
    
    return None


def run_single_detection(account: str, config: AppConfig) -> int:
    """单个账号检测模式。"""
    # Gemini 异步客户端用于图文分析
    gemini_client = AsyncGeminiMultimodalClient(
        api_key=config.gemini_api_key,
        model=config.gemini_model,
        timeout_seconds=config.api_timeout_seconds,
    )
    
    print("正在初始化 Agent...", file=sys.stderr, flush=True)
    agent = WeiboBotDetectorAgent(
        config=config,
        scraper=WeiboScraper(config),
        multimodal_checker=MultimodalConsistencyChecker(config, gemini_client),
    )

    agent_timeout = max(600.0, config.api_timeout_seconds * 4)
    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        timed_out = False
        try:
            future = executor.submit(agent.run, account)
            try:
                result = future.result(timeout=agent_timeout)
            except concurrent.futures.TimeoutError:
                timed_out = True
                raise TimeoutError(f"检测超时（>{agent_timeout:.0f}秒），进程可能被 Gemini API 卡住")
        finally:
            # 正常完成时等待线程结束以释放资源；超时时不等待以避免挂起
            executor.shutdown(wait=not timed_out)
    except WeiboLoginRequiredError as exc:
        print(f"微博登录态缺失或已失效: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"执行失败: {exc!r}", file=sys.stderr)
        return 1

    output_paths = write_outputs(result.to_dict(), config.output_path, result.profile.uid, use_timestamp=False)
    result.raw_data_paths = output_paths
    Path(output_paths["result_json"]).write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"账号: {result.profile.screen_name or result.profile.uid}")
    print(f"结论: {result.verdict}")
    print(f"置信度: {result.confidence:.2f}")
    print(f"摘要: {result.summary}")
    print(f"输出文件: {output_paths['result_json']}")
    return 0


def run_batch_detection(
    uid_dict: dict[str, int],
    config: AppConfig,
    output_path: Path | None = None,
) -> int:
    """批量检测模式，支持断点续传和自动重试。

    output_path: 覆盖 config.output_path，用于评测模式下的子目录隔离（如 run1/test/）。
    """
    effective_output = output_path if output_path is not None else config.output_path
    # Gemini 异步客户端用于图文分析
    gemini_client = AsyncGeminiMultimodalClient(
        api_key=config.gemini_api_key,
        model=config.gemini_model,
        timeout_seconds=config.api_timeout_seconds,
    )
    
    print("正在初始化 Agent...", file=sys.stderr, flush=True)
    agent = WeiboBotDetectorAgent(
        config=config,
        scraper=WeiboScraper(config),
        multimodal_checker=MultimodalConsistencyChecker(config, gemini_client),
    )

    total = len(uid_dict)
    success_count = 0
    failed_count = 0
    skipped_count = 0
    gc_freq = max(1, min(5, total // 20))  # 每 N 个账号触发一次 GC，最少 1 最多 5
    
    print(f"\n开始批量检测，共 {total} 个账号", file=sys.stderr, flush=True)
    print(f"输出目录: {effective_output}", file=sys.stderr, flush=True)
    print(f"API 超时设置: {config.api_timeout_seconds} 秒", file=sys.stderr, flush=True)
    print("=" * 60, file=sys.stderr, flush=True)
    
    for idx, (uid, _) in enumerate(uid_dict.items(), 1):
        # 断点续传：当且仅当 result.json 文件存在才视为已完成
        result_path = effective_output / uid / "result.json"
        if result_path.is_file():
            print(f"[{idx}/{total}] 跳过 {uid} (已有检测结果)", flush=True)
            skipped_count += 1
            continue
        
        print(f"\n[{idx}/{total}] 正在检测 {uid}...", flush=True)
        
        # 重试机制：最多重试 2 次
        max_retries = 2
        retry_delay = 5  # 秒
        detected = False
        
        agent_timeout = max(600.0, config.api_timeout_seconds * 4)
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    print(f"  第 {attempt} 次重试...", flush=True)
                    import time
                    time.sleep(retry_delay)
                
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                timed_out = False
                try:
                    future = executor.submit(agent.run, uid, str(effective_output))
                    try:
                        result = future.result(timeout=agent_timeout)
                    except concurrent.futures.TimeoutError:
                        timed_out = True
                        raise TimeoutError(f"检测超时（>{agent_timeout:.0f}秒），Gemini API 无响应")
                finally:
                    # 正常完成时等待线程结束以释放资源；超时时不等待以避免挂起
                    executor.shutdown(wait=not timed_out)
                
                # 写入结果（批量模式不使用时间戳目录）
                output_paths = write_outputs(result.to_dict(), effective_output, result.profile.uid, use_timestamp=False)
                result.raw_data_paths = output_paths
                Path(output_paths["result_json"]).write_text(
                    json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                
                print(f"  账号: {result.profile.screen_name or result.profile.uid}")
                print(f"  结论: {result.verdict}")
                print(f"  置信度: {result.confidence:.2f}")
                print(f"  摘要: {result.summary}")
                print(f"  结果已保存: {output_paths['result_json']}")
                success_count += 1
                detected = True
                # 释放本次检测产生的大量对象
                del result
                break  # 成功，跳出重试循环
                
            except WeiboLoginRequiredError as exc:
                print(f"  微博登录态缺失或已失效: {exc}", file=sys.stderr)
                failed_count += 1
                # 登录失败则终止批量检测
                print("\n❌ 登录态失效，终止批量检测", file=sys.stderr)
                print("=" * 60, file=sys.stderr, flush=True)
                print(f"总计: {total} | 成功: {success_count} | 失败: {failed_count} | 跳过: {skipped_count}", file=sys.stderr, flush=True)
                return 3
                
            except Exception as exc:  # noqa: BLE001
                error_msg = repr(exc)
                if attempt < max_retries:
                    print(f"  检测失败 (第 {attempt}/{max_retries} 次): {error_msg}", file=sys.stderr)
                    print(f"  等待 {retry_delay} 秒后重试...", file=sys.stderr)
                else:
                    print(f"  ❌ 检测失败 (已重试 {max_retries} 次): {error_msg}", file=sys.stderr)
                    failed_count += 1
                    # 单个失败不影响其他账号，继续检测
                    continue
        
        if detected:
            # 定期触发垃圾回收，释放累积的临时对象（LangChain agent 图、图片数据等）
            if success_count % gc_freq == 0:
                gc.collect()
            continue
    
    # 输出统计信息
    print("\n" + "=" * 60, file=sys.stderr, flush=True)
    print(f"✅ 批量检测完成！", file=sys.stderr, flush=True)
    print(f"总计: {total} | 成功: {success_count} | 失败: {failed_count} | 跳过: {skipped_count}", file=sys.stderr, flush=True)
    print(f"结果目录: {effective_output}", file=sys.stderr, flush=True)
    
    return 0 if failed_count == 0 else 1


_SPLIT_FILE = Path("account_to_detect/split.json")
_LABELS_FILE = Path("account_to_detect/Weibo_Labels.json")


def load_test_accounts() -> dict[str, int]:
    """从 account_to_detect/split.json 和 Weibo_Labels.json 中加载测试集账号及其真实标签。

    跳过 label=1（疑似）的账号。返回 {uid: label} 字典，label 为 0（人类）或 2（机器人）。
    """
    with open(_SPLIT_FILE, encoding="utf-8") as f:
        split_data = json.load(f)
    with open(_LABELS_FILE, encoding="utf-8") as f:
        labels_data = json.load(f)

    test_accounts: dict[str, int] = {}
    for uid, split in split_data.items():
        if split == "test":
            label = labels_data.get(str(uid))
            if label is not None and int(label) != 1:  # 跳过疑似账号
                test_accounts[str(uid)] = int(label)
    return test_accounts


def load_agent_accounts() -> dict[str, int]:
    """从 account_to_detect/Weibo_Labels.json 中加载 Agent 账号（label=-1）。"""
    with open(_LABELS_FILE, encoding="utf-8") as f:
        labels_data = json.load(f)
    return {str(uid): int(label) for uid, label in labels_data.items() if int(label) == -1}


def compute_metrics_from_results(
    uid_label_dict: dict[str, int],
    output_path: Path,
    positive_labels: set[int],
) -> dict:
    """从已有 result.json 文件中计算 acc 及 macro precision/recall/F1（sklearn 实现）。

    positive_labels: 被视为正类（bot）的真实标签集合，映射为 y=1；其余为 y=0。
    无法读取 result.json 的账号计入 missing，不参与指标计算。
    """
    y_true: list[int] = []
    y_pred: list[int] = []
    missing = 0

    for uid, true_label in uid_label_dict.items():
        result_path = output_path / str(uid) / "result.json"
        if not result_path.is_file():
            missing += 1
            continue

        with open(result_path, encoding="utf-8") as f:
            result_data = json.load(f)

        y_true.append(1 if true_label in positive_labels else 0)
        y_pred.append(1 if result_data.get("verdict", "") == "likely_bot" else 0)

    if not y_true:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0, "missing": missing}

    acc       = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall    = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1        = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # labels=[0,1] 保证即使某一类无样本也能得到 2×2 矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    return {
        "acc": acc, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": len(y_true), "missing": missing,
    }


def print_metrics(metrics: dict, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    total_with_missing = metrics["total"] + metrics["missing"]
    print(f"账号总数: {total_with_missing} | 有结果: {metrics['total']} | 无结果(跳过): {metrics['missing']}")
    print(f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
    print(f"Accuracy:        {metrics['acc']:.4f}  ({metrics['acc'] * 100:.2f}%)")
    print(f"Macro-Precision: {metrics['precision']:.4f}  ({metrics['precision'] * 100:.2f}%)")
    print(f"Macro-Recall:    {metrics['recall']:.4f}  ({metrics['recall'] * 100:.2f}%)")
    print(f"Macro-F1:        {metrics['f1']:.4f}  ({metrics['f1'] * 100:.2f}%)")
    print(f"{'=' * 60}")


def average_metrics(metrics_list: list[dict]) -> dict:
    """对多次 run 的 macro 指标取算术均值。"""
    keys = ("acc", "precision", "recall", "f1")
    avg = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}
    return avg


def print_average_metrics(avg: dict, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"Accuracy:        {avg['acc']:.4f}  ({avg['acc'] * 100:.2f}%)")
    print(f"Macro-Precision: {avg['precision']:.4f}  ({avg['precision'] * 100:.2f}%)")
    print(f"Macro-Recall:    {avg['recall']:.4f}  ({avg['recall'] * 100:.2f}%)")
    print(f"Macro-F1:        {avg['f1']:.4f}  ({avg['f1'] * 100:.2f}%)")
    print(f"{'=' * 60}")


def run_evaluation(config: AppConfig, num_runs: int = 3) -> int:
    """评测模式：对测试集和 OOD Agent 账号各进行 N 次重复实验，报告每次及均值指标。

    输出目录结构：
        <output_dir>/run{1..N}/test/{uid}/  ← 测试集账号
        <output_dir>/run{1..N}/ood/{uid}/   ← Agent 账号
    """
    # ── 加载账号列表（只读一次） ───────────────────────────────
    test_accounts = load_test_accounts()
    agent_accounts = load_agent_accounts()
    # Agent label -1 → 2，与测试集正类（label=2，机器人）统一，方便 macro 计算
    agent_accounts_bot = {uid: 2 for uid in agent_accounts}

    bot_count   = sum(1 for v in test_accounts.values() if v == 2)
    human_count = sum(1 for v in test_accounts.values() if v == 0)
    print(f"\n测试集账号数: {len(test_accounts)}  (机器人: {bot_count}, 人类: {human_count})", flush=True)
    print(f"OOD Agent 账号数: {len(agent_accounts)}", flush=True)

    NUM_RUNS = num_runs
    test_metrics_all: list[dict] = []
    ood_metrics_all:  list[dict] = []
    overall_ret = 0

    for run_num in range(1, NUM_RUNS + 1):
        print("\n" + "#" * 60, flush=True)
        print(f"  实验 Run {run_num} / {NUM_RUNS}", flush=True)
        print("#" * 60, flush=True)

        test_output = config.output_path / f"run{run_num}" / "test"
        ood_output  = config.output_path / f"run{run_num}" / "ood"

        # ── 测试集检测 ────────────────────────────────────────
        print(f"\n[Run {run_num}] 阶段 1/2：测试集账号检测", flush=True)
        ret_test = run_batch_detection(test_accounts, config, output_path=test_output)
        if ret_test == 3:
            return 3

        m_test = compute_metrics_from_results(test_accounts, test_output, positive_labels={2})
        print_metrics(m_test, f"Run {run_num} · 测试集  (Macro, 正类 = 机器人 label=2)")
        test_metrics_all.append(m_test)

        # ── OOD Agent 检测 ────────────────────────────────────
        print(f"\n[Run {run_num}] 阶段 2/2：OOD Agent 账号检测", flush=True)
        ret_ood = run_batch_detection(agent_accounts, config, output_path=ood_output)
        if ret_ood == 3:
            return 3

        m_ood = compute_metrics_from_results(agent_accounts_bot, ood_output, positive_labels={2})
        print_metrics(m_ood, f"Run {run_num} · OOD Agent  (Macro, 正类 = Bot/Agent)")
        ood_metrics_all.append(m_ood)

        if ret_test != 0 or ret_ood != 0:
            overall_ret = 1

    # ── 3 次均值汇总 ──────────────────────────────────────────
    print_average_metrics(average_metrics(test_metrics_all), f"{NUM_RUNS} 次平均 · 测试集  (Macro)")
    print_average_metrics(average_metrics(ood_metrics_all),  f"{NUM_RUNS} 次平均 · OOD Agent  (Macro)")

    return overall_ret


def write_outputs(result_dict: dict, output_root: Path, user_id: str, use_timestamp: bool = True) -> dict[str, str]:
    if use_timestamp:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / f"{user_id}_{timestamp}"
    else:
        # 批量检测模式：直接使用 uid 作为目录名，便于断点续传
        run_dir = output_root / user_id
    run_dir.mkdir(parents=True, exist_ok=True)

    profile_path = run_dir / "profile.json"
    posts_path = run_dir / "posts.json"
    image_analyses_path = run_dir / "image_analyses.json"
    result_path = run_dir / "result.json"

    profile_path.write_text(
        json.dumps(result_dict["profile"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    posts_path.write_text(
        json.dumps(result_dict["posts"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    image_analyses_path.write_text(
        json.dumps(result_dict["image_analyses"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "run_dir": str(run_dir),
        "profile_json": str(profile_path),
        "posts_json": str(posts_path),
        "image_analyses_json": str(image_analyses_path),
        "result_json": str(result_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
