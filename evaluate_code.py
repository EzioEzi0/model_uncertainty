import argparse
import pickle
import json
import re
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import TimeoutError as FutureTimeoutError
from pebble import ProcessPool
from pebble.common import ProcessExpired
import io
import sys
import builtins
import contextlib
from io import StringIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


@contextlib.contextmanager
def stdinIO(input_str):
    buf = io.StringIO(input_str)

    def mock_input(prompt=None):
        try:
            return buf.readline().rstrip("\n")
        except Exception as e:
            pass

    original_input = builtins.input
    builtins.input = mock_input
    yield
    builtins.input = original_input


def preprocess_input(input_data):
    if isinstance(input_data, list):
        return "\n".join(input_data)
    else:
        return input_data


def worker(code, input_data=""):
    """Worker for executing python code"""
    input_data = preprocess_input(input_data)

    with stdinIO(input_data), stdoutIO() as s:
        try:
            exec(code, {"__name__": "__main__"})
            res = s.getvalue().strip()
        except SystemExit as e:
            res = s.getvalue().strip()
        except Exception as e:
            res = f"ERROR: {e}"

    return res


def extract_code_from_response(response: str) -> str:
    """Extract Python code from response"""
    # Try to find code between ```python and ```
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try to find code between ``` and ```
    pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return ""


def check_output(actual, expected):
    """Check if actual output matches expected"""
    try:
        actual = actual.strip()
        expected = expected.strip()
    except:
        return False

    if actual == expected:
        return True

    # Try float comparison
    try:
        actual_f = float(actual)
        expected_f = float(expected)
        if abs(actual_f - expected_f) < 1e-4:
            return True
        if abs(actual_f - expected_f) < max(abs(expected_f), abs(actual_f)) * 1e-4:
            return True
    except:
        pass

    # Try multi-line comparison
    try:
        if "\n" in actual and "\n" in expected:
            actual_lines = [i.strip() for i in actual.split("\n")]
            expected_lines = [i.strip() for i in expected.split("\n")]
            if len(actual_lines) == len(expected_lines):
                all_match = True
                for a, e in zip(actual_lines, expected_lines):
                    if a != e:
                        try:
                            if abs(float(a) - float(e)) >= 1e-4:
                                all_match = False
                                break
                        except:
                            all_match = False
                            break
                if all_match:
                    return True
    except:
        pass

    return False


def test_code(pool, code, test_cases):
    """Test a single code against all test cases. Returns (all_passed, first_failed_idx).

    Returns:
        all_passed: True if all test cases pass
        first_failed_idx: Index of first failed test case, or -1 if all passed
    """
    if not code:
        return False, 0  # No code means fail at first test

    for i, tc in enumerate(test_cases):
        input_data = tc['input']
        expected = str(tc['output']).strip()

        try:
            future = pool.schedule(worker, (code, input_data), timeout=2)
            actual = future.result(timeout=3)
        except (FutureTimeoutError, ProcessExpired, Exception) as e:
            return False, i  # Failed at test i

        if not check_output(actual, expected):
            return False, i  # Failed at test i

    return True, -1  # All passed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate code responses for APPS dataset")
    parser.add_argument("--data_type", type=str, default="test", choices=["test", "train"],
                        help="Data type: test or train")
    parser.add_argument("--run_id", type=int, default=1,
                        help="Run ID for multiple train runs (1, 2, 3, ...)")
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    save_dir = script_dir / "output"
    save_dir.mkdir(exist_ok=True)
    data_dir = script_dir / "data"

    # Select data file based on data_type
    if args.data_type == "test":
        data_path = data_dir / "apps_test_1.pkl"  # 测试用，改回 apps_test_split.pkl 跑全量
        output_suffix = ""
    else:  # train
        data_path = data_dir / "apps_train_1.pkl"  # 测试用，改回 apps_train_2670.pkl 跑全量
        output_suffix = f"_run{args.run_id}"

    gen_results_path = save_dir / f"apps_generation_results_{args.data_type}{output_suffix}.pkl"

    # Load generation results
    logger.info(f"Loading generation results from {gen_results_path}...")
    with open(gen_results_path, 'rb') as f:
        gen_results = pickle.load(f)
    logger.info(f"Loaded {len(gen_results)} generation results")

    # Load original data with test cases
    logger.info(f"Loading original data with test cases from {data_path}...")
    with open(data_path, 'rb') as f:
        original_data = pickle.load(f)
    logger.info(f"Loaded {len(original_data)} original questions")

    # Create id -> test_cases mapping
    id_to_tests = {}
    for item in original_data:
        qid = item['id']
        if 'input_output' in item:
            io_data = item['input_output']
            if isinstance(io_data, str):
                io_data = json.loads(io_data)
            inputs = io_data.get('inputs', [])
            outputs = io_data.get('outputs', [])
            test_cases = [{'input': inp, 'output': out} for inp, out in zip(inputs, outputs)]
            id_to_tests[qid] = test_cases

    logger.info(f"Created test case mapping for {len(id_to_tests)} questions")

    # Prepare evaluation tasks
    eval_tasks = []
    skipped_no_code = 0

    for result in gen_results:
        qid = result['question_id']
        greedy_response = result.get('greedy_response', '')
        sampled_responses = result.get('sampled_responses', [])

        test_cases = id_to_tests.get(qid, [])
        if not test_cases:
            continue

        # Extract all codes (1 greedy + 10 sampled)
        all_codes = []

        # Greedy code
        greedy_code = extract_code_from_response(greedy_response)
        all_codes.append(('greedy', greedy_code))

        # Sampled codes
        for i, sampled in enumerate(sampled_responses):
            sampled_response = sampled.get('response', '')
            sampled_code = extract_code_from_response(sampled_response)
            all_codes.append((f'sampled_{i}', sampled_code))

        # Check if at least one has code (from any of the 11 responses)
        has_any_code = any(code for _, code in all_codes)
        if not has_any_code:
            skipped_no_code += 1
            continue

        eval_tasks.append({
            'question_id': qid,
            'level': result.get('level', 'unknown'),
            'all_codes': all_codes,
            'test_cases': test_cases
        })

    logger.info(f"Prepared {len(eval_tasks)} valid evaluation tasks")
    logger.info(f"Skipped (no code in any of 11 responses): {skipped_no_code}")

    # Run evaluations
    logger.info("Running code evaluations (testing all 11 responses per question)...")

    results = []
    correct_count = 0
    greedy_correct_count = 0
    level_stats = {'competition': {'correct': 0, 'total': 0, 'greedy_correct': 0},
                   'interview': {'correct': 0, 'total': 0, 'greedy_correct': 0},
                   'introductory': {'correct': 0, 'total': 0, 'greedy_correct': 0}}

    with ProcessPool(max_workers=16) as pool:
        for task in tqdm(eval_tasks, desc="Evaluating"):
            qid = task['question_id']
            level = task['level']
            all_codes = task['all_codes']
            test_cases = task['test_cases']

            level_stats[level]['total'] += 1

            # Test ALL codes and record each one's correctness
            any_passed = False
            greedy_passed = False
            passed_response = None
            individual_results = []  # 记录每个 response 的正确性

            for resp_type, code in all_codes:
                is_correct, first_failed_idx = test_code(pool, code, test_cases)
                individual_results.append({
                    'type': resp_type,
                    'correct': is_correct,
                    'has_code': bool(code),
                    'first_failed_idx': first_failed_idx  # -1 if all passed, otherwise the index of first failed test
                })

                if is_correct:
                    if not any_passed:  # 记录第一个通过的
                        passed_response = resp_type
                    any_passed = True
                    if resp_type == 'greedy':
                        greedy_passed = True

            if any_passed:
                correct_count += 1
                level_stats[level]['correct'] += 1

            if greedy_passed:
                greedy_correct_count += 1
                level_stats[level]['greedy_correct'] += 1

            # 统计 sampled 中通过的数量
            sampled_correct_count = sum(1 for r in individual_results if r['type'].startswith('sampled') and r['correct'])

            results.append({
                'question_id': qid,
                'level': level,
                'correct': any_passed,
                'greedy_correct': greedy_passed,
                'passed_response': passed_response,
                'individual_results': individual_results,
                'sampled_correct_count': sampled_correct_count
            })

    # Calculate accuracy
    total = len(eval_tasks)
    accuracy = correct_count / total if total > 0 else 0
    greedy_accuracy = greedy_correct_count / total if total > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS (Any of 11 responses correct)")
    logger.info("=" * 60)
    logger.info(f"Original questions: {len(gen_results)}")
    logger.info(f"Skipped (no code in any response): {skipped_no_code}")
    logger.info(f"Valid questions evaluated: {total}")
    logger.info("")
    logger.info(f"Greedy only correct: {greedy_correct_count}")
    logger.info(f"Greedy Accuracy: {greedy_accuracy:.2%}")
    logger.info("")
    logger.info(f"Any correct (pass@11): {correct_count}")
    logger.info(f"Pass@11 Accuracy: {accuracy:.2%}")
    logger.info("")
    logger.info("Accuracy by level:")
    for level, stats in level_stats.items():
        if stats['total'] > 0:
            level_acc = stats['correct'] / stats['total']
            greedy_acc = stats['greedy_correct'] / stats['total']
            logger.info(f"  {level}:")
            logger.info(f"    Greedy: {stats['greedy_correct']}/{stats['total']} = {greedy_acc:.2%}")
            logger.info(f"    Pass@11: {stats['correct']}/{stats['total']} = {level_acc:.2%}")

    # Save results
    output_path = save_dir / f"apps_eval_results_{args.data_type}{output_suffix}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'accuracy': accuracy,
            'greedy_accuracy': greedy_accuracy,
            'correct_count': correct_count,
            'greedy_correct_count': greedy_correct_count,
            'total': total,
            'level_stats': level_stats,
            'skipped_no_code': skipped_no_code
        }, f)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
