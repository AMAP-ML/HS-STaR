from simplerl_math_eval.grader import math_equal
from simplerl_math_eval.parser import strip_string, extract_answer

from .data import RespSampleBase
from pebble import ProcessPool  # 新增
from tqdm import tqdm  # 新增




def run_extract_answer_numina_math(result):
    prediction = extract_answer(result, data_name="numina")
    prediction = strip_string(prediction, skip_unit=False)
    return prediction

def run_eval_numina_math(sample) -> bool:
    """Evaluate a sample based on comprehensive information."""
    ans = sample.ans if sample.ans is not None else run_extract_answer_numina_math(sample.resp)
    correct = math_equal(ans, sample.ref_ans, timeout=True)
    return correct


class EvaluatorMathBatchNuminaMath():
    def __init__(
        self,       
    ):
        super(EvaluatorMathBatchNuminaMath, self).__init__()
        print("#"*50 + "EvaluatorMathBatchNuminaMath" + "#"*50)
        self.timeout = 5
    def batch_eval(
        self, samples: list[RespSampleBase], n_procs: int = 2, use_tqdm: bool = True
    ) -> tuple[list[str], list[bool]]:
        """Evaluate a batch of `samples` based on comprehensive information in place."""

        n_samples = len(samples)
        with ProcessPool(max_workers=min(n_procs, n_samples), max_tasks=1024) as pool:
            # NOTE: multi-processing does not support modifications in place
            resps = [sample.resp for sample in samples]
            iterator = pool.map(run_extract_answer_numina_math, resps, timeout=self.timeout).result()
            answers = []
            pbar = tqdm(total=n_samples, desc="Extracting") if use_tqdm else None
            corrects = []
            while True:
                try:
                    result = next(iterator)
                    answers.append(result)
                except StopIteration:
                    break
                except Exception:
                    answers.append("")
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()

            for sample, ans in zip(samples, answers):
                sample.ans = str(ans)

            iterator = pool.map(run_eval_numina_math, samples, timeout=self.timeout).result()
            pbar = tqdm(total=n_samples, desc="Evaluating") if use_tqdm else None
            corrects = []
            while True:
                try:
                    result = next(iterator)
                    corrects.append(result)
                except StopIteration:
                    break
                except Exception:
                    corrects.append(False)
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()

            for sample, correct in zip(samples, corrects):
                sample.correct = bool(correct)

        return answers, corrects


    def extract_explicit_ans(self, resp: str) -> str:
        """Extract the answer from the response."""
        return run_extract_answer_numina_math(resp)

