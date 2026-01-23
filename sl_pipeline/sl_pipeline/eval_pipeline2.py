from logging import raiseExceptions
from typing import Literal
from .dataset_token_clf import TokenClassificationDataset
from transformers.trainer_utils import set_seed
import random
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import numpy as np
import evaluate
from transformers.data.data_collator import DataCollatorForTokenClassification

import torch
import math

Bias = Literal["none", "all", "lora_only"]

supported_models = [
    "bert",  # Bert for Baseline #1
    "distilbert",  # DistilBert for Baseline #2
    "roberta",
    "modern-bert",
    "qwen2",  # qwen2 just testing peft
    "qwen3",  # qwen3 custom Token CLF class test
    "qwen3-eager",  # qwen3 custom Token CLF class test
    "qwen3-normal",  # qwen3 Token CLF DEFAULT
    "qwen3-unmasked",  # unmaksed qwen3
    "qwen3-unmasked-eager",
    "qwen3-unmasked-middle",
    "qwen3-unmasked-SSPFB",
    "qwen3-alpha-mask",
    "qwen3-summed-attention",
    "qwen3-summed-headwise",
    "qwen3-SUM",
    "qwen3-repeat-k",
    "gemma",
    "gemma-unmasked",
    "gemma-unmasked-middle",
    "gemma-unmasked-SSPFB",
    "gemma-summed-headwise",
    "gemma-repeat-k",
    "gemma2",
    "gemma2-unmasked",
    "gemma2-unmasked-middle",
    "gemma2-unmasked-SSPFB",
    "gemma2-summed-headwise",
    "gemma2-repeat-k",
    "mistral",
    "mistral-unmasked",
    "mistral-unmasked-middle",
    "mistral-unmasked-SSPFB",
    "mistral-summed-headwise",
    "mistral-repeat-k",
]


supported_datasets = [
    "wnut2017",
    "conll2003",
    "aac",
    "nlupp",
    "ontonotes",
    "absa-restaurants",
    "ace",
]


def get_supported_models():
    return supported_models


class EvalModelDataset2:

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        hf_name: str,
        load_dir: str,
        split: str = "validation",
        k_repeat: int = -1,
    ):
        if model_name not in supported_models:
            raise Exception(
                "MODEL NOT SUPPORTED\nSUPPORTED MODELS ARE:\n"
                + " ".join(supported_models)
            )
        if dataset_name not in supported_datasets:
            raise Exception(
                "MODEL NOT SUPPORTED\nSUPPORTED MODELS ARE:\n"
                + " ".join(supported_datasets)
            )
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.hf_name = hf_name
        self.load_dir = load_dir
        self.k_repeat = k_repeat
        self.tokenizerInstance()
        self.datasetInstance(split=split)
        # self.modelInstance()

    def tokenizerInstance(self):

        from transformers import AutoTokenizer
        from transformers import RobertaTokenizerFast

        if self.model_name == "roberta":
            self.tokenizer = RobertaTokenizerFast.from_pretrained(
                self.hf_name, add_prefix_space=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def datasetInstance(self, split: str = "validation"):
        self.dataset = TokenClassificationDataset(
            dataset_name=self.dataset_name, tokenizer=self.tokenizer, split=split
        )

    def setPeftModel(self):
        from peft import prepare_model_for_kbit_training, get_peft_model

        self.org_model = prepare_model_for_kbit_training(self.org_model)
        if self.loraConfig is None:
            raise Exception("LORA CONFIG NOT SET!")
        self.model = get_peft_model(self.org_model, self.loraConfig)

    def modelInstance(self, seed: int = 42):
        from transformers import AutoConfig

        bla = "-" + str(self.k_repeat) if "repeat" in self.model_name else ""
        self.full_dir = (
            self.load_dir
            + self.dataset_name
            + "/"
            + self.hf_name
            + "/"
            + self.model_name
            + bla
            + "/"
            + str(seed)
        )

        self.config = AutoConfig.from_pretrained(self.hf_name)
        self.config.num_labels = len(self.dataset.all_labels)
        self.config.id2label = self.dataset.id2label
        self.config.label2id = self.dataset.label2id
        if "repeat" in self.model_name:
            self.config.k_repeat = self.k_repeat

        if (
            self.model_name == "bert"
            or self.model_name == "distilbert"
            or self.model_name == "modern-bert"
            or self.model_name == "roberta"
            or self.model_name == "qwen2"
            or self.model_name == "gemma"
            or self.model_name == "gemma2"
            or self.model_name == "mistral"
        ):
            from transformers import AutoModelForTokenClassification
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = AutoModelForTokenClassification.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)
            # self.model.classifier = self.model.classifier.to(torch.float32)
            # self.model.classifier.modules_to_save.default.load_state_dict(
            #    torch.load(self.full_dir + "/classifier.pt")
            # )
        elif self.model_name == "qwen3-normal":
            from transformers import AutoModelForTokenClassification
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = AutoModelForTokenClassification.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)
            # self.model.classifier = self.model.classifier.to(torch.float32)
            # self.model.classifier.modules_to_save.default.load_state_dict(
            #    torch.load(self.full_dir + "/classifier.pt")
            # )

        elif self.model_name == "qwen3":
            from unmasking import Qwen3ForTokenClassification
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3ForTokenClassification.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)
            # self.model.classifier = self.model.classifier.to(torch.float32)
            # self.model.classifier.modules_to_save.default.load_state_dict(
            #    torch.load(self.full_dir + "/classifier.pt")
            # )

        elif self.model_name == "qwen3-unmasked":
            from unmasking import Qwen3Unmasked
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3Unmasked.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)
            # self.model.classifier = self.model.classifier.to(torch.float32)
            # self.model.classifier.modules_to_save.default.load_state_dict(
            #    torch.load(self.full_dir + "/classifier.pt")
            # )

        elif (
            self.model_name == "qwen3-unmasked-middle"
            or self.model_name == "qwen3-unmasked-SSPFB"
        ):
            from unmasking import Qwen3MiddleUnmasked
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3MiddleUnmasked.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)
            # self.model.classifier = self.model.classifier.to(torch.float32)
            # self.model.classifier.modules_to_save.default.load_state_dict(
            #    torch.load(self.full_dir + "/classifier.pt")
            # )
            # print(self.model.classifier.modules_to_save.default.weight)

        elif self.model_name == "qwen3-alpha-mask":
            from unmasking import Qwen3AlphaMasking
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3AlphaMasking.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)
        elif self.model_name == "qwen3-summed-attention":
            from unmasking import Qwen3SummedAttention
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3SummedAttention.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif self.model_name == "qwen3-summed-headwise":
            from unmasking import Qwen3SummedHeadwise
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3SummedHeadwise.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif self.model_name == "qwen3-repeat-k":
            from unmasking import Qwen3RepeatK
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            if self.config.k_repeat == -1:
                raise Exception("WRONG REPEAT COUNT/NOT SET")

            self.org_model = Qwen3RepeatK.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif self.model_name == "gemma-unmasked":
            from unmasking import GemmaUnmasked
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.config.use_cache = False
            self.org_model = GemmaUnmasked.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif (
            self.model_name == "gemma-unmasked-middle"
            or self.model_name == "gemma-unmasked-SSPFB"
        ):
            from unmasking import GemmaUnmaskedMiddle
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.config.use_cache = False
            self.org_model = GemmaUnmaskedMiddle.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif self.model_name == "gemma-summed-headwise":
            from unmasking import GemmaHeadwise
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.config.use_cache = False
            self.org_model = GemmaHeadwise.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)
        elif self.model_name == "gemma-repeat-k":
            from unmasking import GemmaRepeatK
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            if self.config.k_repeat == -1:
                raise Exception("WRONG REPEAT COUNT/NOT SET")

            self.config.use_cache = False
            self.org_model = GemmaRepeatK.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif self.model_name == "gemma2-unmasked":
            from unmasking import Gemma2Unmasked
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.config.use_cache = False
            self.org_model = Gemma2Unmasked.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif (
            self.model_name == "gemma2-unmasked-middle"
            or self.model_name == "gemma2-unmasked-SSPFB"
        ):
            from unmasking import Gemma2UnmaskedMiddle
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.config.use_cache = False
            self.org_model = Gemma2UnmaskedMiddle.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif self.model_name == "gemma2-summed-headwise":
            from unmasking import Gemma2Headwise
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.config.use_cache = False
            self.org_model = Gemma2Headwise.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif self.model_name == "gemma2-repeat-k":
            from unmasking import Gemma2RepeatK
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            if self.config.k_repeat == -1:
                raise Exception("WRONG REPEAT COUNT/NOT SET")

            self.config.use_cache = False
            self.org_model = Gemma2RepeatK.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif self.model_name == "mistral-unmasked":
            from unmasking import MistralUnmasked
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.config.use_cache = False
            self.org_model = MistralUnmasked.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif (
            self.model_name == "mistral-unmasked-middle"
            or self.model_name == "mistral-unmasked-SSPFB"
        ):
            from unmasking import MistralUnmaskedMiddle
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.config.use_cache = False
            self.org_model = MistralUnmaskedMiddle.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif self.model_name == "mistral-summed-headwise":
            from unmasking import MistralHeadwise
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.config.use_cache = False
            self.org_model = MistralHeadwise.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

        elif self.model_name == "mistral-repeat-k":
            from unmasking import MistralRepeatK
            from peft import PeftModel

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            if self.config.k_repeat == -1:
                raise Exception("WRONG REPEAT COUNT/NOT SET")

            self.config.use_cache = False
            self.org_model = MistralRepeatK.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.model = PeftModel.from_pretrained(self.org_model, self.full_dir)

    def compute_metrics(self, p):

        label_list = self.dataset.all_labels

        predictions, labels = p
        preds = np.argmax(predictions, axis=2)

        # convert to list-of-lists (skipping -100)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        from seqeval.metrics import classification_report, accuracy_score
        from seqeval.scheme import IOB2

        # full-sequence report
        full_report = classification_report(
            true_labels, true_predictions, mode="strict", scheme=IOB2, output_dict=True
        )
        acc = accuracy_score(true_labels, true_predictions)

        valid_pairs = [
            (g, p) for g, p in zip(true_labels, true_predictions) if len(g) > 0
        ]
        fraction_results = {}

        if len(valid_pairs) == 0:
            for k in range(1, 9):
                fraction_results[f"micro_f1_{k}_of_8"] = None
        else:
            gold_seqs, pred_seqs = zip(*valid_pairs)

            for k in range(1, 9):
                prefix_gold = []
                prefix_pred = []
                frac = k / 8.0
                print(f"\n=== k = {k}/8 (taking ~{frac:.2f} of each sequence) ===")
                for i, (gold, pred) in enumerate(zip(gold_seqs, pred_seqs)):
                    L = len(gold)
                    n = math.ceil(L * frac)
                    g_prefix = gold[:n]
                    p_prefix = pred[:n]
                    prefix_gold.append(g_prefix)
                    prefix_pred.append(p_prefix)
                    print(f"seq {i} (len={L}) → n={n}")
                    print(f"  gold: {g_prefix}")
                    print(f"  pred: {p_prefix}")

                rep = classification_report(
                    prefix_gold,
                    prefix_pred,
                    mode="strict",
                    scheme=IOB2,
                    output_dict=True,
                )
                micro_f1 = rep.get("micro avg", {}).get("f1-score", None)
                print(
                    f"→ micro_f1_{k}_of_8 = {micro_f1:.4f}"
                    if micro_f1 is not None
                    else "→ no micro avg"
                )
                fraction_results[f"micro_f1_{k}_of_8"] = micro_f1

        results = {
            "results": {
                "micro_precision": full_report["micro avg"]["precision"],
                "micro_recall": full_report["micro avg"]["recall"],
                "micro_f1": full_report["micro avg"]["f1-score"],
                "accuracy": acc,
                **fraction_results,
            }
        }
        return results

    def setTrainingArgs(
        self,
        batch_size: int = 32,
    ):
        self.training_args = TrainingArguments(
            per_device_eval_batch_size=batch_size,
            label_names=["labels"],
            do_train=False,
            do_eval=True,
            logging_dir=None,
        )

    def setBnbConf4B(self):
        from transformers.utils.quantization_config import BitsAndBytesConfig

        compute_dtype = getattr(torch, "float16")
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

    def setLoraConfig(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        bias: Bias = "none",
        lora_dropout: float = 0.1,
        task_type: str = "TOKEN_CLS",
        target_modules: list[str] = ["q_proj", "v_proj"],  # ,"o_proj"],
        modules_to_save: list[str] = ["score"],
    ):

        from peft import LoraConfig

        self.loraConfig = LoraConfig(
            r=r,  # Rank
            lora_alpha=lora_alpha,
            bias=bias,
            lora_dropout=lora_dropout,
            task_type=task_type,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )

    # def createPeftAdapter(self, use_reentrant: bool = True):

    def eval(self, seed: int = 42):
        print(
            f"Evaluating model: {self.model_name}, from HF: {self.hf_name}, on seed: {seed}"
        )
        self.modelInstance(seed)
        set_seed(seed)
        random.seed(seed)
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=self.dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        if "summed" in self.model_name:
            layer_idx = 0
            for name, param in self.model.named_parameters():
                if "unmasking_logit" in name:
                    load = torch.load(
                        self.load_dir
                        + self.dataset_name
                        + "/"
                        + self.hf_name
                        + "/"
                        + self.model_name
                        + "/"
                        + str(seed)
                        + "/unmasking_logit_"
                        + str(layer_idx)
                        + ".pt",
                    )
                    param.copy_(load)
                    layer_idx += 1
        return trainer.evaluate()["eval_results"]

    def evalSeeds(
        self,
        seeds: list[int] = [5, 29, 42, 81, 123],
    ):
        final = {}
        for seed in seeds:
            dic = self.eval(seed=seed)
            final[seed] = dic
            try:
                del self.model
            except NameError:
                pass
            try:
                del self.org_model
            except Exception:
                pass
        return final
