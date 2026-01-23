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


class TrainModelDataset:

    def __init__(
        self,
        dataset_name: str = "wnut2017",
        model_name: str = "roberta",
        hf_name: str = "FacebookAI/roberta-large",
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
        self.k_repeat = k_repeat
        self.tokenizerInstance()
        self.datasetInstance()
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

    def datasetInstance(self):
        self.train_dataset = TokenClassificationDataset(
            dataset_name=self.dataset_name, tokenizer=self.tokenizer, split="train"
        )
        self.val_dataset = TokenClassificationDataset(
            dataset_name=self.dataset_name, tokenizer=self.tokenizer, split="validation"
        )

    def setPeftModel(self):
        from peft import prepare_model_for_kbit_training, get_peft_model

        self.org_model = prepare_model_for_kbit_training(self.org_model)
        if self.loraConfig is None:
            raise Exception("LORA CONFIG NOT SET!")
        self.model = get_peft_model(self.org_model, self.loraConfig)

    def modelInstance(self):
        from transformers import AutoConfig

        self.config = AutoConfig.from_pretrained(self.hf_name)
        self.config.num_labels = len(self.train_dataset.all_labels)
        self.config.id2label = self.train_dataset.id2label
        self.config.label2id = self.train_dataset.label2id
        self.config.classifier_dropout = 0.1
        if "repeat" in self.model_name:
            self.config.k_repeat = self.k_repeat

        if (
            self.model_name == "bert"
            or self.model_name == "distilbert"
            or self.model_name == "roberta"
            or self.model_name == "modern-bert"
            or self.model_name == "qwen2"
            or self.model_name == "gemma"
            or self.model_name == "gemma2"
            or self.model_name == "mistral"
        ):
            from transformers import AutoModelForTokenClassification

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = AutoModelForTokenClassification.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()
        elif self.model_name == "qwen3-normal":
            from transformers import AutoModelForTokenClassification

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = AutoModelForTokenClassification.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "qwen3":
            from unmasking import Qwen3ForTokenClassification

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3ForTokenClassification.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "qwen3-unmasked":
            from unmasking import Qwen3Unmasked

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3Unmasked.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "qwen3-eager":
            from unmasking import Qwen3Eager

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3Eager.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "qwen3-unmasked-eager":
            from unmasking import Qwen3UnmaskedEager

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3UnmaskedEager.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "qwen3-unmasked-middle":
            from unmasking import Qwen3MiddleUnmasked

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3MiddleUnmasked.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()
        elif self.model_name == "qwen3-unmasked-SSPFB":
            from unmasking import Qwen3SSPFB

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3SSPFB.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()
        elif self.model_name == "qwen3-alpha-mask":
            from unmasking import Qwen3AlphaMasking

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3AlphaMasking.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.org_model.initialize_alphas()
            self.setPeftModel()
            for name, param in self.model.named_parameters():
                if "alpha" in name:
                    param.requires_grad = True
        elif self.model_name == "qwen3-summed-attention":
            from unmasking import Qwen3SummedAttention

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3SummedAttention.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.org_model.initialize_unm()
            self.setPeftModel()
            for name, param in self.model.named_parameters():
                if "unmasking_logit" in name:
                    param.requires_grad = True
        elif self.model_name == "qwen3-summed-headwise":
            from unmasking import Qwen3SummedHeadwise

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3SummedHeadwise.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.org_model.initialize_unm()
            self.setPeftModel()
            for name, param in self.model.named_parameters():
                if "unmasking_logit" in name:
                    param.requires_grad = True
        elif self.model_name == "qwen3-SUM":
            from unmasking import Qwen3SUM

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Qwen3SUM.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "qwen3-repeat-k":
            from unmasking import Qwen3RepeatK

            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            if self.config.k_repeat == -1:
                raise Exception("WRONG REPEAT COUNT/NOT SET")

            self.org_model = Qwen3RepeatK.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "gemma-unmasked":
            from unmasking import GemmaUnmasked

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = GemmaUnmasked.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "gemma-unmasked-middle":
            from unmasking import GemmaUnmaskedMiddle

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = GemmaUnmaskedMiddle.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "gemma-unmasked-SSPFB":
            from unmasking import GemmaSSPFB

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = GemmaSSPFB.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "gemma-summed-headwise":
            from unmasking import GemmaHeadwise

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = GemmaHeadwise.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.org_model.initialize_unm()
            self.setPeftModel()
            for name, param in self.model.named_parameters():
                if "unmasking_logit" in name:
                    param.requires_grad = True
        elif self.model_name == "gemma-repeat-k":
            from unmasking import GemmaRepeatK

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            if self.config.k_repeat == -1:
                raise Exception("WRONG REPEAT COUNT/NOT SET")

            self.org_model = GemmaRepeatK.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "gemma2-unmasked":
            from unmasking import Gemma2Unmasked

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Gemma2Unmasked.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "gemma2-unmasked-middle":
            from unmasking import Gemma2UnmaskedMiddle

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Gemma2UnmaskedMiddle.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "gemma2-unmasked-SSPFB":
            from unmasking import Gemma2SSPFB

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Gemma2SSPFB.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "gemma2-summed-headwise":
            from unmasking import Gemma2Headwise

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = Gemma2Headwise.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.org_model.initialize_unm()
            self.setPeftModel()
            for name, param in self.model.named_parameters():
                if "unmasking_logit" in name:
                    param.requires_grad = True

        elif self.model_name == "gemma2-repeat-k":
            from unmasking import Gemma2RepeatK

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            if self.config.k_repeat == -1:
                raise Exception("WRONG REPEAT COUNT/NOT SET")

            self.org_model = Gemma2RepeatK.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "mistral-unmasked":
            from unmasking import MistralUnmasked

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = MistralUnmasked.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "mistral-unmasked-middle":
            from unmasking import MistralUnmaskedMiddle

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = MistralUnmaskedMiddle.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()
        elif self.model_name == "mistral-unmasked-SSPFB":
            from unmasking import MistralSSPFB

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = MistralSSPFB.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

        elif self.model_name == "mistral-summed-headwise":
            from unmasking import MistralHeadwise

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            self.org_model = MistralHeadwise.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.org_model.initialize_unm()
            self.setPeftModel()
            for name, param in self.model.named_parameters():
                if "unmasking_logit" in name:
                    param.requires_grad = True
        elif self.model_name == "mistral-repeat-k":
            from unmasking import MistralRepeatK

            self.config.use_cache = False
            if self.bnb_config is None:
                raise Exception("BNB CONFIG NOT SET!")
            if self.config.k_repeat == -1:
                raise Exception("WRONG REPEAT COUNT/NOT SET")

            self.org_model = MistralRepeatK.from_pretrained(
                self.hf_name, config=self.config, quantization_config=self.bnb_config
            )

            self.setPeftModel()

    def compute_metrics(self, p):
        label_list = self.train_dataset.all_labels
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # results = seqeval.compute(predictions=true_predictions, references=true_labels)
        from seqeval.metrics import classification_report
        from seqeval.scheme import IOB2
        from seqeval.metrics import accuracy_score

        results = classification_report(
            true_labels, true_predictions, mode="strict", scheme=IOB2, output_dict=True
        )
        acc = accuracy_score(true_labels, true_predictions)
        # results = seqeval.compute(
        #    predictions=true_predictions,
        #    references=true_labels,
        #    mode="strict",
        #    scheme="IOB2",
        # )
        # print(results)
        return {
            "micro_precision": results["micro avg"]["precision"],
            "micro_recall": results["micro avg"]["recall"],
            "micro_f1": results["micro avg"]["f1-score"],
            "accuracy": acc,
        }

    def setTrainingArgs(
        self,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 6,
        weight_decay: float = 0.05,
        save_strategy: str = "epoch",
        logging_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        gradient_accumulation_steps: int = 1,
        fp16: bool = False,
        output_dir: str = "../trained_models/",
    ):
        self.training_args = TrainingArguments(
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model="micro_f1",
            greater_is_better=True,
            push_to_hub=False,
            label_names=["labels"],
            eval_strategy="epoch",
            save_strategy=save_strategy,
            logging_strategy=logging_strategy,
            fp16=fp16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            gradient_accumulation_steps=gradient_accumulation_steps,
            output_dir=output_dir + "trainer_output/",
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
        target_modules: list[str] = [
            "q_proj",
            "v_proj",
            "o_proj",
            "k_proj",
        ],  # ,"o_proj"],
        modules_to_save: list[str] = ["classifier", "score"],
    ):

        from peft import LoraConfig

        if "distil" in self.model_name:
            target_modules = ["q_lin", "v_lin", "k_lin", "out_lin"]
        elif "bert" in self.model_name and "modern" not in self.model_name:
            target_modules = ["query", "value", "key", "attention.output.dense"]
        elif "modern" in self.model_name:
            target_modules = ["Wqkv", "Wo"]
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

    def train(
        self, seed: int = 42, save: bool = True, save_dir: str = "../trained_models/"
    ):
        print(
            f"Training model: {self.model_name}, from HF: {self.hf_name}, on seed: {seed}"
        )

        set_seed(seed)
        random.seed(seed)
        self.modelInstance()

        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        if self.model_name == "qwen3-alpha-mask":
            alpha_params, base_params = [], []
            for n, p in self.model.named_parameters():
                if "alpha" in n and p.requires_grad:
                    alpha_params.append(p)
                elif p.requires_grad:
                    base_params.append(p)

            optimizer_grouped_parameters = [
                {"params": base_params, "lr": self.training_args.learning_rate},
                {
                    "params": alpha_params,
                    "lr": 5e-3,
                },
            ]

            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
                eps=self.training_args.adam_epsilon,
                weight_decay=self.training_args.weight_decay,
            )
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                processing_class=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                optimizers=(optimizer, None),
            )
        elif "summed" in self.model_name:
            alpha_params, base_params = [], []
            for n, p in self.model.named_parameters():
                if "unmasking_logit" in n and p.requires_grad:
                    alpha_params.append(p)
                elif p.requires_grad:
                    base_params.append(p)

            optimizer_grouped_parameters = [
                {"params": base_params, "lr": self.training_args.learning_rate},
                {
                    "params": alpha_params,
                    "lr": 5e-2,
                },
            ]

            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
                eps=self.training_args.adam_epsilon,
                weight_decay=self.training_args.weight_decay,
            )
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                processing_class=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                optimizers=(optimizer, None),
            )

        elif (
            self.model_name == "qwen3-unmasked-SSPFB"
            or self.model_name == "gemma-unmasked-SSPFB"
            or self.model_name == "gemma2-unmasked-SSPFB"
            or self.model_name == "mistral-unmasked-SSPFB"
        ):
            from unmasking import TrainerSSPFB

            trainer = TrainerSSPFB(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                processing_class=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                num_hidden_layers=self.config.num_hidden_layers,
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                processing_class=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )

        trainer.train()
        if save:
            print(f"saving with {self.dataset_name}")
            bla = "-" + str(self.k_repeat) if "repeat" in self.model_name else ""
            trainer.model.save_pretrained(
                save_dir
                + self.dataset_name
                + "/"
                + self.hf_name
                + "/"
                + self.model_name
                + bla
                + "/"
                + str(seed),
                save_embedding_layers=True,
            )
            if "summed" in self.model_name:
                print("saving")
                layer_idx = 0
                for name, param in self.model.named_parameters():
                    if "unmasking_logit" in name:
                        save = param.detach().cpu().clone()
                        torch.save(
                            save,
                            save_dir
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
                        layer_idx += 1

    def trainSeeds(
        self,
        seeds: list[int] = [5, 29, 42, 81, 123],
        save_dir: str = "../trained_models/",
    ):
        for seed in seeds:
            self.train(seed=seed, save_dir=save_dir)
            try:
                del self.model
            except NameError:
                pass
            try:
                del self.org_model
            except Exception:
                pass
