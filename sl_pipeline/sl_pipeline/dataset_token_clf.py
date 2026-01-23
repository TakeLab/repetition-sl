import os

from datasets import load_dataset
from torch.utils import data
from pathlib import Path

# change the working directory for HPC execution
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

REPO_HOME = Path.cwd()

# fighting OOM issues
# longest example is around 250 tokens for conll2003
MAX_LEN = 256

class TokenClassificationDataset(data.Dataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        split,
        mask_proba=None,
        sample_ontonotes=False,
    ):
        self.tokenizer = tokenizer
        self.tokens, self.labels, self.label_ids = [], [], []
        self.dataset_name = dataset_name
        self.split = split
        self.all_labels, self.id2label, self.label2id = self.get_labels(dataset_name)
        self.mask_proba = mask_proba
        self.sample_ontonotes = sample_ontonotes

        self.load_data_split(dataset_name, split)

    def get_labels(self, dataset_name):
        all_labels, id2label, label2id = [], [], []

        if dataset_name == "conll2003":
            label2id = {
                "O": 0,
                "B-PER": 1,
                "I-PER": 2,
                "B-ORG": 3,
                "I-ORG": 4,
                "B-LOC": 5,
                "I-LOC": 6,
                "B-MISC": 7,
                "I-MISC": 8,
            }

            id2label = {v: k for k, v in label2id.items()}

            all_labels = list(label2id.keys())

        elif dataset_name == "wnut2017":
            label2id = {
                "O": 0,
                "B-corporation": 1,
                "I-corporation": 2,
                "B-creative-work": 3,
                "I-creative-work": 4,
                "B-group": 5,
                "I-group": 6,
                "B-location": 7,
                "I-location": 8,
                "B-person": 9,
                "I-person": 10,
                "B-product": 11,
                "I-product": 12,
            }

            id2label = {v: k for k, v in label2id.items()}

            all_labels = list(label2id.keys())

        elif "aac" in dataset_name:
            topic = dataset_name.replace("aac", "")

            if topic == "ab":
                self.dataset_classes = [
                    "safety/health_effects_of_legal_abortion",
                    "consequences_of_childbirth",
                    "fetal_defects/disabilities",
                    "adoption",
                    "fetal/newborn_rights",
                    "rape",
                    "responsibility",
                    "funding_of_abortion",
                    "health_effects_of_pregnancy/childbirth",
                    "psychological_effects_of_abortion",
                    "parental_consent",
                    "illegal_abortions",
                    "moral/ethical_values",
                    "abortion_industry",
                    "bodily_autonomy/women's_rights",
                ]
            elif topic == "mj":
                self.dataset_classes = [
                    "drug_abuse",
                    "health/psychological_effects",
                    "personal_freedom",
                    "child_and_teen_safety",
                    "legal_drugs",
                    "medical_marijuana",
                    "addiction",
                    "national_budget",
                    "community/sociatal_effects",
                    "harm",
                    "gateway_drug",
                    "illegal_trade",
                    "drug_policy",
                ]
            elif topic == "mw":
                self.dataset_classes = [
                    "capital_vs_labor",
                    "social_justice/injustice",
                    "economic_impact",
                    "prices",
                    "low_skilled",
                    "turnover",
                    "government",
                    "youth_and_secondary_wage_earners",
                    "competition/business_challenges",
                    "motivation/chances",
                    "welfare",
                    "un/employment_rate",
                ]
            elif topic == "ne":
                self.dataset_classes = [
                    "waste",
                    "weapons",
                    "costs",
                    "health_effects",
                    "environmental_impact",
                    "reliability",
                    "fossil_fuels",
                    "accidents/security",
                    "technological_innovation",
                    "public_debate",
                    "energy_policy",
                    "renewables",
                ]

            self.dataset_classes = [x.lower() for x in self.dataset_classes]

            all_labels = ["O"] + [
                x for item in self.dataset_classes for x in ("B-" + item, "I-" + item)
            ]

            label2id = {all_labels[k]: k for k in range(len(all_labels))}

            id2label = {v: k for k, v in label2id.items()}

        elif "nlupp" in dataset_name:
            topic = dataset_name.replace("nlupp", "")

            if topic == "banking":
                self.dataset_classes = [
                    "time_from",
                    "time_period",
                    "person_name",
                    "shopping_category",
                    "date_from",
                    "date",
                    "amount_of_money",
                    "date_to",
                    "date_period",
                    "time",
                    "company_name",
                    "number",
                    "time_to",
                ]
            elif topic == "hotels":
                self.dataset_classes = [
                    "time_from",
                    "time_period",
                    "person_name",
                    "date_from",
                    "date",
                    "adults",
                    "rooms",
                    "kids",
                    "people",
                    "date_to",
                    "date_period",
                    "time",
                    "number",
                    "time_to",
                ]
            else:
                self.dataset_classes = [
                    "time_from",
                    "person_name",
                    "shopping_category",
                    "date_from",
                    "date",
                    "number",
                    "adults",
                    "rooms",
                    "amount_of_money",
                    "kids",
                    "people",
                    "date_to",
                    "date_period",
                    "time",
                    "company_name",
                    "time_period",
                    "time_to",
                ]

            label2id = {
                "O": 0,
                **{
                    f"B-{label}": i * 2 + 1
                    for i, label in enumerate(self.dataset_classes)
                },
                **{
                    f"I-{label}": i * 2 + 2
                    for i, label in enumerate(self.dataset_classes)
                },
            }

            id2label = {v: k for k, v in label2id.items()}

            all_labels = list(label2id.keys())
        elif dataset_name == "ontonotes":
            self.dataset_classes = [
                "ARG0",
                "ARG1",
                "ARG2",
                "ARG3",
                "ARG4",
                "ARGM-ADJ",
                "ARGM-ADV",
                "ARGM-CAU",
                "ARGM-COM",
                "ARGM-DIR",
                "ARGM-DIS",
                "ARGM-EXT",
                "ARGM-GOL",
                "ARGM-LOC",
                "ARGM-MNR",
                "ARGM-MOD",
                "ARGM-NEG",
                "ARGM-PNC",
                "ARGM-PRD",
                "ARGM-PRP",
                "ARGM-TMP",
                "C-ARG0",
                "C-ARG1",
                "C-ARG2",
                "R-ARG0",
                "R-ARG1",
                "V",
            ]

            label2id = {
                "O": 0,
                **{
                    f"B-{label}": i * 2 + 1
                    for i, label in enumerate(self.dataset_classes)
                },
                **{
                    f"I-{label}": i * 2 + 2
                    for i, label in enumerate(self.dataset_classes)
                },
            }

            id2label = {v: k for k, v in label2id.items()}

            all_labels = list(label2id.keys())
        elif dataset_name == "absa-restaurants":
            self.dataset_classes = ["positive", "negative", "neutral", "conflict"]

            label2id = {
                "O": 0,
                **{
                    f"B-{label}": i * 2 + 1
                    for i, label in enumerate(self.dataset_classes)
                },
                **{
                    f"I-{label}": i * 2 + 2
                    for i, label in enumerate(self.dataset_classes)
                },
            }

            id2label = {v: k for k, v in label2id.items()}

            all_labels = list(label2id.keys())
        elif "ace" in dataset_name:
            self.dataset_classes = [x.lower() for x in [
                "Business:Merge-Org",
                "Business:Start-Org",
                "Business:Declare-Bankruptcy",
                "Business:End-Org",
                "Justice:Pardon",
                "Justice:Extradite",
                "Justice:Execute",
                "Justice:Fine",
                "Justice:Trial-Hearing",
                "Justice:Sentence",
                "Justice:Appeal",
                "Justice:Convict",
                "Justice:Sue",
                "Justice:Release-Parole",
                "Justice:Arrest-Jail",
                "Justice:Charge-Indict",
                "Justice:Acquit",
                "Conflict:Demonstrate",
                "Conflict:Attack",
                "Contact:Phone-Write",
                "Contact:Meet",
                "Personnel:Start-Position",
                "Personnel:Elect",
                "Personnel:End-Position",
                "Personnel:Nominate",
                "Transaction:Transfer-Ownership",
                "Transaction:Transfer-Money",
                "Life:Marry",
                "Life:Divorce",
                "Life:Be-Born",
                "Life:Die",
                "Life:Injure",
                "Movement:Transport",
            ]]
            label2id = {
                "O": 0,
                **{
                    f"B-{label}": i * 2 + 1
                    for i, label in enumerate(self.dataset_classes)
                },
                **{
                    f"I-{label}": i * 2 + 2
                    for i, label in enumerate(self.dataset_classes)
                },
            }

            id2label = {v: k for k, v in label2id.items()}

            all_labels = list(label2id.keys())
        return all_labels, id2label, label2id

    def load_conll2003(self, dataset_name, split):
        data_split = load_dataset("conll2003", split=split, trust_remote_code=True)

        self.tokens = data_split["tokens"]
        if dataset_name == "conll2003":
            self.label_ids = data_split["ner_tags"]

        for l in self.label_ids:
            self.labels.append([self.id2label[t] for t in l])

    def load_wnut2017(self, dataset_name, split):
        data_split = load_dataset("wnut_17", split=split, trust_remote_code=True)

        self.tokens = data_split["tokens"]
        if dataset_name == "wnut2017":
            self.label_ids = data_split["ner_tags"]

        for l in self.label_ids:
            self.labels.append([self.id2label[t] for t in l])

    def load_aac(self, dataset_name, split):
        # either train on one topic or all topics
        if dataset_name != "aac":
            topics = [dataset_name.replace("aac", "")]
        else:
            topics = ["ab", "mj", "mw", "ne"]

        sentences = []

        for topic in topics:
            with open(
                os.path.join(REPO_HOME, "data/processed/aac", topic, split + ".txt"),
                "r",
            ) as f:
                data = f.read()
                sentences.extend([x.split("\n") for x in data.split("\n\n") if x])

        self.tokens = [[x.split("\t")[1].strip() for x in y] for y in sentences]
        self.labels = [
            self.convert_to_iob2(
                [
                    "_".join(x.split("\t")[2].strip().replace("-", " ").split())
                    for x in y
                ]
            )
            for y in sentences
        ]

        for l in self.labels:
            self.label_ids.append([self.label2id[t] for t in l])

    def load_nlupp(self, dataset_name, split):
        import json

        import spacy

        # variable for tokenization
        nlp = spacy.load("en_core_web_lg")

        if dataset_name != "nlupp":
            topics = [dataset_name.replace("nlupp", "")]
        else:
            topics = ["banking", "hotels"]

        sentences = []
        slots = []

        for topic in topics:
            for fold in os.listdir(
                os.path.join(REPO_HOME, f"data/processed/nlupp/{topic}/{split}")
            ):
                with open(
                    os.path.join(
                        REPO_HOME, f"data/processed/nlupp/{topic}/{split}/{fold}"
                    ),
                    "r",
                ) as f:
                    data = json.load(f)
                    for item in data:
                        sentences.append(item["text"])
                        slots.append(item.get("slots", {}))

        slots = [
            (
                {
                    v["text"]: (k, v["span"])
                    for k, v in dict(
                        sorted(x.items(), key=lambda item: item[1]["span"])
                    ).items()
                }
                if x
                else {}
            )
            for x in slots
        ]

        self.tokens, self.labels = [], []

        for i in range(len(sentences)):
            sent_toks, sent_tags = self.tokenize_create_bio_tags(
                sentences[i], slots[i], nlp
            )
            self.tokens.append(sent_toks)
            self.labels.append(sent_tags)

        for l in self.labels:
            self.label_ids.append([self.label2id[t] for t in l])

    def load_ontonotes(self, split):
        import pandas as pd

        data_split = pd.read_csv(
            os.path.join(REPO_HOME, "data/processed/ontonotes", split + ".csv"),
            converters={"tokens": eval, "BIO_tags": eval},
        )

        if self.sample_ontonotes:
            # always sample the same
            data_split = data_split.sample(6000, random_state=42)

        self.tokens = data_split["tokens"].tolist()
        self.verbs = data_split["verb"].tolist()
        self.labels = [[y for y in x] for x in data_split["BIO_tags"].tolist()]
        for l in self.labels:
            self.label_ids.append([self.label2id[t] for t in l])

    def load_absa(self, split):
        from ast import literal_eval

        import pandas as pd

        if split == "train":
            fpath = os.path.join(REPO_HOME, "data/processed/absa-restaurants/train.csv")
        elif split == "validation":
            fpath = os.path.join(REPO_HOME, "data/processed/absa-restaurants/valid.csv")
        else:
            # split must be test
            fpath = os.path.join(REPO_HOME, "data/processed/absa-restaurants/test.csv")

        df = pd.read_csv(fpath)

        df["tokens"] = df["tokens"].apply(literal_eval)
        df["tags"] = df["tags"].apply(literal_eval)

        self.tokens = df.tokens.tolist()
        self.labels = df.tags.tolist()
        for x in self.labels:
            self.label_ids.append([self.label2id[y] for y in x])

    def load_ace(self, split):
        import json
        if split == "train":
            fpath = os.path.join(REPO_HOME, "data/processed/ace/train.json")
        elif split == "validation":
            fpath = os.path.join(REPO_HOME, "data/processed/ace/dev.json")
        else:
            # split must be test
            fpath = os.path.join(REPO_HOME, "data/processed/ace/test.json")

        with open(fpath, "r") as f:
            data = json.load(f)
            for item in data:
                words = item["words"]
                tags = ["O"] * len(words)

                for event_mention in item["golden-event-mentions"]:
                    for i in range(
                        event_mention["trigger"]["start"],
                        event_mention["trigger"]["end"],
                    ):
                        trigger_type = event_mention["event_type"]
                        if i == event_mention["trigger"]["start"]:
                            tags[i] = "B-{}".format(trigger_type.lower())
                        else:
                            tags[i] = "I-{}".format(trigger_type.lower())

                self.tokens.append(words)
                self.labels.append(tags)
                self.label_ids.append([self.label2id[x] for x in tags])

    def load_data_split(self, dataset_name, split):
        if "conll2003" in dataset_name:
            self.load_conll2003(dataset_name, split)
        elif "wnut2017" in dataset_name:
            self.load_wnut2017(dataset_name, split)
        elif "aac" in dataset_name:
            self.load_aac(dataset_name, split)
        elif "nlupp" in dataset_name:
            self.load_nlupp(dataset_name, split)
        elif "ontonotes" in dataset_name:
            self.load_ontonotes(split)
        elif "absa-restaurants" in dataset_name:
            self.load_absa(split)
        elif "ace" in dataset_name:
            self.load_ace(split)

    def convert_to_iob2(self, tags):
        iob2_tags = []
        prev_tag = "O"

        for tag in tags:
            if tag == "O":
                iob2_tags.append("O")
            elif tag != prev_tag:
                iob2_tags.append(f"B-{tag.lower()}")
            else:
                iob2_tags.append(f"I-{tag.lower()}")

            prev_tag = tag

        return iob2_tags

    def tokenize_create_bio_tags(self, text, args, nlp):
        tokens = nlp(text)
        bio_tags = ["O"] * len(tokens)
        tokens_text = []

        error = False

        for key in args.keys():
            label = args[key][0]
            args_start, args_end = args[key][1][0], args[key][1][1]
            out = tokens.char_span(args_start, args_end)
            if out is not None:
                start_tok, end_tok = out.start, out.end
                if start_tok == end_tok:
                    bio_tags[start_tok] = "B-" + label
                else:
                    bio_tags[start_tok] = "B-" + label
                    for j in range(start_tok + 1, end_tok):
                        bio_tags[j] = "I-" + label
            else:
                print("Error for sentence", text)
                error = True

        for t in tokens:
            tokens_text.append(t.text)

        if error:
            print(tokens_text, args, bio_tags)

        return tokens_text, bio_tags

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens_idx, labels_idx = self.tokens[idx], self.label_ids[idx]

        example_tokenized = self.tokenize_and_align_labels(tokens_idx, labels_idx)

        if self.mask_proba is not None:
            example_tokenized["mask_proba"] = self.mask_proba
        # take just the head word of verb
        if hasattr(self, "verbs"):
            example_tokenized["verb"] = self.tokenizer(
                self.verbs[idx], add_special_tokens=False
            )["input_ids"][0]

        return example_tokenized

    def tokenize_and_align_labels(self, tokens, tags_id):
        """Taken from https://huggingface.co/docs/transformers/tasks/token_classification"""
        tokenized_input = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True, max_length=MAX_LEN
        )

        word_ids = tokenized_input.word_ids(
            batch_index=0
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                label_ids.append(tags_id[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        tokenized_input["labels"] = label_ids
        return tokenized_input
