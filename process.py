#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import List

import numpy as np
import pandas as pd
from dragon_baseline import DragonBaseline
from dragon_baseline.architectures.clf_multi_head import \
    AutoModelForMultiHeadSequenceClassification
from dragon_baseline.architectures.reg_multi_head import \
    AutoModelForMultiHeadSequenceRegression
from dragon_baseline.nlp_algorithm import ProblemType
from scipy.special import expit, softmax
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


class DragonCltlMedrobertaNlDomainSpecific(DragonBaseline):
    def __init__(self, **kwargs):
        """
        Adapt the DRAGON baseline to use the CLTL/MedRoBERTa.nl model.
        Note: when manually changing the model, update the Dockerfile to pre-download that model.
        """
        super().__init__(**kwargs)
        self.model_name = "CLTL/MedRoBERTa.nl"
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 2
        self.gradient_checkpointing = False
        self.max_seq_length = 512
        self.learning_rate = 1e-05

    def predict_huggingface(self, *, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the labels for the test data.
        
        Change w.r.t. the original implementation:
        tokenizer.model_max_length = self.max_seq_length
        """
        # load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_save_dir, truncation_side=self.task.recommended_truncation_side)
        tokenizer.model_max_length = self.max_seq_length  # set the maximum sequence length
        if self.task.target.problem_type == ProblemType.MULTI_LABEL_REGRESSION:
            model = AutoModelForMultiHeadSequenceRegression.from_pretrained(self.model_save_dir).to(self.device)
        elif self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
            model = AutoModelForMultiHeadSequenceClassification.from_pretrained(self.model_save_dir).to(self.device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_save_dir).to(self.device)

        # predict
        results = []
        for _, row in tqdm(df.iterrows(), desc="Predicting", total=len(df)):
            # tokenize inputs
            inputs = row[self.task.input_name] if self.task.input_name == "text_parts" else [row[self.task.input_name]]
            tokenized_inputs = tokenizer(*inputs, return_tensors="pt", truncation=True).to(self.device)

            # predict
            result: SequenceClassifierOutput = model(**tokenized_inputs)

            if self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
                logits: List[np.ndarray] = [logits.detach().cpu().numpy() for logits in result.logits]
            else:
                logits: np.ndarray = result.logits.detach().cpu().numpy()

            # convert to labels
            if self.task.target.problem_type == ProblemType.SINGLE_LABEL_REGRESSION:
                expected_shape = (1, 1)
                if logits.shape != expected_shape:
                    raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits.shape}")

                prediction = {self.task.target.prediction_name: logits[0][0]}
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_REGRESSION:
                expected_shape = (1, len(self.df_train[self.task.target.label_name].iloc[0]))
                if logits.shape != expected_shape:
                    raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits.shape}")

                prediction = {self.task.target.prediction_name: logits[0]}
            elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_BINARY_CLASSIFICATION:
                expected_shape = (1, 2)
                if logits.shape != expected_shape:
                    raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits.shape}")

                # calculate sigmoid to map the logits to [0, 1]
                prediction = softmax(logits, axis=-1)[0, 1]
                prediction = {self.task.target.prediction_name: prediction}
            elif self.task.target.problem_type == ProblemType.SINGLE_LABEL_MULTI_CLASS_CLASSIFICATION:
                expected_shape = (1, len(self.task.target.values))
                if logits.shape != expected_shape:
                    raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits.shape}")

                p = model.config.id2label[np.argmax(logits[0])]
                prediction = {self.task.target.prediction_name: p}
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_BINARY_CLASSIFICATION:
                expected_shape = (1, len(self.task.target.values))
                if logits.shape != expected_shape:
                    raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits.shape}")

                prediction = expit(logits)[0]  # calculate sigmoid to map the logits to [0, 1]
                prediction = {self.task.target.prediction_name: prediction}
            elif self.task.target.problem_type == ProblemType.MULTI_LABEL_MULTI_CLASS_CLASSIFICATION:
                expected_length = len(self.df_train[self.task.target.label_name].iloc[0])
                if len(logits) != expected_length:
                    raise ValueError(f"Expected logits to have length {expected_length}, but got {len(logits)}")
                label_names = [f"{self.task.target.label_name}_{i}" for i in range(len(logits))]
                for logits_, label_name in zip(logits, label_names):
                    expected_shape = (1, len(self.df_train[label_name].unique()))
                    if logits_.shape != expected_shape:
                        raise ValueError(f"Expected logits to have shape {expected_shape}, but got {logits_.shape}")

                preds = [np.argmax(p) for p in logits]
                prediction = {
                    self.task.target.prediction_name: [
                        id2label[str(p)]
                        for p, id2label in zip(preds, model.config.id2labels)
                    ]
                }
            else:
                raise ValueError(f"Unexpected problem type '{self.task.target.problem_type}'")

            results.append({"uid": row["uid"], **prediction})

        df_pred = pd.DataFrame(results)

        # scale the predictions (inverse of the normalization during preprocessing)
        df_pred = self.unscale_predictions(df_pred)

        return df_pred

if __name__ == "__main__":
    DragonCltlMedrobertaNlDomainSpecific().process()
