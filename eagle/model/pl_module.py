import copy
import json
import logging
from typing import *

import bitsandbytes as bnb
import hkkang_utils.file as file_utils
import lightning as L
import torch
from omegaconf import DictConfig
from torch.optim.swa_utils import SWALR, AveragedModel
from transformers import EvalPrediction, get_linear_schedule_with_warmup

from eagle.dataset.corpus import Corpus
from eagle.metrics.utils import (
    aggregate_final_metrics,
    aggregate_intermediate_metrics,
    compute_metrics,
)
from eagle.model.base_model import BaseModel
from eagle.model.registry import MODEL_REGISTRY
from eagle.model.utils import (
    append_dummy_pid,
    pid_found_percentage,
    unwrap_logging_items,
)
from eagle.phrase.noun import SpacyModel
from eagle.search.base_searcher import BaseSearcher
from eagle.search.plaid import PLAID
from eagle.search.registry import SEARCHER_REGISTRY
from eagle.tokenization.tokenizers import Tokenizers
from eagle.utils import handle_old_ckpt, remove_key_with_none_value
from scripts.utils import preprocess, pretty_print_tokens_with_their_indices

CAPABILITY = torch.cuda.get_device_capability()

logger = logging.getLogger("PLModule")


class LightningNewModel(L.LightningModule):
    def __init__(
        self, cfg: DictConfig, train_batch_num: int = None, index_dir_path: str = None
    ):
        super().__init__()
        ## Need to set this to False to avoid the automatic optimization
        self.automatic_optimization = False
        # Set configurations
        self.cfg = cfg.training
        self.dataset_name = cfg.dataset.name
        self.train_batch_num = train_batch_num
        # Tmp
        self.tokenizers = Tokenizers(
            cfg.tokenizers.query, cfg.tokenizers.document, cfg.model.backbone_name
        )
        # Load model
        self.model: BaseModel = MODEL_REGISTRY[cfg.model.name](
            cfg=cfg.model, tokenizers=self.tokenizers
        )  # Initialize your model with required args
        if (
            "use_torch_compile" in self.cfg
            and self.cfg.use_torch_compile
            and CAPABILITY[0] >= 7
        ):
            self.model = torch.compile(self.model, dynamic=True)

        self.swa_model = (
            AveragedModel(self.model)
            if handle_old_ckpt(self.cfg, "is_use_swa")
            else None
        )
        self.optim_class = (
            bnb.optim.AdamW8bit
            if handle_old_ckpt(cfg.model, "is_use_quantization")
            else torch.optim.AdamW
        )
        # Save hyperparameters
        self.save_hyperparameters(cfg)
        # For end-to-end retrieval during test (after training & indexing the corpus)
        self.index_dir_path = index_dir_path
        self.searcher = None
        self.final_eval_results: List[Dict[str, float]] = []
        self.intermediate_eval_results: List[Dict[str, float]] = []
        # For debugging
        self.dataset_cfg = cfg
        self.corpus: Optional[Corpus] = None

    def _load_searcher(self) -> PLAID:
        # Load the searcher
        self.searcher: BaseSearcher = SEARCHER_REGISTRY[self.model.cfg.name](
            cfg=self.cfg, model=self.model, index_dir_path=self.index_dir_path
        )
        return self.searcher

    def _test_reranking(self, batch: Dict, batch_idx: int) -> None:
        bsize = len(batch["q_tok_ids"])

        return_dict, scores = self.model(**batch, is_analyze=True)

        # Compute accuracy for each item in the batch (for DDP)
        metrics: List[Dict[str, float]] = []
        for b_idx in range(bsize):
            eval_preds = EvalPrediction(
                scores[b_idx : b_idx + 1], batch["labels"][b_idx : b_idx + 1]
            )
            tmp = compute_metrics(eval_preds, prefix="test")
            metrics.append(tmp)

        # Log metrics
        # self.log_dict(metrics, batch_size=bsize, on_step=False, on_epoch=True)
        self.final_eval_results.append(metrics)
        is_analyze = False
        if is_analyze:
            assert (
                len(batch["q_tok_ids"]) == 1
            ), f"Only one query is supported for analysis, but found {len(batch['q_id'])}"
            # query = self.tokenizers.q_tokenizer.tokenizer.decode(
            #     batch["q_tok_ids"][0], skip_special_tokens=True
            # )
            q_toks = self.tokenizers.q_tokenizer.tokenizer.convert_ids_to_tokens(
                batch["q_tok_ids"][0]
            )
            # Find the rank of the positive document (i.e., the first document in the list)
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            sorted_scores.squeeze()
            sorted_indices = sorted_indices.squeeze()
            pos_rank = (sorted_indices == 0).nonzero(as_tuple=False)[0][0].item()

            loop_num = min(10, pos_rank)
            logger.info(f"Positive rank: {pos_rank}")
            for loop_idx in range(loop_num + 1):
                logger.info(f"\nRank: {loop_idx}")
                # Show the text for the top-10 documents
                doc_idx = sorted_indices[loop_idx].item()
                doc_ids = batch["doc_tok_ids"][0][doc_idx]
                decoded_doc_tokens = (
                    self.tokenizers.d_tokenizer.tokenizer.convert_ids_to_tokens(doc_ids)
                )

                qd_scores = return_dict["intra_qd_scores"][doc_idx]

                # Find the max scores for each token
                max_scores, max_indices = qd_scores.max(0)

                # Find the total score of the query-document
                total_score = max_scores[batch["q_tok_mask"][0] == 0].sum().item()

                # Show the max document token and score for each query token
                logger.info(f"Total score: {total_score}")
                for i, (max_score, max_idx) in enumerate(zip(max_scores, max_indices)):
                    doc_token = decoded_doc_tokens[max_idx]
                    q_tok = q_toks[i]
                    q_tok_mask = batch["q_tok_mask"][0][i].item()
                    if q_tok_mask == 1:
                        continue
                    logger.info(
                        f"Q Token {i:2d}: {q_tok:<10}\t->\tD token {max_idx:2d}: {doc_token:<10}\t(Score: {max_score:.5f})"
                    )

                print("Query tokens:")
                pretty_print_tokens_with_their_indices(
                    decoded_tokens=q_toks, max_tokens_per_line=20
                )
                print("Document tokens:")
                pretty_print_tokens_with_their_indices(
                    decoded_tokens=decoded_doc_tokens, max_tokens_per_line=20
                )
                print("")

        return None

    def _test_full_retrieval(self, batch: Dict, batch_idx: int) -> None:
        bsize = len(batch["pos_doc_ids"])
        # Load the searcher if not loaded
        if self.searcher is None:
            self._load_searcher()

        # if self.corpus is None:
        #     import os

        #     from eagle.dataset.utils import read_corpus

        #     print("Reading corpus...")
        #     self.corpus = read_corpus(
        #         os.path.join(
        #             self.dataset_cfg.dataset.dir_path,
        #             self.dataset_cfg.dataset.name,
        #             self.dataset_cfg.dataset.corpus_file,
        #         )
        #     )
        #     print("Done reading corpus!")

        # Perform search on the index
        all_pids, all_scores, all_qd_scores, all_intermediate_pids = self.searcher(
            **batch
        )

        # Analyze
        # Get pid
        # pid = all_pids[0][0].item()
        # # Get the document
        # doc = " ".join(self.corpus[str(pid)])
        # tmp = preprocess(doc, self.tokenizers.d_tokenizer, False)

        # Post-process the results if the dataset is BEIR-ArguAna
        if self.dataset_name == "beir-arguana":
            new_pids = []
            new_scores = []
            for pids, scores in zip(all_pids, all_scores, strict=True):
                # Remove the rank 1 document (i.e., the same text as the query)
                pids = torch.cat([pids[1:], pids[0:1]], dim=0)
                scores = torch.cat(
                    [
                        scores[1:],
                        torch.tensor([0], device=scores.device, dtype=scores.dtype),
                    ],
                    dim=0,
                )
                new_pids.append(pids)
                new_scores.append(scores)
            all_pids = new_pids
            all_scores = new_scores
        # Prepare evaluation
        # Get max positive document number
        max_pos_doc_num = max(
            [len(pos_doc_idxs) for pos_doc_idxs in batch["pos_doc_ids"]]
        )
        # number of positive doc ids to append
        num_pids_to_append = (
            max_pos_doc_num
            + max(self.searcher.plaid.ndocs, len(all_pids[0]))
            - len(all_pids[0])
        )
        # Append the positive doc id at the end if not found.
        # This is for correctly format the input for the evaluation script)
        new_pids = []
        new_scores = []
        all_labels = []
        for b_idx, (pids, scores) in enumerate(zip(all_pids, all_scores, strict=True)):
            pids, pos_indices = append_dummy_pid(
                pids=pids,
                target_pids=[int(item) for item in batch["pos_doc_ids"][b_idx]],
                max_num=num_pids_to_append,
            )
            scores = torch.cat(
                [
                    scores,
                    torch.tensor(
                        [0.0] * num_pids_to_append,
                        device=scores.device,
                    ),
                ]
            )
            # Create label
            labels = torch.zeros_like(scores)
            labels[pos_indices] = 1

            # Aggregate
            new_pids.append(pids)
            new_scores.append(scores)
            all_labels.append(labels)

        all_pids = new_pids
        all_scores = new_scores

        # Evaluate the retrieved results
        # Evaluate individually so we can remove repeated quries at the end (due to DDP)
        all_stage_1_accs: List = []
        all_stage_3_accs: List = []
        all_stage_2_accs: List = []
        metrics: List[Dict[str, float]] = []
        for b_idx in range(bsize):
            # Evaluate the final results
            eval_preds = EvalPrediction(
                all_scores[b_idx].unsqueeze(0),
                all_labels[b_idx].unsqueeze(0),
                all_pids[b_idx].unsqueeze(0),
            )
            metrics.append(compute_metrics(eval_preds, prefix="test"))
            # Evaluate the intermediate results
            pos_doc_ids = [int(item) for item in batch["pos_doc_ids"][b_idx]]
            all_stage_1_accs.append(
                pid_found_percentage(pos_doc_ids, all_intermediate_pids[b_idx][0])
            )
            all_stage_2_accs.append(
                pid_found_percentage(pos_doc_ids, all_intermediate_pids[b_idx][1])
            )
            all_stage_3_accs.append(
                pid_found_percentage(pos_doc_ids, all_intermediate_pids[b_idx][2])
            )

        # Log metrics
        self.final_eval_results.append(metrics)
        self.intermediate_eval_results.append(
            (all_stage_1_accs, all_stage_2_accs, all_stage_3_accs)
        )

        return None

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        with torch.no_grad():
            if self.index_dir_path is None:
                return self._test_reranking(batch, batch_idx)
            return self._test_full_retrieval(batch, batch_idx)

    def on_test_epoch_end(self) -> Dict:
        # Print the result
        gathered_final_results = self.all_gather(self.final_eval_results)

        # Write results
        copied_gathered_final_results = copy.deepcopy(gathered_final_results)
        collected = []
        for list_of_eval_results in copied_gathered_final_results:
            for eval_result in list_of_eval_results:
                collected.append(eval_result["test_NDCG@10"].tolist())
        # FILE_PATH = "/root/EAGLE/eval_result.json"
        # file_utils.write_json_file(collected, FILE_PATH)
        gathered_intermediate_results = self.all_gather(self.intermediate_eval_results)
        total_data_num = len(self.trainer.datamodule.val_dataset)
        # Aggregate the final results
        gathered_final_metrics = aggregate_final_metrics(
            gathered_final_results,
            total_data_num=total_data_num,
        )
        # Aggregate the intermediate results
        if len(gathered_intermediate_results) == 0:
            gathered_intermediate_metrics = {}
        else:
            gathered_stage1_prob = aggregate_intermediate_metrics(
                [_[0] for _ in gathered_intermediate_results],
                total_data_num=total_data_num,
            )
            gathered_stage2_prob = aggregate_intermediate_metrics(
                [_[1] for _ in gathered_intermediate_results],
                total_data_num=total_data_num,
            )
            gathered_stage3_prob = aggregate_intermediate_metrics(
                [_[2] for _ in gathered_intermediate_results],
                total_data_num=total_data_num,
            )
            gathered_intermediate_metrics = {
                "stage1": gathered_stage1_prob,
                "stage2": gathered_stage2_prob,
                "stage3": gathered_stage3_prob,
            }

        if self.trainer.is_global_zero:
            print("Intermediate results:")
            print(json.dumps(gathered_intermediate_metrics, indent=4))
            print(f"\nFinal results (Total data: {total_data_num}):")
            print(json.dumps(gathered_final_metrics, indent=4))
        # self.trainer._logger_connector._logged_metrics = gathered_metrics
        self.trainer.strategy.barrier()
        return gathered_final_metrics

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        bsize = len(batch["labels"])
        if self.cfg.use_swa and batch_idx > self.cfg.swa_start_batch_idx:
            loss_dic, scores = self.swa_model(**batch)
        else:
            loss_dic, scores = self.model(**batch)

        # Apppend val prefix and detach value
        loss_dic = {
            f"val_{key}": (
                value.detach().item() if isinstance(value, torch.Tensor) else value
            )
            for key, value in loss_dic.items()
        }

        # Compute accuracy
        eval_preds = EvalPrediction(scores, batch["labels"])
        metrics = compute_metrics(eval_preds, prefix="val")

        # Log metrics
        all_dic = loss_dic | metrics
        all_dic = remove_key_with_none_value(all_dic)
        self.log_dict(
            all_dic,
            batch_size=bsize,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return all_dic

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        bsize = batch["q_tok_ids"].size(0)

        # Get the optimizers and learning rate schedulers
        llm_opt, head_opt = self.optimizers()
        if self.cfg.use_swa:
            llm_scheduler, head_scheduler, swa_llm_scheduler, swa_head_scheduler = (
                self.lr_schedulers()
            )
        else:
            llm_scheduler, head_scheduler = self.lr_schedulers()

        # Compute loss
        loss_dic: Dict = self.model(**batch)
        loss = loss_dic["loss"] / self.cfg.gradient_accumulation_steps

        # Backward
        self.manual_backward(loss)

        # Convert tensor to values Log
        loss_dic = unwrap_logging_items(loss_dic)
        self.log_dict(loss_dic, batch_size=bsize)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
            # Clip gradients
            self.clip_gradients(
                llm_opt,
                gradient_clip_val=self.cfg.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
            self.clip_gradients(
                head_opt,
                gradient_clip_val=self.cfg.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )

            # Update weights
            llm_opt.step()
            head_opt.step()
            # Update scheduler
            if self.cfg.use_swa and batch_idx > self.cfg.swa_start_batch_idx:
                self.swa_model.update_parameters(self.model)
                swa_llm_scheduler.step()
                swa_head_scheduler.step()
            else:
                llm_scheduler.step()
                head_scheduler.step()
            # Zero the gradients
            llm_opt.zero_grad()
            head_opt.zero_grad()
        # Return
        return None

    def forward(self, *args, **kwargs) -> Tuple[Dict, List[List[float]]]:
        """For examine the weights of query terms"""
        # Add args
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device != self.device:
                new_args.append(arg.to(self.device))
            else:
                new_args.append(arg)
        new_args = tuple(new_args)

        # Add the kwargs
        new_kargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.device != self.device:
                new_kargs[key] = value.to(self.device)
            else:
                new_kargs[key] = value
        loss_dict, scores = self.model(*new_args, **new_kargs, is_inference=True)
        loss_dict = {
            key: (
                value.squeeze().detach().cpu()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in loss_dict.items()
        }
        scores = scores.detach().cpu()
        return loss_dict, scores

    def configure_optimizers(
        self,
    ) -> Tuple[
        Tuple[torch.optim.Optimizer], Tuple[torch.optim.lr_scheduler.LRScheduler]
    ]:
        # Divide the parameters into two groups: LLM and Head
        llm_params = []
        head_params = []
        for name, params in self.model.named_parameters():
            if params.requires_grad:
                if "llm" in name:
                    llm_params.append(params)
                else:
                    head_params.append(params)

        # Check if the parameters are found
        assert len(llm_params) > 0, "LLM parameters not found!"
        assert len(head_params) > 0, "Head parameters not found!"

        # Get optimizers and schedulers
        optimizers = []
        schedulers = []

        # Create optimizers
        llm_optimizer = self.optim_class(
            params=llm_params, lr=torch.tensor(self.cfg.llm_learning_rate)
        )
        head_optimizer = self.optim_class(
            params=head_params, lr=torch.tensor(self.cfg.head_learning_rate)
        )
        optimizers.extend([llm_optimizer, head_optimizer])

        # Create learning rate schedulers
        alpha = 10000
        num_training_steps = (
            self.train_batch_num // self.cfg.gradient_accumulation_steps + alpha
        )
        llm_lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer=llm_optimizer,
                num_warmup_steps=self.cfg.warmup_steps,
                num_training_steps=num_training_steps,
            ),
            "name": "llm_lr_scheduler",
        }
        head_lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer=head_optimizer,
                num_warmup_steps=self.cfg.warmup_steps,
                num_training_steps=num_training_steps,
            ),
            "name": "head_lr_scheduler",
        }
        schedulers.extend([llm_lr_scheduler, head_lr_scheduler])
        if self.cfg.use_swa:
            swa_llm_lr_scheduler = {
                "scheduler": SWALR(
                    optimizer=llm_optimizer, swa_lr=self.cfg.llm_learning_rate / 2
                ),
                "name": "swa_llm_lr_scheduler",
            }
            swa_head_lr_scheduler = {
                "scheduler": SWALR(
                    optimizer=head_optimizer, swa_lr=self.cfg.head_learning_rate / 2
                ),
                "name": "swa_head_lr_scheduler",
            }
            schedulers.extend([swa_llm_lr_scheduler, swa_head_lr_scheduler])

        # Return optimizers and learning rate schedulers
        return optimizers, schedulers
