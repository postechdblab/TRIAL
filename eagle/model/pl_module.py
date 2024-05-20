from typing import *

import bitsandbytes as bnb
import hkkang_utils.time as time_utils
import lightning as L
import torch
import tqdm
from omegaconf import DictConfig
from torch.optim.swa_utils import SWALR, AveragedModel
from transformers import EvalPrediction, get_linear_schedule_with_warmup

from eagle.dataset.utils import get_mask
from eagle.metrics import compute_metrics
from eagle.model.late_interaction import NewModel
from eagle.model.utils import append_dummy_pid, unwrap_logging_items
from eagle.search import PLAID
from eagle.tokenizer import NewTokenizer
from eagle.utils import handle_old_ckpt


class LightningNewModel(L.LightningModule):
    def __init__(
        self, cfg: DictConfig, train_batch_num: int = None, index_dir_path: str = None
    ):
        super().__init__()
        ## Need to set this to False to avoid the automatic optimization
        self.automatic_optimization = False
        # Set configurations
        self.cfg = cfg.training
        self.train_batch_num = train_batch_num
        # Tmp
        self.q_tokenizer = NewTokenizer(cfg=cfg.q_tokenizer)
        self.d_tokenizer = NewTokenizer(cfg=cfg.d_tokenizer)
        # Load model
        self.model = NewModel(
            cfg=cfg.model, q_tokenizer=self.q_tokenizer, d_tokenizer=self.d_tokenizer
        )  # Initialize your model with required args
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
        self.prank: Dict[str, int] = {}
        self.timer = time_utils.Timer()
        self.timer_is_started = False

    def _load_searcher(self) -> PLAID:
        # Load the searcher
        assert self.index_dir_path, "Index directory path is not provided!"
        self.searcher = PLAID(index_path=self.index_dir_path)
        return self.searcher

    def _test_reranking(self, batch: Dict, batch_idx: int) -> None:
        _, scores = self.model(**batch, is_analyze=True)

        # Compute accuracy
        eval_preds = EvalPrediction(scores, batch["labels"])
        metrics = compute_metrics(eval_preds, prefix="test")

        # Log metrics
        bsize = batch["q_tok_ids"].size(0)
        self.log_dict(
            metrics, batch_size=bsize, on_step=False, on_epoch=True
        )
        is_analyze = True
        if is_analyze:
            assert (
                len(batch["q_id"]) == 1
            ), f"Only one query is supported for analysis, but found {len(batch['q_id'])}"
            is_correct = metrics["test_custom@10"] == 1.0
            query = self.q_tokenizer.tokenizer.decode(batch["q_tok_ids"][0])
            q_toks = self.q_tokenizer.tokenizer.convert_ids_to_tokens(
                batch["q_tok_ids"][0]
            )
            # Find the rank of the positive document (i.e., the first document in the list)
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            sorted_scores.squeeze()
            sorted_indices = sorted_indices.squeeze()
            pos_rank = (sorted_indices == 0).nonzero(as_tuple=False)[0][0].item()
            # Show the text for the top-10 documents
            doc_ids_list = batch["doc_tok_ids"][0]
            score_q_max = scores.max().item()
            # Compute relevance scores
            max_q_rel_scores, max_q_d_indices = _["tok_intra_qd_scores"].max(dim=1)
            q_weights = (
                None if _["q_tok_weight"] is None else _["q_tok_weight"].squeeze()
            )
            print(f"\n\nis_correct:{is_correct}")
            print(f"Query {batch_idx}: {query}")

            def show(d_idx, rank=0):
                doc = self.d_tokenizer.tokenizer.decode(
                    doc_ids_list[d_idx], skip_special_tokens=True
                )
                doc_toks = self.d_tokenizer.tokenizer.convert_ids_to_tokens(
                    doc_ids_list[d_idx]
                )
                max_q_d_idx_list = max_q_d_indices[d_idx].tolist()
                print(
                    f"\nDoc (rank:{rank} score:{scores[0][d_idx]} idx:{d_idx} ): {doc}"
                )
                # Print the query scores
                print(
                    [
                        (
                            f"{q_toks[i]}-{doc_toks[max_q_d_idx_list[i]]}",
                            s,
                            None if q_weights is None else q_weights[i].tolist(),
                        )
                        for i, s in enumerate(max_q_rel_scores[d_idx].tolist())
                    ]
                )

            # for rank, d_idx in enumerate(sorted_indices[:10]):
            #     show(d_idx, rank=rank)
        self.prank[batch["q_id"][0]] = pos_rank
        return None

    def _test_full_retrieval(self, batch: Dict, batch_idx: int) -> None:
        bsize = batch["q_tok_ids"].size(0)

        # Load the searcher if not loaded
        if self.searcher is None:
            self._load_searcher()

        # Extract query ids, mask, and scatter indices
        q_tok_ids = batch["q_tok_ids"]
        q_tok_att_mask = batch["q_tok_att_mask"]
        q_tok_mask = batch["q_tok_mask"]
        q_scatter_indices = batch["q_scatter_indices"]

        # Encode the query
        q_encoded, q_projected, q_weight, q_scale_factor = self.model.encode_q_text(
            tok_ids=q_tok_ids,
            att_mask=q_tok_att_mask,
            tok_mask=q_tok_mask,
            scatter_indices=q_scatter_indices,
        )
        q_projected = q_projected.half()
        if q_weight is not None:
            q_weight = q_weight.half()

        # Search the corpus with indexed document corpus
        all_pids: List = []
        all_scores: List = []
        all_labels: List = []
        for bidx in range(bsize):
            # Retrieve pids and scores
            query = q_projected[bidx]
            mask = q_tok_mask[bidx].squeeze()
            weight = None if q_weight is None else q_weight[bidx]
            pids, scores = self.searcher(query=query, mask=mask, weight=weight)
            # Find the positive doc id
            pos_doc_idxs = batch["pos_doc_idxs"][bidx]
            # Append the positive doc id if not found
            pids, pos_indices = append_dummy_pid(pids=pids, target_pids=pos_doc_idxs)
            scores = torch.cat(
                [scores, torch.tensor([0.0] * len(pos_doc_idxs), device=scores.device)]
            )
            # Create labels
            labels = torch.zeros_like(scores)
            labels[pos_indices] = 1
            # Append to the list
            all_pids.append(pids)
            all_scores.append(scores)
            all_labels.append(labels)
        # Stack the results
        all_pids = torch.stack(all_pids)
        all_scores = torch.stack(all_scores)
        all_labels = torch.stack(all_labels)

        # Evaluate the results
        eval_preds = EvalPrediction(all_scores, all_labels, all_pids)
        metrics = compute_metrics(eval_preds, prefix="test")

        # Log metrics
        self.log_dict(
            metrics, batch_size=bsize, on_step=False, on_epoch=True
        )

        return None

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        if self.index_dir_path is None:
            return self._test_reranking(batch, batch_idx)
        return self._test_full_retrieval(batch, batch_idx)

    def on_test_epoch_end(self) -> Dict:
        metrics: Dict[str, torch.Tensor] = self.trainer.logged_metrics
        metrics = {
            key: value.cpu().item() if type(value) == torch.Tensor else value
            for key, value in metrics.items()
            if "test" in key
        }
        # synchronize self.prank
        self.all_gather(self.prank)

        # Print the result
        if self.trainer.is_global_zero:
            pass
        # Write the results to a file
        import hkkang_utils.file as file_utils

        print("Writing the results to a file")

        file_utils.write_json_file(self.prank, f"prank_{self.local_rank}.json")

        return metrics

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        bsize = batch["q_tok_ids"].size(0)
        if self.cfg.is_use_swa and batch_idx > self.cfg.swa_start_batch_idx:
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
        self.log_dict(
            all_dic, batch_size=bsize, on_step=False, on_epoch=True, sync_dist=True,
        )

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        bsize = batch["q_tok_ids"].size(0)

        # Get the optimizers and learning rate schedulers
        llm_opt, head_opt = self.optimizers()
        if self.cfg.is_use_swa:
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
        loss_dic = unwrap_logging_items(loss_dic, target_key="loss")
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
            if self.cfg.is_use_swa and batch_idx > self.cfg.swa_start_batch_idx:
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
        if self.cfg.is_use_swa:
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

    # Custom methods for indexing corpus
    def docFromText(
        self,
        docs: List[str],
        bsize: Optional[int] = None,
        keep_dims="flatten",
        showprogress=False,
    ):
        assert keep_dims == "flatten", "Only 'flatten' is supported for keep_dims."
        assert bsize, "Please provide the batch size for the indexing."

        # Tensorize the documents
        from colbert.modeling.tokenization.utils import (_sort_by_length,
                                                         _split_into_batches)

        # Tokenize
        result = self.d_tokenizer(docs, padding=True, return_tensors="pt")
        ids, att_mask = result["input_ids"], result["attention_mask"]
        ids, att_mask, reverse_indices = _sort_by_length(ids, att_mask, bsize)
        # Create mask
        skip_ids = self.d_tokenizer.special_toks_ids + self.d_tokenizer.punctuations
        tok_mask = get_mask(input_ids=ids, skip_ids=skip_ids).unsqueeze(-1)

        # Create batch
        text_batches = _split_into_batches(ids, att_mask, tok_mask, bsize=bsize)

        # Encode
        result_batches = [
            self.model.encode_d_text(
                tok_ids=input_ids.to(self.device),
                att_mask=attention_mask.to(self.device),
                tok_mask=token_mask.to(self.device),
            )
            for input_ids, attention_mask, token_mask in tqdm.tqdm(
                text_batches, disable=not showprogress
            )
        ]

        # Flatten
        D, mask = [], []
        for i in range(len(result_batches)):
            D_ = result_batches[i][0].half()
            mask_ = text_batches[i][2].bool()
            D.append(D_)
            mask.append(mask_)

        D, mask = torch.cat(D)[reverse_indices], torch.cat(mask)[reverse_indices]
        doclens = mask.squeeze(-1).sum(-1).tolist()

        # Serialize and remove the masked tokens
        D = D.view(-1, D.shape[-1])
        D = D[mask.bool().flatten()].cpu()

        # Check if flatten is correct
        assert len(D) == sum(doclens), f"len(D)={len(D)} != sum(doclens)={sum(doclens)}"

        return D, doclens
