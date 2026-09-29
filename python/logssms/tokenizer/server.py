from __future__ import annotations

import multiprocessing as mp

import torch

from logssms.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from logssms.message.queue import ZmqPullQueue, ZmqPushQueue
from logssms.utils import init_logger, load_tokenizer

from .tokenizer import TokenizeManager, DetokenizeManager


def _unwrap_msg(msg: BaseTokenizerMsg) -> list[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    tokenizer_addr: str,
    backend_addr  : str,
    local_bs      : int,
    tokenizer_id  : int                  = -1,
    ack_queue     : mp.Queue[str] | None = None,
) -> None:
    assert local_bs > 0

    backend_sender = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    receiver = ZmqPullQueue(tokenizer_addr, create=False, decoder=BatchTokenizerMsg.decoder)
    tokenizer = load_tokenizer(tokenizer_path)
    tokenize_manager = TokenizeManager(tokenizer)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = []
            while len(pending_msg) < local_bs and not receiver.empty():
                pending_msg.extend(_unwrap_msg(receiver.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]

            if len(tokenize_msg) > 0:
                tensors = tokenize_manager.tokenize(tokenize_msg)
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                            profile_tag=msg.profile_tag,
                            return_token_ids=msg.return_token_ids,
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                backend_sender.put(batch_output)
                
            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                backend_sender.put(batch_output)

    except KeyboardInterrupt:
        pass


@torch.inference_mode()
def detokenize_worker(
    *,
    tokenizer_path  : str,
    detokenizer_addr: str,
    backend_addr    : str,
    frontend_addr   : str,
    local_bs        : int,
    detokenizer_id  : int                  = -1,
    ack_queue       : mp.Queue[str] | None = None,
) -> None:
    assert local_bs > 0
    
    backend_sender = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    frontend_sender = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    receiver = ZmqPullQueue(detokenizer_addr, create=False, decoder=BatchTokenizerMsg.decoder)
    tokenizer = load_tokenizer(tokenizer_path)
    detokenize_manager = DetokenizeManager(tokenizer)
    logger = init_logger(__name__, f"detokenizer_{detokenizer_id}")

    if ack_queue is not None:
        ack_queue.put(f"Detokenize server {detokenizer_id} is ready")

    try:
        while True:
            pending_msg = []
            while len(pending_msg) < local_bs and not receiver.empty():
                pending_msg.extend(_unwrap_msg(receiver.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]

            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                            token_ids=(
                                [msg.next_token] if isinstance(msg.next_token, int)
                                else list(msg.next_token)
                            ) if msg.return_token_ids else None,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                frontend_sender.put(batch_output)

            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                backend_sender.put(batch_output)

    except KeyboardInterrupt:
        pass
