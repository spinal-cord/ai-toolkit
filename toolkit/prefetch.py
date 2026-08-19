"""
Rotating, device-resident prefetch stream for training.

This wraps a torch DataLoader and keeps a small ring of *optimizer step*
batches resident on the target device (VRAM).  A background worker thread
pulls batches from the underlying dataloader (which performs all disk I/O
and on-the-fly decryption of raw media, cached latents, prompt embeddings
and optical flow caches), moves the batch tensors to the device, and
rotates them through the ring.

Design:
  * The number of items to prefetch is derived from the batch number:
        items_per_step   = batch_size * gradient_accumulation
        prefetched_items = items_per_step * prefetch_steps
    so with the default ``prefetch_steps=2`` exactly two full steps
    (2 x the batch number of items) are always staged in VRAM ahead of
    the training loop.
  * A ring slot can only be refilled after the training loop has released
    the batch it held (:meth:`PrefetchStream.release`), so peak VRAM usage
    stays at exactly ``prefetch_steps`` steps of data - batches rotate in
    and out of VRAM as training consumes them.
  * Consumers that never call ``release()`` are still safe: while the
    underlying source is not exhausted and the worker is starved for free
    slots, the stream force-rotates the oldest held batch out.
  * The stream is transparent to iteration (``for batch in stream``) and
    re-arms per epoch: when the underlying dataloader is exhausted,
    ``__next__`` raises ``StopIteration`` once the remaining prefetched
    batches have been drained, and the next ``iter()`` call runs the
    per-epoch dataset setup (``trigger_dataloader_setup_epoch``) and
    resumes filling the ring with the new epoch.
"""

import threading

import torch


class _Slot:
    EMPTY = 0
    FILLING = 1
    READY = 2
    IN_USE = 3

    __slots__ = ("state", "batch")

    def __init__(self):
        self.state = _Slot.EMPTY
        self.batch = None


# Batch-level attributes that hold tensors (see DataLoaderBatchDTO)
_BATCH_TENSOR_ATTRS = (
    'tensor',
    'latents',
    'control_tensor',
    'clip_image_tensor',
    'mask_tensor',
    'unaugmented_tensor',
    'unconditional_tensor',
    'unconditional_latents',
    'extra_values',
    'audio_tensor',
    'first_frame_latents',
    'audio_latents',
    'flow',
    'inpaint_tensor',
)


def _tensor_to_device(obj, device):
    """Recursively move any tensors inside ``obj`` to ``device`` (no dtype change)."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device) if obj.device.type != device.type else obj
    if isinstance(obj, (list, tuple)):
        return [_tensor_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _tensor_to_device(v, device) for k, v in obj.items()}
    return obj


def move_batch_to_device(batch, device):
    """Move every tensor held by a ``DataLoaderBatchDTO`` to ``device`` in place."""
    if batch is None:
        return None
    for attr in _BATCH_TENSOR_ATTRS:
        value = getattr(batch, attr, None)
        if value is not None:
            setattr(batch, attr, _tensor_to_device(value, device))

    control_tensor_list = getattr(batch, 'control_tensor_list', None)
    if control_tensor_list is not None:
        batch.control_tensor_list = [
            [_tensor_to_device(t, device) for t in group]
            for group in control_tensor_list
        ]

    # list of {"waveform": tensor, "sample_rate": int}
    audio_data = getattr(batch, 'audio_data', None)
    if audio_data is not None:
        batch.audio_data = [_tensor_to_device(entry, device) for entry in audio_data]

    # list of per-item safetensors state dicts (tensors)
    for attr in ('clip_image_embeds', 'clip_image_embeds_unconditional'):
        value = getattr(batch, attr, None)
        if value is not None:
            setattr(batch, attr, [
                {k: _tensor_to_device(v, device) for k, v in embeds.items()}
                for embeds in value
            ])

    # PromptEmbeds (text/pooled/attention-mask tensors)
    if getattr(batch, 'prompt_embeds', None) is not None:
        batch.prompt_embeds = batch.prompt_embeds.to(device)
    return batch


class PrefetchStream:
    """A rotating, device-resident prefetch buffer that wraps a DataLoader.

    The stream keeps ``prefetch_steps`` full optimizer steps of data staged
    on ``device`` ahead of the consumer.  Batches are moved to the device by
    a background thread as soon as the underlying dataloader produces them
    (disk reads + decryption happen there), and a slot is only refilled
    after its batch has been released, keeping peak device memory flat.
    """

    def __init__(self, dataloader, device, batch_size=1, gradient_accumulation=1, prefetch_steps=2):
        self.dataloader = dataloader
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.gradient_accumulation = max(1, int(gradient_accumulation))
        self.prefetch_steps = max(1, int(prefetch_steps))
        # number of batch slots resident on the device: prefetch_steps full
        # optimizer steps (each step consumes `gradient_accumulation` batches
        # of `batch_size` items)
        self.ring_size = self.gradient_accumulation * self.prefetch_steps
        # number of items staged in VRAM, derived from the batch number
        self.items_per_step = self.batch_size * self.gradient_accumulation
        self.prefetched_items = self.ring_size * self.batch_size

        self._slots = [_Slot() for _ in range(self.ring_size)]
        self._cond = threading.Condition()
        self._fill_pos = 0
        self._take_pos = 0
        self._armed = False
        self._exhausted = False
        self._worker_error = None
        self._worker = None
        self._source = None
        # id(batch) -> _Slot for batches currently held by the consumer
        self._in_use = {}
        self._epoch_count = 0

    # ------------------------------------------------------------------ #
    # iteration
    # ------------------------------------------------------------------ #

    def __iter__(self):
        self._epoch_count += 1
        first_epoch = self._epoch_count == 1
        with self._cond:
            if not first_epoch:
                # per-epoch dataset setup (bucket reshuffle etc.) must run
                # before the worker pulls any batch of the new epoch.
                # imported lazily to avoid a circular import at module load.
                from toolkit.data_loader import trigger_dataloader_setup_epoch
                trigger_dataloader_setup_epoch(self.dataloader)
            self._source = iter(self.dataloader)
            self._exhausted = False
            self._worker_error = None
            self._armed = True
            self._ensure_worker_locked()
            self._cond.notify_all()
        return self

    def _ensure_worker_locked(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="aitk-prefetch-stream",
                daemon=True,
            )
            self._worker.start()

    def __next__(self):
        with self._cond:
            if not self._armed:
                # defensive: auto-arm if iter() was never called
                self._epoch_count += 1
                self._source = iter(self.dataloader)
                self._armed = True
                self._ensure_worker_locked()
        while True:
            with self._cond:
                while True:
                    if self._worker_error is not None:
                        # surface the error (e.g. decryption failure) to the
                        # training loop and keep it raised until re-armed
                        raise self._worker_error
                    slot = self._slots[self._take_pos]
                    if slot.state == _Slot.READY:
                        break
                    if self._exhausted and not self._has_live_slots():
                        # epoch fully drained -> let the loop reset the epoch
                        raise StopIteration
                    if (
                        slot.state != _Slot.READY
                        and not self._exhausted
                        and not self._has_live_slots()
                        and self._slots[self._fill_pos].state != _Slot.EMPTY
                        and self._in_use
                    ):
                        # consumer never released batches and the worker is
                        # starved for free slots (its fill target is not
                        # EMPTY) -> force-rotate the oldest held batch out
                        # (streaming fallback). Note: if the fill target is
                        # EMPTY the worker can already make progress, so we
                        # simply wait for it instead.
                        self._release_oldest_in_use_locked()
                        self._cond.notify_all()
                    self._cond.wait(timeout=1.0)
                slot.state = _Slot.IN_USE
                batch = slot.batch
                self._take_pos = (self._take_pos + 1) % self.ring_size
                self._in_use[id(batch)] = slot
                try:
                    batch._prefetch_stream = self
                except Exception:
                    pass
                return batch

    # ------------------------------------------------------------------ #
    # release / rotation
    # ------------------------------------------------------------------ #

    def release(self, batch):
        """Return a consumed batch to the ring so its slot can be refilled.

        Called by the training loop once a batch has been fully processed.
        Frees the batch's tensors (VRAM) and wakes the worker thread.
        """
        if batch is None:
            return
        with self._cond:
            slot = self._in_use.pop(id(batch), None)
            if slot is None:
                # already rotated out (or not from this stream)
                return
            self._release_slot_locked(slot)
            self._cond.notify_all()

    def _release_slot_locked(self, slot):
        batch = slot.batch
        slot.batch = None
        slot.state = _Slot.EMPTY
        if batch is not None:
            try:
                batch.cleanup()
            except Exception:
                pass

    def _release_oldest_in_use_locked(self):
        batch_id = next(iter(self._in_use))
        slot = self._in_use.pop(batch_id)
        self._release_slot_locked(slot)

    def _has_live_slots(self):
        for slot in self._slots:
            if slot.state in (_Slot.READY, _Slot.FILLING):
                return True
        return False

    # ------------------------------------------------------------------ #
    # worker thread
    # ------------------------------------------------------------------ #

    def _worker_loop(self):
        while True:
            with self._cond:
                while True:
                    if not self._armed:
                        return
                    if self._exhausted:
                        # nothing more to pull this epoch; wait for re-arm.
                        # (must be checked before the EMPTY-slot check, or
                        # the worker would busy-loop on an exhausted source)
                        self._cond.notify_all()
                        self._cond.wait()
                        continue
                    slot = self._slots[self._fill_pos]
                    if slot.state == _Slot.EMPTY and self._worker_error is None:
                        break
                    self._cond.wait()
                slot.state = _Slot.FILLING
            batch = None
            try:
                batch = next(self._source)
                try:
                    move_batch_to_device(batch, self.device)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    # transient VRAM pressure while staging the batch: free
                    # cached memory and retry once before giving up
                    import gc
                    torch.cuda.empty_cache()
                    gc.collect()
                    move_batch_to_device(batch, self.device)
            except StopIteration:
                with self._cond:
                    slot.state = _Slot.EMPTY
                    self._exhausted = True
                    self._cond.notify_all()
                continue
            except Exception as e:
                with self._cond:
                    slot.state = _Slot.EMPTY
                    self._worker_error = e
                    self._cond.notify_all()
                continue
            with self._cond:
                slot.batch = batch
                slot.state = _Slot.READY
                self._fill_pos = (self._fill_pos + 1) % self.ring_size
                self._cond.notify_all()

    # ------------------------------------------------------------------ #
    # misc
    # ------------------------------------------------------------------ #

    def close(self):
        with self._cond:
            self._armed = False
            self._cond.notify_all()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset
