import math
import random
from einops import rearrange
import torch
from framework.modules.post_processor import Processor
from framework.utils.compute_metrics import compute_metrics
from framework.utils.util import from_pretrained_checkpoint
from utils.util import AverageMeter, get_lr
from omegaconf import DictConfig
from tqdm import tqdm
from hydra.utils import instantiate, get_class
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger(__name__)

def _print_clip_info(name, clips):
    if len(clips) == 0:
        print(f"{name}: empty")
        return
    c = clips[0]
    if torch.is_tensor(c):
        print(f"{name}: len={len(clips)}, "
              f"shape={tuple(c.shape)}, "
              f"dtype={c.dtype}, "
              f"device={c.device}")
    else:
        print(f"{name}: len={len(clips)}, type={type(c)}")


class Trainer:
    def __init__(self,
                 resumed_training: bool = False,
                 generic: DictConfig = None,
                 renderer: DictConfig = None,
                 model: DictConfig = None,
                 criterion: DictConfig = None,
                 **kwargs):
        # # current working directory: outputs/${trainer.task_name}/${data.data_name}/${run_id}
        # folder: save/${trainer.task_name}/${data.data_name}  # ckpt_name: checkpoint.pth
        # # last ckpt directory
        # ckpt_dir: ${get_last_checkpoint:${trainer.folder}}  # ${trainer.run_id}
        # # for example, ckpt_dir: save/motion_diffusion/react_2024/checkpoints
        # resume_run_id: ${old_run_id}

        super().__init__()
        self.resumed_training = resumed_training
        self.renderer = renderer
        self.model_cfg = model
        self.criterion_cfg = criterion

        if torch.cuda.device_count() > 0:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.device = device
        self.kwargs = kwargs
        self.trainer_cfg = generic
        self.optim_cfg = kwargs.pop("optim")
        self.task = kwargs.get("task")

    def set_data_module(self, data_module):
        self.data_module = data_module

    def data_resample(self,
                      speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips,
                      listener_video_clips, listener_emotion_clips, listener_3dmm_clips,
                      speaker_seq_lengths, listener_seq_lengths,):

        s_ratio = self.trainer_cfg.s_ratio
        window_size = self.trainer_cfg.window_size
        clip_length = self.trainer_cfg.clip_length
        s_window_size = s_ratio * window_size
        l_window_size = window_size

        # print("===== data_resample params =====")
        # print(f"task: {self.task}")
        # print(f"s_ratio={s_ratio}, window_size={window_size}, clip_length={clip_length}")
        # print(f"s_window_size={s_window_size}, l_window_size={l_window_size}")

        # _print_clip_info("speaker_audio_clips", speaker_audio_clips)
        # _print_clip_info("speaker_emotion_clips", speaker_emotion_clips)
        # _print_clip_info("speaker_3dmm_clips", speaker_3dmm_clips)
        # _print_clip_info("listener_video_clips", listener_video_clips)
        # _print_clip_info("listener_emotion_clips", listener_emotion_clips)
        # _print_clip_info("listener_3dmm_clips", listener_3dmm_clips)

        # print(f"speaker_seq_lengths: {speaker_seq_lengths}")
        # print(f"listener_seq_lengths: {listener_seq_lengths}")

        if self.task == 'offline':
            stack = lambda clips: torch.stack(clips, dim=0)
            speaker_audio, speaker_emotion, speaker_3dmm = (
                stack(clips) for clips in (speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips))
            listener_video, listener_emotion, listener_3dmm = (
                stack(clips) for clips in (listener_video_clips, listener_emotion_clips, listener_3dmm_clips))
            past_listener_emotion = past_listener_3dmm = None
            seq_lengths = torch.tensor(speaker_seq_lengths).clamp(max=clip_length)
            # Tensor([58, 750, 632, ...])

        elif self.task == "online":
            def get_padded(clip: torch.Tensor, length: int, target_len: int) -> torch.Tensor:
                clip = clip[:length]
                if length < target_len:
                    pad_shape = (target_len - length, *clip.shape[1:])
                    clip = torch.cat([clip, clip.new_zeros(pad_shape)], dim=0)
                return clip

            speaker_audio, speaker_emotion, speaker_3dmm = [], [], []
            listener_video, listener_emotion, listener_3dmm = [], [], []
            past_listener_emotion, past_listener_3dmm = [], []
            speaker_future_emotion = []

            for (speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip, speaker_seq_length,
                 listener_video_clip, listener_emotion_clip, listener_3dmm_clip, listener_seq_length) in \
                    zip(speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips, speaker_seq_lengths,
                        listener_video_clips, listener_emotion_clips, listener_3dmm_clips, listener_seq_lengths):
                seq_length = speaker_seq_length
                assert speaker_seq_length == listener_seq_length, "Sequence length not equal"

                speaker_audio_clip = get_padded(speaker_audio_clip, seq_length, s_window_size)
                speaker_emotion_clip = get_padded(speaker_emotion_clip, seq_length, s_window_size)
                speaker_3dmm_clip = get_padded(speaker_3dmm_clip, seq_length, s_window_size)
                listener_video_clip = get_padded(listener_video_clip, seq_length, s_window_size)
                listener_emotion_clip = get_padded(listener_emotion_clip, seq_length, s_window_size)
                listener_3dmm_clip = get_padded(listener_3dmm_clip, seq_length, s_window_size)

                # Extract future 10 frames
                future_len = 10

                # Strategy: Try to sample such that we have enough future frames
                if seq_length < clip_length:
                    max_idx = seq_length - s_window_size - future_len
                    if max_idx >= 0:
                        cp = random.randint(0, max_idx)
                    else:
                        # Fallback to original sampling if sequence is too short
                        cp = random.randint(0, seq_length - s_window_size) if seq_length > s_window_size else 0
                else:
                    # For long clips, we can safely reserve space for future frames
                    cp = random.randint(0, clip_length - s_window_size - future_len)

                du = cp + s_window_size
                
                # Get future emotion with Replication Padding fallback (just in case)
                if du + future_len <= speaker_emotion_clip.shape[0]:
                    future_emotion = speaker_emotion_clip[du: du + future_len]
                else:
                    # If we still run out of bounds (e.g. very short sequence), use Replication Padding
                    available_future = speaker_emotion_clip[du:]
                    num_available = available_future.shape[0]
                    if num_available > 0:
                        last_frame = available_future[-1:]
                        padding = last_frame.repeat(future_len - num_available, 1)
                        future_emotion = torch.cat([available_future, padding], dim=0)
                    else:
                        # If du is already at the end, repeat the last frame of the current window
                        last_frame = speaker_emotion_clip[du-1:du]
                        future_emotion = last_frame.repeat(future_len, 1)
                
                speaker_audio_clip = speaker_audio_clip[cp:du]
                speaker_emotion_clip = speaker_emotion_clip[cp:du]
                speaker_3dmm_clip = speaker_3dmm_clip[cp:du]
                listener_video_clip = listener_video_clip[du - l_window_size:du]
                past_listener_emotion_clip = listener_emotion_clip[(du - 2 * l_window_size): (du - l_window_size)]
                listener_emotion_clip = listener_emotion_clip[(du - l_window_size): du]
                past_listener_3dmm_clip = listener_3dmm_clip[(du - 2 * l_window_size): (du - l_window_size)]
                listener_3dmm_clip = listener_3dmm_clip[(du - l_window_size): du]

                speaker_audio.append(speaker_audio_clip)
                speaker_emotion.append(speaker_emotion_clip)
                speaker_3dmm.append(speaker_3dmm_clip)
                listener_video.append(listener_video_clip)
                listener_emotion.append(listener_emotion_clip)
                listener_3dmm.append(listener_3dmm_clip)
                past_listener_emotion.append(past_listener_emotion_clip)
                past_listener_3dmm.append(past_listener_3dmm_clip)
                speaker_future_emotion.append(future_emotion)

            speaker_audio = torch.stack(speaker_audio, dim=0)  # (bs, s_w, d)
            speaker_emotion = torch.stack(speaker_emotion, dim=0)  # (bs, s_w, 25)
            speaker_3dmm = torch.stack(speaker_3dmm, dim=0)  # (bs, s_w, 58)
            listener_video = torch.stack(listener_video, dim=0)  # (bs, l_w, 3, 224, 224)
            listener_emotion = torch.stack(listener_emotion, dim=0)  # (bs, l_w, 25)
            listener_3dmm = torch.stack(listener_3dmm, dim=0)  # (bs, l_w, 58)
            past_listener_emotion = torch.stack(past_listener_emotion, dim=0)  # (bs, l_w, 25)
            past_listener_3dmm = torch.stack(past_listener_3dmm, dim=0)  # (bs, l_w, 58)
            speaker_future_emotion = torch.stack(speaker_future_emotion, dim=0) # (bs, 10, 25)
            seq_lengths = None
        else:
            raise ValueError("Unknown task type")

        return (speaker_audio, speaker_emotion, speaker_3dmm, listener_video, listener_emotion,
                listener_3dmm, past_listener_emotion, past_listener_3dmm, seq_lengths, speaker_future_emotion)

    def fit(self):
        """
        # relative directory
        root_dir = save/${trainer.task_name}/${data.data_name}/${folder_name}
        # absolute directory
        saving_dir = Path(hydra.utils.to_absolute_path(root_dir))
        # get saving path
        saving_path = str(saving_dir / ...)
        """

        self.start_epoch = self.trainer_cfg.start_epoch
        self.epochs = self.trainer_cfg.epochs
        self.tb_dir = self.trainer_cfg.tb_dir
        self.clip_grad = self.trainer_cfg.clip_grad
        self.val_period = self.trainer_cfg.val_period
        stage = "fit"

        logger.info("Loading data module")
        self.train_loader, self.val_loader = self.data_module.get_dataloader(stage=stage)
        logger.info("Data module loaded")

        logger.info("Loading criterion")
        self.criterion = instantiate(self.criterion_cfg)
        logger.info("Criterion loaded")

        logger.info("Loading writer")
        self.writer = SummaryWriter(self.tb_dir)
        logger.info(f"Writer loaded: {self.tb_dir}")
        self.main_diffusion(stage)

    def main_diffusion(self, stage):
        model = instantiate(self.model_cfg.diff_model,
                            stage=stage,
                            resumed_training=self.resumed_training,
                            latent_embedder=self.model_cfg.latent_embedder \
                                if hasattr(self.model_cfg, "latent_embedder") else None,
                            audio_encoder=self.model_cfg.audio_encoder \
                                if hasattr(self.model_cfg, "audio_encoder") else None,
                            **self.kwargs,
                            _recursive_=False)
        model.to(self.device)

        # load optimizer
        # Layer-wise Learning Rate: GRU=5e-5, Diffusion=1e-4
        predictor_params = []
        diffusion_params = []
        for name, param in model.named_parameters():
            if "speaker_emotion_predictor" in name:
                predictor_params.append(param)
            else:
                diffusion_params.append(param)
        
        # Use get_class to respect the config's _target_ (e.g. AdamW) while avoiding instantiate()'s 
        # type conversion issues with parameter groups containing Tensors.
        OptimizerClass = get_class(self.optim_cfg._target_)
        optim_args = dict(self.optim_cfg)
        if "_target_" in optim_args:
            del optim_args["_target_"]
            
        optimizer = OptimizerClass([
            {"params": predictor_params, "lr": 5e-5},
            {"params": diffusion_params, "lr": self.trainer_cfg.lr}
        ], lr=self.trainer_cfg.lr, **optim_args)

        if self.resumed_training:
            checkpoint_path = model.get_ckpt_path(model.diffusion_decoder.model, runid="resume_runid", last=True)
            best_diff_decoder_loss, self.start_epoch = (
                from_pretrained_checkpoint(checkpoint_path, optimizer, self.device)
            )
            logger.info(f"Resume training from epoch {self.start_epoch}")
        else:
            best_diff_decoder_loss = float('inf')
        print(f"Best validation loss: {best_diff_decoder_loss}")

        # load scheduler
        scheduler = instantiate(self.kwargs.pop("scheduler"), optimizer, len(self.train_loader))

        # Warmup strategy: Freeze diffusion model initially to train GRU predictor
        warmup_epochs = 10
        logger.info(f"Starting warmup for {warmup_epochs} epochs: Freezing diffusion, training GRU predictor only.")
        
        # Freeze diffusion parts
        for name, param in model.named_parameters():
            if "speaker_emotion_predictor" not in name:
                param.requires_grad = False
        
        for epoch in range(self.start_epoch, self.start_epoch + warmup_epochs):
             # Only train predictor
            diff_decoder_loss, au_rec_loss, va_rec_loss, em_rec_loss, pred_loss= (
                self.train_diffusion(model, self.train_loader, optimizer, None,
                                     self.criterion, epoch, self.writer, self.device))
            logging.info(f"Warmup Epoch: {epoch + 1}  pred_loss: {pred_loss:.5f}")

            if (epoch + 1) % self.val_period == 0:
                diff_decoder_loss, au_rec_loss, va_rec_loss, em_rec_loss, pred_loss= (
                    self.val_diffusion(model, self.val_loader, self.criterion, self.device))
                logging.info(f"Warmup Validation Epoch: {epoch + 1}  val_pred_loss: {pred_loss:.5f}")
            
        # Unfreeze all
        logger.info("Warmup finished. Unfreezing all parameters for joint training.")
        for param in model.parameters():
            param.requires_grad = True

        for epoch in range(self.start_epoch + warmup_epochs, self.epochs):
            diff_decoder_loss, au_rec_loss, va_rec_loss, em_rec_loss, pred_loss = (
                self.train_diffusion(model, self.train_loader, optimizer, scheduler,
                                     self.criterion, epoch, self.writer, self.device))
            logging.info(f"Epoch: {epoch + 1}  train_diff_loss: {diff_decoder_loss:.5f}  au_rec_loss: {au_rec_loss:.5f}"
                         f"  va_rec_loss: {va_rec_loss:.5f}  em_rec_loss: {em_rec_loss:.5f}  pred_loss: {pred_loss:.5f}"
                         )

            if (epoch + 1) % self.val_period == 0:
                diff_decoder_loss, au_rec_loss, va_rec_loss, em_rec_loss, pred_loss = (
                    self.val_diffusion(model, self.val_loader, self.criterion, self.device))
                logging.info(f"Epoch: {epoch + 1}  val_diff_loss: {diff_decoder_loss:.5f}  au_rec_loss: {au_rec_loss:.5f}"
                             f"  va_rec_loss: {va_rec_loss:.5f}  em_rec_loss: {em_rec_loss:.5f}  pred_loss: {pred_loss:.5f}"
                             )

                if diff_decoder_loss < best_diff_decoder_loss:
                    best_diff_decoder_loss = diff_decoder_loss
                    logging.info(
                        f"New best diff_decoder_loss ({best_diff_decoder_loss:.5f}) at epoch {epoch + 1}, "
                        f"saving checkpoint"
                    )
                    model.save_ckpt(optimizer, best=True, epoch=(epoch+1), best_loss=best_diff_decoder_loss)

                model.save_ckpt(optimizer, epoch=(epoch + 1), best_loss=best_diff_decoder_loss)
                model.save_ckpt(optimizer, last=True, epoch=(epoch+1), best_loss=best_diff_decoder_loss)

    def train_diffusion(self, model, data_loader, optimizer, scheduler,
                        criterion, epoch, writer, device):
        whole_losses = AverageMeter()
        au_rec_losses = AverageMeter()
        va_rec_losses = AverageMeter()
        em_rec_losses = AverageMeter()
        pred_losses = AverageMeter()


        model.train()
        for batch_idx, (
                speaker_audio_clip,
                speaker_video_clip,
                speaker_emotion_clip,
                speaker_3dmm_clip,
                listener_video_clip,
                listener_emotion_clip,
                listener_3dmm_clip,
                speaker_clip_length,
                listener_clip_length,
        ) in enumerate(tqdm(data_loader)):

            (speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip,
             listener_video_clip, listener_emotion_clip, listener_3dmm_clip,
             past_listener_emotion, past_listener_3dmm, motion_lengths, speaker_future_emotion) = self.data_resample(
                    speaker_audio_clips=speaker_audio_clip, speaker_emotion_clips=speaker_emotion_clip,
                    speaker_3dmm_clips=speaker_3dmm_clip, listener_video_clips=listener_video_clip,
                    listener_emotion_clips=listener_emotion_clip, listener_3dmm_clips=listener_3dmm_clip,
                    speaker_seq_lengths=speaker_clip_length, listener_seq_lengths=listener_clip_length,)

            (speaker_audio_clip,  # (78-d)
             speaker_emotion_clip,  # (25-d)
             speaker_3dmm_clip,  # (58-d)
             listener_video_clip,
             listener_emotion_clip,  # (25-d)
             speaker_future_emotion
             ) = (speaker_audio_clip.to(device),
                 speaker_emotion_clip.to(device),
                 speaker_3dmm_clip.to(device),
                 listener_video_clip.to(device),
                 listener_emotion_clip.to(device),
                 speaker_future_emotion.to(device))

            batch_size = speaker_audio_clip.shape[0]

            outputs = model(
                speaker_audio_input=speaker_audio_clip,
                speaker_emotion_input=speaker_emotion_clip,
                speaker_3dmm_input=speaker_3dmm_clip,
                listener_emotion_input=listener_emotion_clip,
                past_listener_emotion=past_listener_emotion,
                motion_length=motion_lengths,
                speaker_future_emotion_input=speaker_future_emotion,
            )
            # outputs['prediction_emotion'].shape: [bs, k, l_w, 25]
            # outputs['target_emotion'].shape: [bs, k, l_w, 25]
            
            # # 打印outputs的类型和所有key的shape
            # print("outputs type:", type(outputs))
            # for k, v in outputs.items():
            #     if hasattr(v, "shape"):
            #         print(f"outputs['{k}'] shape: {v.shape}")
            #     else:
            #         print(f"outputs['{k}'] type: {type(v)}")

            output = criterion(outputs)
            loss = output["loss"]

            iteration = batch_idx + len(data_loader) * epoch
            if writer is not None:
                writer.add_scalar("Train/loss", loss.data.item(), iteration)
                # writer.add_scalar("Train/temporal_loss", temporal_loss.data.item(), iteration)

            whole_losses.update(loss.data.item(), batch_size)
            au_rec_losses.update(output["loss_au"].data.item(), batch_size)
            va_rec_losses.update(output["loss_va"].data.item(), batch_size)
            em_rec_losses.update(output["loss_em"].data.item(), batch_size)

            if "loss_emotion_pred" in output:
                pred_losses.update(output["loss_emotion_pred"].data.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None and (epoch + 1) >= 15:
            scheduler.step()
        lr = get_lr(optimizer=optimizer)
        if writer is not None:
            writer.add_scalar("Train/lr", lr, epoch)

        return whole_losses.avg, au_rec_losses.avg, va_rec_losses.avg, em_rec_losses.avg, pred_losses.avg

    def val_diffusion(self, model, val_loader, criterion, device):
        whole_losses = AverageMeter()
        au_rec_losses = AverageMeter()
        va_rec_losses = AverageMeter()
        em_rec_losses = AverageMeter()
        pred_losses = AverageMeter()
     

        model.eval()
        for batch_idx, (
                speaker_audio_clip,
                speaker_video_clip,
                speaker_emotion_clip,
                speaker_3dmm_clip,
                listener_video_clip,
                listener_emotion_clip,
                listener_3dmm_clip,
                speaker_clip_length,
                listener_clip_length,
        ) in enumerate(tqdm(val_loader)):

            (speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip,
             listener_video_clip, listener_emotion_clip, listener_3dmm_clip,
             past_listener_emotion, past_listener_3dmm, motion_lengths, speaker_future_emotion) = self.data_resample(
                    speaker_audio_clips=speaker_audio_clip, speaker_emotion_clips=speaker_emotion_clip,
                    speaker_3dmm_clips=speaker_3dmm_clip, listener_video_clips=listener_video_clip,
                    listener_emotion_clips=listener_emotion_clip, listener_3dmm_clips=listener_3dmm_clip,
                    speaker_seq_lengths=speaker_clip_length, listener_seq_lengths=listener_clip_length)

            (speaker_audio_clip,  # (78-d)
             speaker_emotion_clip,  # (25-d)
             speaker_3dmm_clip,  # (58-d)
             listener_video_clip,
             listener_emotion_clip,  # (25-d)
             speaker_future_emotion
             ) = (speaker_audio_clip.to(device),
                 speaker_emotion_clip.to(device),
                 speaker_3dmm_clip.to(device),
                 listener_video_clip.to(device),
                 listener_emotion_clip.to(device),
                 speaker_future_emotion.to(device))

            batch_size = speaker_audio_clip.shape[0]

            with torch.no_grad():
                outputs = model(
                    speaker_audio_input=speaker_audio_clip,
                    speaker_emotion_input=speaker_emotion_clip,
                    speaker_3dmm_input=speaker_3dmm_clip,
                    listener_emotion_input=listener_emotion_clip,
                    past_listener_emotion=past_listener_emotion,
                    motion_length=motion_lengths,
                    speaker_future_emotion_input=speaker_future_emotion,
                )

                output = criterion(outputs)
                loss = output["loss"]

            whole_losses.update(loss.data.item(), batch_size)
            au_rec_losses.update(output["loss_au"].data.item(), batch_size)
            va_rec_losses.update(output["loss_va"].data.item(), batch_size)
            em_rec_losses.update(output["loss_em"].data.item(), batch_size)
            if "loss_emotion_pred" in output:
                pred_losses.update(output["loss_emotion_pred"].data.item(), batch_size)
         

        return whole_losses.avg, au_rec_losses.avg, va_rec_losses.avg, em_rec_losses.avg, pred_losses.avg
    
    def test(self):
        stage = "test"
        data_clamp = self.kwargs.pop("data_clamp")

        logger.info("Loading test data module")
        test_loader = self.data_module.get_dataloader(stage=stage)
        logger.info("Test data module loaded")

        clip_len = self.trainer_cfg.clip_length
        w = self.trainer_cfg.window_size
        s_ratio = self.trainer_cfg.s_ratio
        s_w = s_ratio * w

        model = instantiate(
            self.model_cfg.diff_model,
            stage=stage,
            latent_embedder=self.model_cfg.latent_embedder
            if hasattr(self.model_cfg, "latent_embedder") else None,
            audio_encoder=self.model_cfg.audio_encoder
            if hasattr(self.model_cfg, "audio_encoder") else None,
            **self.kwargs,
            _recursive_=False
        )
        model.to(self.device)
        model.eval()

        logger.info("Loading post processor")
        post_processor = Processor(
            config_name=self.kwargs.pop("post_config_name"),
            clip_len_test=self.kwargs.pop("post_clip_length"),
            device=self.device,
        )
        logger.info("Post processor loaded")

        GT_listener_emotions_all = []
        pred_listener_emotions_all = []
        input_speaker_emotions_all = []
        all_meta = []

        for batch_idx, (
                speaker_audio_clips,
                speaker_video_clips,
                speaker_emotion_clips,
                speaker_3dmm_clips,
                listener_video_clips,
                listener_emotion_clips,
                _,
                speaker_seq_lengths,
                listener_seq_lengths,
                metas,
        ) in enumerate(tqdm(test_loader)):

            all_meta.extend(metas)

            # listener_emotion_clips:
            # List[ batch ] -> each item is a list of 10 tensors with shape [T, 25]
            GT_listener_emotions_all.extend(listener_emotion_clips)
            input_speaker_emotions_all.extend(speaker_emotion_clips)

            clip_batch_size = 8  # in case too long data sequence
            speaker_audios = []
            speaker_emotions = []
            speaker_3dmms = []
            motion_lengths = []
            sample_batch_size = []

            for speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip, speaker_seq_length in zip(
                speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips, speaker_seq_lengths
            ):
                length = speaker_seq_length

                if length == 751:
                    length = 750

                speaker_3dmm_clip = speaker_3dmm_clip[:length]

                if self.task == "offline":
                    remain_length = length % clip_len
                    b = math.ceil((length + clip_len - remain_length) / clip_len)
                    lengths = torch.tensor([clip_len] * (b - 1) + [remain_length])
                    sample_batch_size.append(b)

                    speaker_audio_clip = torch.cat(
                        (
                            speaker_audio_clip,
                            torch.zeros(size=(clip_len - remain_length, speaker_audio_clip.shape[-1]))
                        ),
                        dim=0
                    )
                    speaker_audio_clip = rearrange(speaker_audio_clip, '(b l) d -> b l d', b=b)

                    speaker_emotion_clip = torch.cat(
                        (
                            speaker_emotion_clip,
                            torch.zeros(size=(clip_len - remain_length, speaker_emotion_clip.shape[-1]))
                        ),
                        dim=0
                    )
                    speaker_emotion_clip = rearrange(speaker_emotion_clip, '(b l) d -> b l d', b=b)

                    speaker_3dmm_clip = torch.cat(
                        (
                            speaker_3dmm_clip,
                            torch.zeros(size=(clip_len - remain_length, speaker_3dmm_clip.shape[-1]))
                        ),
                        dim=0
                    )
                    speaker_3dmm_clip = rearrange(speaker_3dmm_clip, '(b l) d -> b l d', b=b)

                    speaker_audios.append(speaker_audio_clip)
                    speaker_emotions.append(speaker_emotion_clip)
                    speaker_3dmms.append(speaker_3dmm_clip)
                    motion_lengths.append(lengths)

                else:  # online task
                    num_windows = math.ceil(length / w)
                    sample_batch_size.append(num_windows)

                    speaker_audio_clip = torch.cat(
                        (
                            torch.zeros(size=((s_w - w), speaker_audio_clip.shape[-1])),
                            speaker_audio_clip,
                            torch.zeros(size=((num_windows * w - length), speaker_audio_clip.shape[-1]))
                        ),
                        dim=0
                    )
                    speaker_emotion_clip = torch.cat(
                        (
                            torch.zeros(size=((s_w - w), speaker_emotion_clip.shape[-1])),
                            speaker_emotion_clip,
                            torch.zeros(size=((num_windows * w - length), speaker_emotion_clip.shape[-1]))
                        ),
                        dim=0
                    )
                    speaker_3dmm_clip = torch.cat(
                        (
                            torch.zeros(size=((s_w - w), speaker_3dmm_clip.shape[-1])),
                            speaker_3dmm_clip,
                            torch.zeros(size=((num_windows * w - length), speaker_3dmm_clip.shape[-1]))
                        ),
                        dim=0
                    )

                    motion_length_list = []
                    speaker_audio_clip_list = []
                    speaker_emotion_clip_list = []
                    speaker_3dmm_clip_list = []

                    for i in range(num_windows):
                        if i < num_windows - 1:
                            motion_length_list.append(w)
                        else:
                            motion_length_list.append(length - i * w)

                        speaker_audio_clip_list.append(speaker_audio_clip[i * w: i * w + s_w])
                        speaker_emotion_clip_list.append(speaker_emotion_clip[i * w: i * w + s_w])
                        speaker_3dmm_clip_list.append(speaker_3dmm_clip[i * w: i * w + s_w])

                    motion_length = torch.tensor(motion_length_list)
                    speaker_audio_clip = torch.stack(speaker_audio_clip_list, dim=0)
                    speaker_emotion_clip = torch.stack(speaker_emotion_clip_list, dim=0)
                    speaker_3dmm_clip = torch.stack(speaker_3dmm_clip_list, dim=0)

                    motion_lengths.append(motion_length)
                    speaker_audios.append(speaker_audio_clip)
                    speaker_emotions.append(speaker_emotion_clip)
                    speaker_3dmms.append(speaker_3dmm_clip)

            motion_lengths = torch.cat(motion_lengths, dim=0)
            speaker_audios = torch.cat(speaker_audios, dim=0)
            speaker_emotions = torch.cat(speaker_emotions, dim=0)
            speaker_3dmms = torch.cat(speaker_3dmms, dim=0)
            sample_batch_size = torch.tensor(sample_batch_size)

            pred_listener_emotions = []
            all_batch_size = speaker_audios.shape[0]

            for i in range(math.ceil(all_batch_size / clip_batch_size)):
                speaker_audio_clip = speaker_audios[i * clip_batch_size: (i + 1) * clip_batch_size]
                speaker_emotion_clip = speaker_emotions[i * clip_batch_size: (i + 1) * clip_batch_size]
                speaker_3dmm_clip = speaker_3dmms[i * clip_batch_size: (i + 1) * clip_batch_size]
                motion_length = motion_lengths[i * clip_batch_size: (i + 1) * clip_batch_size]

                speaker_audio_clip = speaker_audio_clip.to(self.device)
                speaker_emotion_clip = speaker_emotion_clip.to(self.device)
                speaker_3dmm_clip = speaker_3dmm_clip.to(self.device)

                with torch.no_grad():
                    outputs = model(
                        speaker_audio_input=speaker_audio_clip,
                        speaker_emotion_input=speaker_emotion_clip,
                        speaker_3dmm_input=speaker_3dmm_clip,
                        motion_length=motion_length,
                        speaker_future_emotion_input=None,
                    )

                pred_listener_emotions.append(outputs["prediction_emotion"].detach().cpu())

            pred_listener_emotions = torch.cat(pred_listener_emotions, dim=0)  # (L', num_preds, l_w, 25)

            pred_listener_emotion_list = []
            bounds = torch.cat((torch.tensor([0]), torch.cumsum(sample_batch_size, dim=0)), dim=0)
            intervals = list(zip(bounds[:-1], bounds[1:]))

            for (l, r) in intervals:
                pred_listener_emotion = pred_listener_emotions[l:r]  # (b', num_preds, l_w, 25)
                motion_length = motion_lengths[l:r]
                clip_length = torch.sum(motion_length, dim=0, keepdim=False)

                pred_listener_emotion = rearrange(
                    pred_listener_emotion,
                    'b n w d -> n (b w) d'
                )[:, :clip_length]

                if data_clamp:
                    pred_listener_emotion[:, :, :15] = torch.round(pred_listener_emotion[:, :, :15])

                pred_listener_emotion_list.append(pred_listener_emotion)

            pred_listener_emotions_all.extend(pred_listener_emotion_list)

        # post process GT so it aligns with prediction format
        if len(pred_listener_emotions_all):
            GT_listener_emotions_all = post_processor.forward(
                prediction_list=pred_listener_emotions_all,
                target_list=GT_listener_emotions_all,
            )

        try:
            torch.save(
                {
                    'GT': GT_listener_emotions_all,
                    'PRED': pred_listener_emotions_all,
                    'META': all_meta,
                },
                'results.pt'
            )
            print("Successfully saved Tensor List with META")
        except Exception as e:
            print(f"Failed to save Tensor List: {e}")

        results = compute_metrics(
            input_speaker_emotions_all,
            pred_listener_emotions_all,
            GT_listener_emotions_all,
        )
        logger.info(results)