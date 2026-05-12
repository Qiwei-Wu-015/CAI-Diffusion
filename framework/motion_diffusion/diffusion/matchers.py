"""
Code adapted from:
https://github.com/BarqueroGerman/BeLFusion
"""
from pathlib import Path
import hydra
import torch
import torch.nn as nn
import os
from einops import rearrange
from omegaconf import DictConfig
from hydra.utils import instantiate
from framework.motion_diffusion.diffusion.diffusion_decoder.transformer_denoiser import TransformerDenoiser, \
    lengths_to_mask
from framework.motion_diffusion.diffusion.gaussian_diffusion import DecoderLatentDiffusion
from framework.motion_diffusion.diffusion.resample import UniformSampler
from framework.motion_diffusion.diffusion.rnn import LatentEmbedder
from framework.utils.util import from_pretrained_checkpoint, save_checkpoint


class BaseLatentModel(nn.Module):
    def __init__(self, cfg, emb_preprocessing=False, freeze_encoder=True, **kwargs):
        super(BaseLatentModel, self).__init__()
        self.emb_preprocessing = emb_preprocessing
        self.freeze_encoder = freeze_encoder
        def_dtype = torch.get_default_dtype()

        self.audio_encoder = instantiate(cfg.audio_encoder)
        if cfg.latent_embedder is not None:
            self.latent_embedder = instantiate(cfg.latent_embedder)
            model_path = os.path.join(hydra.utils.get_original_cwd(), cfg.latent_embedder.checkpoint_path)
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            self.latent_embedder.load_state_dict(state_dict)
            print(f"Successfully loaded latent embedder from {model_path}")
        else:
            self.latent_embedder = LatentEmbedder()

        if self.freeze_encoder:  # freeze modules
            for para in self.latent_embedder.parameters():
                para.requires_grad = False

        torch.set_default_dtype(def_dtype)
        self.init_params = None

    def deepcopy(self):
        assert self.init_params is not None, "Cannot deepcopy LatentUNetMatcher if init_params is None."
        # I can't deep copy this class. I need to do this trick to make the deepcopy of everything
        model_copy = self.__class__(**self.init_params)
        weights_path = f'weights_temp_{id(model_copy)}.pt'
        torch.save(self.state_dict(), weights_path)
        model_copy.load_state_dict(torch.load(weights_path))
        os.remove(weights_path)
        return model_copy

    def preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb  # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return (emb - stats["mean"]) / torch.sqrt(stats["var"])
        elif "normalize" in self.emb_preprocessing:
            return 2 * (emb - stats["min"]) / (stats["max"] - stats["min"]) - 1
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def undo_preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb  # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return torch.sqrt(stats["var"]) * emb + stats["mean"]
        elif "normalize" in self.emb_preprocessing:
            return (emb + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def forward(self, pred, timesteps, seq_em):
        raise NotImplementedError("This is an abstract class.")

    # override checkpointing
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def cuda(self):
        return self.to(torch.device("cuda"))

    # override eval and train
    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()


class DecoderLatentMatcher(BaseLatentModel):
    def __init__(self,
                 conf: DictConfig = None,
                 module_dict_cfg: DictConfig = None,
                 stage: str = 'fit',
                 task: str = 'online',
                 **kwargs):
        cfg = conf.args
        super(DecoderLatentMatcher, self).__init__(
            module_dict_cfg,
            emb_preprocessing=cfg.emb_preprocessing,
            freeze_encoder=cfg.freeze_encoder,
            **kwargs,
        )

        self.stage = stage
        self.task = task
        self.token_len = cfg.token_len
        self.window_size = cfg.get("window_size", 30)
        self.s_ratio = cfg.get("s_ratio", 2)
        self.emotion_dim = cfg.get("nfeats", 25)
        self.encode_emotion = cfg.get("encode_emotion", False)
        self.encode_3dmm = cfg.get("encode_3dmm", False)

        self.init_params = {
            "task": task,
            "window_size": self.window_size,
            "encode_emotion": self.encode_emotion,
            "encode_3dmm": self.encode_3dmm,
            "ablation_skip_connection": cfg.get("ablation_skip_connection", True),
            "nfeats": cfg.get("nfeats", 25),
            "latent_dim": cfg.get("latent_dim", 512),
            "ff_size": cfg.get("ff_size", 1024),
            "num_layers": cfg.get("num_layers", 6),
            "num_heads": cfg.get("num_heads", 4),
            "dropout": cfg.get("dropout", 0.1),
            "normalize_before": cfg.get("normalize_before", False),
            "activation": cfg.get("activation", "gelu"),
            "flip_sin_to_cos": cfg.get("flip_sin_to_cos", True),
            "return_intermediate_dec": cfg.get("return_intermediate_dec", False),
            "position_embedding": cfg.get("position_embedding", "learned"),
            "arch": cfg.get("arch", "trans_enc"),
            "freq_shift": cfg.get("freq_shift", 0),
            "time_encoded_dim": cfg.get("time_encoded_dim", 64),
            "s_audio_dim": cfg.get("s_audio_dim", 768),
            "s_audio_scale": cfg.get("s_audio_scale", cfg.get("latent_dim", 512) ** -0.5),
            "s_emotion_dim": cfg.get("s_emotion_dim", 25),
            "s_3dmm_dim": cfg.get("s_3dmm_dim", 58),
            "concat": cfg.get("concat", "concat_first"),
            "condition_concat": cfg.get("condition_concat", "token_concat"),
            "guidance_scale": cfg.get("guidance_scale", 7.5),
            "s_audio_enc_drop_prob": cfg.get("s_audio_enc_drop_prob", 0.2),
            "s_latent_embed_drop_prob": cfg.get("s_latent_embed_drop_prob", 0.2),
            "s_3dmm_enc_drop_prob": cfg.get("s_3dmm_enc_drop_prob", 0.2),
            "s_emotion_enc_drop_prob": cfg.get("s_emotion_enc_drop_prob", 1.0),
            "past_l_emotion_drop_prob": cfg.get("past_l_emotion_drop_prob", 1.0),
        }
        self.use_past_frames = cfg.get("use_past_frames", False)

        self.model = TransformerDenoiser(**self.init_params)
        
        # Initialize SpeakerEmotionPredictor
        self.speaker_emotion_predictor = SpeakerEmotionPredictor(
            emotion_dim=self.init_params["s_emotion_dim"],
            audio_dim=self.init_params["s_audio_dim"],
            _3dmm_dim=self.init_params["s_3dmm_dim"],
            hidden_dim=256,
            num_layers=2,
            output_dim=self.emotion_dim,
            future_frames=10
        )

        self.decoder_diffusion = DecoderLatentDiffusion(
            conf.scheduler,
            conf.scheduler.num_train_timesteps,
            conf.scheduler.num_inference_timesteps,
        )
        self.schedule_sampler = UniformSampler(self.decoder_diffusion)
        self.num_preds = conf.scheduler.num_preds

    def _forward(
            self,
            speaker_audio_input=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            past_listener_emotion=None,
            motion_length=None,
            speaker_future_emotion_input=None,
    ):
        # Check dimensions and align sequence lengths
        if speaker_audio_input is not None and speaker_emotion_input is not None and speaker_3dmm_input is not None:
            pass
            #  print(f"Speaker Audio Input Shape: {speaker_audio_input.shape}")
            #  print(f"Speaker Emotion Input Shape: {speaker_emotion_input.shape}")
            #  print(f"Speaker 3DMM Input Shape: {speaker_3dmm_input.shape}")
             
            #  # Ensure all inputs have the same sequence length
            #  min_len = min(speaker_audio_input.shape[1], speaker_emotion_input.shape[1], speaker_3dmm_input.shape[1])
             
            #  if speaker_audio_input.shape[1] > min_len:
            #      speaker_audio_input = speaker_audio_input[:, :min_len, :]
            #  if speaker_emotion_input.shape[1] > min_len:
            #      speaker_emotion_input = speaker_emotion_input[:, :min_len, :]
            #  if speaker_3dmm_input.shape[1] > min_len:
            #      speaker_3dmm_input = speaker_3dmm_input[:, :min_len, :]

        # Predict future speaker emotion
        # speaker_emotion_input: (batch_size, seq_len, emotion_dim)
        pred_future_emotion = self.speaker_emotion_predictor(speaker_emotion_input, speaker_audio_input, speaker_3dmm_input)
        
        # If speaker_future_emotion_input is provided (during training), use it for loss calculation later
        # But for diffusion conditioning, we use the PREDICTED future emotion
        
        with torch.no_grad():
            s_audio_encodings = self.audio_encoder._encode(speaker_audio_input)
            s_audio_encodings = s_audio_encodings.repeat_interleave(self.num_preds, dim=0)

          # freeze latent RNN_VAE embedder to extract speaker latent embedding
            s_latent_embed = self.latent_embedder.encode(speaker_emotion_input).unsqueeze(1)
            s_latent_embed = s_latent_embed.repeat_interleave(self.num_preds, dim=0)
            # shape: (batch_size * num_preds, 1, ...)

            # s_3dmm_encodings = self.latent_3dmm_embedder.get_encodings(speaker_3dmm_input)
            s_3dmm_encodings = speaker_3dmm_input.repeat_interleave(self.num_preds, dim=0)
            # shape: (bs * num_preds, s_w, ...)

            s_emotion_encodings = speaker_emotion_input.repeat_interleave(self.num_preds, dim=0)
            # shape: (bs * num_preds, s_w, ...)

            past_listener_emotion = past_listener_emotion.repeat_interleave(
                self.num_preds, dim=0) if past_listener_emotion is not None else None
            # shape: (bs * num_preds, l_w, ...)

            motion_length = motion_length.repeat_interleave(
                self.num_preds, dim=0) if motion_length is not None else None
            
            # Repeat predicted future emotion for num_preds
            # speaker_future_emotion_prediction = pred_future_emotion.repeat_interleave(self.num_preds, dim=0)
            # 切断计算图
            speaker_future_emotion_prediction = pred_future_emotion.detach().repeat_interleave(self.num_preds, dim=0)

            model_kwargs = {
                "speaker_audio_encodings": s_audio_encodings,
                "speaker_latent_embed": s_latent_embed,
                "speaker_3dmm_encodings": s_3dmm_encodings,
                "speaker_emotion_encodings": s_emotion_encodings,
                "past_listener_emotion": past_listener_emotion,
                "motion_length": motion_length,
                "speaker_future_emotion_prediction": speaker_future_emotion_prediction,
            }

        if self.stage == "test":
            bs, l, _ = s_audio_encodings.shape  # bz * num_preds
            with torch.no_grad():
                output = [output for output in self.decoder_diffusion.ddim_sample_loop_progressive(
                    matcher=self,
                    model=self.model,
                    model_kwargs=model_kwargs,
                    shape=(bs, self.window_size if self.task == "online" else l, self.emotion_dim),
                )][-1]  # get last output

            output_listener_emotion = output["sample_enc"]  # (bz * num_preds, l_w, d=25)
            output_listener_emotion = rearrange(output_listener_emotion,
                                                "(b n) w d -> b n w d", n=self.num_preds)
            output_whole = {"prediction_emotion": output_listener_emotion}

        else:
            listener_emotion_input = listener_emotion_input.repeat_interleave(self.num_preds, dim=0)
            x_start_selected = listener_emotion_input  # (bs * num_preds, l_w, ...)

            t, _ = self.schedule_sampler.sample(x_start_selected.shape[0], x_start_selected.device)
            timesteps = t.long()

            output_whole = self.decoder_diffusion.denoise(self.model, x_start_selected, timesteps,
                                                          model_kwargs=model_kwargs)
            if motion_length is not None:  # offline task zero masking
                device = x_start_selected.get_device()
                output_mask = lengths_to_mask(motion_length, device=device, max_len=x_start_selected.shape[1])
                # print(f'output_whole["prediction_emotion"] shape: {output_whole["prediction_emotion"].shape}')
                output_whole["prediction_emotion"] = (output_whole["prediction_emotion"]
                                                      * output_mask.float().unsqueeze(-1))

            output_whole = {k: v.view(-1, self.num_preds, *output_whole[k].shape[1:]) for k, v in output_whole.items()}
        
        # Add prediction results to output
        output_whole["pred_future_emotion"] = pred_future_emotion
        if speaker_future_emotion_input is not None:
            output_whole["target_future_emotion"] = speaker_future_emotion_input
            
        return output_whole

    def forward(self, **kwargs):
        return self._forward(**kwargs)


class LatentMatcher(nn.Module):
    def __init__(self,
                 task: str = "online",
                 stage: str = "fit",
                 device: str = None,
                 diffusion_decoder: DictConfig = None,
                 latent_embedder: DictConfig = None,
                 audio_encoder: DictConfig = None,
                 resumed_training: bool = False,
                 **kwargs):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.task = task
        self.stage = stage
        self.kwargs = kwargs

        module_dict_cfg = DictConfig(
            {"latent_embedder": latent_embedder,
             "audio_encoder": audio_encoder,}
        )

        self.diffusion_decoder_cfg = diffusion_decoder
        self.diffusion_decoder = DecoderLatentMatcher(self.diffusion_decoder_cfg,
                                                      task=task,
                                                      stage=stage,
                                                      module_dict_cfg=module_dict_cfg,
                                                      **kwargs)
        load_ckpt = False
        want_last = False
        want_best = False

        if resumed_training:
            load_ckpt = True
            want_last = True
        if stage == "test":
            load_ckpt = True
            want_best = True

        if load_ckpt:
            ckpt_path = self.get_ckpt_path(
                self.diffusion_decoder.model,
                runid="resume_runid",
                epoch=None,
                best=want_best,
                last=want_last,
            )
            from_pretrained_checkpoint(str(ckpt_path), self.diffusion_decoder.model, device)

    def forward(
            self,
            speaker_audio_input=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            past_listener_emotion=None,
            motion_length=None,
            speaker_future_emotion_input=None,
    ):

        outputs = self.diffusion_decoder.forward(
            speaker_audio_input=speaker_audio_input,
            speaker_emotion_input=speaker_emotion_input,
            speaker_3dmm_input=speaker_3dmm_input,
            listener_emotion_input=listener_emotion_input,
            past_listener_emotion=past_listener_emotion,
            motion_length=motion_length,
            speaker_future_emotion_input=speaker_future_emotion_input,
        )
        # outputs['prediction_emotion']: (bz, num_preds, s_w, emotion_dim)
        return outputs

    def get_ckpt_path(self, model, runid="current_runid", epoch=None, best=False, last=False):
        ckpt_dir = Path(hydra.utils.to_absolute_path(self.kwargs.get("ckpt_dir")))
        run_id = Path(self.kwargs.get(runid))
        ckpt_dir = str(ckpt_dir / run_id / model.get_model_name())
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = None
        if epoch is not None:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{epoch}.pth")
        if best:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
        if last:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_last.pth")
        assert ckpt_path is not None, "No checkpoint path is provided."
        return ckpt_path

    def save_ckpt(self, optimizer, epoch=None, best=False, last=False, best_loss=float("inf")):
        model = self.diffusion_decoder.model
        ckpt_path = self.get_ckpt_path(model, epoch=epoch, best=best, last=last)
        save_checkpoint(ckpt_path, model, optimizer, epoch=epoch, best_loss=best_loss)


class SpeakerEmotionPredictor(nn.Module):
    """Autoregressive speaker emotion predictor.

    Encoder: a GRU that ingests concatenated (emotion, audio_proj, _3dmm_proj).
    Decoder: a GRUCell that generates one future frame at a time, using the
    previous predicted emotion as the next-step input. The public API and the
    returned tensor shape remain the same as before: (batch, future_frames, output_dim).
    """
    def __init__(self, emotion_dim=25, audio_dim=768, _3dmm_dim=58, hidden_dim=256, num_layers=2, output_dim=25, future_frames=10):
        super(SpeakerEmotionPredictor, self).__init__()
        self.future_frames = future_frames
        self.output_dim = output_dim

        # Feature projection layers (same as before)
        self.audio_proj = nn.Linear(audio_dim, 64)
        self._3dmm_proj = nn.Linear(_3dmm_dim, 32)

        input_dim = emotion_dim + 64 + 32

        # Encoder GRU (process history)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Decoder: GRUCell that consumes previous predicted emotion (output_dim)
        # and produces a hidden state; fc maps hidden -> single-step prediction
        self.decoder_cell = nn.GRUCell(input_size=output_dim, hidden_size=hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, emotion, audio, _3dmm):
        # emotion: (batch_size, seq_len, emotion_dim)
        # audio: (batch_size, seq_len, audio_dim)
        # _3dmm: (batch_size, seq_len, _3dmm_dim)

        # Project features
        audio_feat = torch.relu(self.audio_proj(audio))
        _3dmm_feat = torch.relu(self._3dmm_proj(_3dmm))

        # Concatenate features -> encoder input
        combined_input = torch.cat([emotion, audio_feat, _3dmm_feat], dim=-1)

        # Encode history
        # output: (batch, seq_len, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        _, h_n = self.gru(combined_input)

        # Initialize decoder hidden state with the top layer of encoder hidden
        # h_n[-1]: (batch, hidden_dim)
        decoder_hidden = h_n[-1]

        # Autoregressive decoding: feed last observed emotion as first input
        # decoder_input shape: (batch, output_dim)
        # If emotion_dim != output_dim, we align by taking the first output_dim dims
        # (in typical usage emotion_dim == output_dim)
        last_input_frame = emotion[:, -1, :]
        if last_input_frame.shape[-1] != self.output_dim:
            # project/truncate/pad as needed (truncate or pad with zeros)
            if last_input_frame.shape[-1] > self.output_dim:
                decoder_input = last_input_frame[:, :self.output_dim]
            else:
                pad_size = self.output_dim - last_input_frame.shape[-1]
                decoder_input = torch.cat([last_input_frame, last_input_frame.new_zeros(last_input_frame.shape[0], pad_size)], dim=-1)
        else:
            decoder_input = last_input_frame

        preds = []
        for _ in range(self.future_frames):
            # GRUCell expects (batch, input_size) and (batch, hidden_size)
            decoder_hidden = self.decoder_cell(decoder_input, decoder_hidden)
            out = self.fc(decoder_hidden)  # (batch, output_dim)
            preds.append(out.unsqueeze(1))
            # next input is the current prediction (autoregressive)
            decoder_input = out

        # Stack predictions -> (batch, future_frames, output_dim)
        prediction = torch.cat(preds, dim=1)

        # Preserve previous behavior: add last input frame as residual (broadcast)
        # This keeps the output in the same scale as before.
        last_input_expanded = last_input_frame.unsqueeze(1).expand(-1, self.future_frames, -1)
        prediction = prediction + last_input_expanded

        return prediction