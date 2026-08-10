"""EEGNet - the seizure-detection model.

Consumes the ``(batch, 1, n_channels, n_times)`` windows served by
:class:`...data.RecordWindowDataset` and returns raw class logits for
:class:`torch.nn.CrossEntropyLoss`. At the default CHB-MIT configuration
(22 channels, 3 s × 256 Hz = 768 samples) the whole network is ~2.2K
parameters — deliberately tiny, because with only minutes of ictal data
per patient, overfitting is the dominant failure mode and parameter
scarcity *is* the regularization and can run easily inside a modern
wearable device.

How to read the architecture (for readers new to EEGNet)
--------------------------------------------------------
The input is laid out as a 2-D "image" of shape ``(n_channels, n_times)``
— electrodes down the rows, time across the columns — with a dummy depth
axis of 1 so the ``Conv2d`` machinery applies. Every kernel in this
network is deliberately **1-D in disguise**: either ``(1, k)`` (slides
along time, treats each electrode row independently) or
``(n_channels, 1)`` (spans all electrodes at a single time instant).
That factorisation is the whole idea of EEGNet — it mirrors how a
neurologist reads an EEG:

1. *"Which frequencies look abnormal?"* → temporal filtering first
   (:attr:`temporal_filtering` — a learned band-pass filter bank).
2. *"Which electrodes show it?"* → spatial filtering second
   (:attr:`spatial_filtering` — learned electrode-weighting maps,
   fitted separately per frequency band).
3. Summarise how those band-and-location signals evolve across the
   window (:attr:`temporal_summary`) and classify (:attr:`classifier`).

Normalization lives inside the model, not in preprocessing
----------------------------------------------------------
The first operation in :meth:`forward` is a per-channel z-score computed
from the window itself: no calibration recordings, no running statistics,
no state carried between windows — properties chosen so the exact same 
computation can run on a wearable stream. Note it is implemented **manually**
rather than with ``nn.InstanceNorm2d(1)``: with the 
``(batch, 1, n_channels, n_times)`` layout the electrodes live in a
*spatial* dimension, so ``InstanceNorm2d(1)`` would compute one global
mean/std over the whole ``(n_channels × n_times)`` block — cross-channel
amplitude differences (a quiet temporal electrode vs a loud frontal one)
would survive, which must be absorbed.

Reference: Lawhern et al. (2018), *EEGNet: A Compact Convolutional
Network for EEG-based Brain-Computer Interfaces* — workflow §19.
"""

import torch
from torch import nn


class EEGNet(nn.Module):
    """Compact CNN for per-window binary seizure detection.

    Three convolutional blocks and a linear head. Shapes below are for
    the default CHB-MIT config (``n_channels=22``, ``segment_length=3``,
    ``sfreq=256`` → ``n_times=768``):

    ====================  =====================  ============================
    Block                 Output shape           Role
    ====================  =====================  ============================
    (input)               ``(B, 1, 22, 768)``    z-scored raw voltage window
    temporal_filtering    ``(B, 8, 22, 768)``    8 learned band-pass filters
    spatial_filtering     ``(B, 16, 1, 192)``    2 electrode-maps per band;
                                                 channel axis collapsed
    temporal_summary      ``(B, 16, 1, 24)``     temporal envelope summary
    classifier            ``(B, n_classes)``     logits
    ====================  =====================  ============================

    Returns **raw logits** — no softmax. :class:`torch.nn.CrossEntropyLoss`
    applies ``log_softmax`` internally, so a softmax here would silently
    double-apply it and flatten the gradients. Apply ``softmax`` outside
    the model when probabilities are needed (e.g. P(ictal) for the
    post-processing state machine).
    """

    def __init__(
        self,
        segment_length: float,
        sfreq: float,
        n_channels: int,
        n_classes: int,
    ):
        """Build the network and size the classifier via a dummy forward pass.

        Args:
            segment_length: Window length in seconds. Together with
                ``sfreq`` this fixes ``n_times = segment_length * sfreq``,
                which must match the windows the Dataset serves —
                a mismatch surfaces as a shape error on the first real
                forward pass. From ``config.yaml``
                (``data.chb_mit.segment_length``).
            sfreq: Sampling frequency in Hz (256 for every CHB-MIT
                record; the loader asserts this).
            n_channels: Electrode count after channel harmonisation —
                22 for the configured CHB-MIT bipolar montage.
                Sets the height of the spatial-filter kernel, so the model
                adapts if the montage changes.
            n_classes: Number of output classes (2 for seizure-detection:
                interictal vs ictal).
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Block 1 — temporal filtering: a LEARNED band-pass filter bank.
        #
        # Kernel (1, 64) slides along the time axis only: each of the 8
        # output maps convolves every electrode row with the same 64-sample
        # (= 0.25 s at 256 Hz) temporal pattern. Convolving with a fixed
        # waveform IS a frequency filter, so after training these 8 kernels
        # typically settle into band-pass shapes resembling the clinical
        # bands (delta/theta/alpha/beta/gamma) plus a few seizure-specific
        # morphologies. Take the FFT of each learned kernel to see which
        # band it tuned itself to (workflow §15).
        #
        # Why length 64: a filter of duration T resolves rhythms down to
        # ~1/T Hz, so 0.25 s reaches ~4 Hz cleanly — covering everything
        # from theta upward. Why padding='same': keep all time samples so
        # no signal is lost before the network decides what matters.
        # ------------------------------------------------------------------
        self.temporal_filtering = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 64),
                padding='same',
                # bias=False on every conv in this network: each conv is
                # immediately followed by BatchNorm, whose learned shift
                # (beta) makes a conv bias redundant parameters.
                bias=False
            ),

            # BatchNorm keeps each of the 8 filtered maps at a stable scale
            # across batches, which stabilises and speeds up optimisation —
            # the following blocks always see inputs in the same numeric
            # range regardless of how the filters evolve during training.
            nn.BatchNorm2d(8),
        )

        # ------------------------------------------------------------------
        # Block 2 — spatial filtering: WHERE on the scalp each band lives.
        #
        # Kernel (n_channels, 1) is the mirror image of Block 1's: it spans
        # ALL electrodes at a single time instant. Each output map is a
        # learned weighted sum of the electrodes — a spatial topography
        # ("this pattern = left-temporal focus") applied at every time
        # step. Conceptually a learned ICA/CSP step. The channel axis
        # collapses to height 1: from here on, "space" has been absorbed
        # into the map identity. Plot each kernel's 22 weights with
        # mne.viz.plot_topomap to see where the model looks.
        #
        # groups=8 makes this DEPTHWISE: each of the 8 band maps gets its
        # own spatial filters, never mixing bands. With out_channels=16
        # that is 2 spatial maps per band ("depth multiplier 2"). This
        # per-band spatial fitting is EEGNet's core insight: different
        # seizure rhythms arise from different scalp locations, so the
        # delta-band topography must be free to differ from the beta one.
        # ------------------------------------------------------------------
        self.spatial_filtering = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(n_channels, 1),
                groups=8,
                bias=False
            ),

            nn.BatchNorm2d(16),

            # ELU over ReLU (per the EEGNet paper): smooth everywhere and
            # saturating to -1 for negative inputs rather than clipping to
            # 0, so negative filter responses still carry gradient — found
            # to work better on EEG, where "signal absent" and "signal
            # inverted" are different states worth distinguishing.
            nn.ELU(),

            # Average-pool along time by 4 (768 → 192; effective rate now
            # 64 Hz). Averaging acts as temporal smoothing — close to a
            # short-window band-power estimate — keeping sustained activity
            # while diluting single-sample spikes. MaxPool would do the
            # opposite (latch onto the loudest instant), making the model
            # more sensitive to transient artifacts like electrode pops.
            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),

            # Aggressive 50% dropout: with only minutes of
            # ictal data per patient, even ~2K parameters can memorise.
            nn.Dropout(0.5)
        )

        # ------------------------------------------------------------------
        # Block 3 — temporal summary: how the band-and-place signals EVOLVE.
        #
        # A "separable" convolution = depthwise + pointwise, a factorised
        # stand-in for a full conv at a fraction of the parameters:
        #  * depthwise (1, 16), groups=16 — each of the 16 maps gets its
        #    own 16-sample temporal kernel (~0.25 s at the pooled 64 Hz
        #    rate), learning that map's envelope dynamics in isolation;
        #  * pointwise (1, 1) — relearns how to combine the 16 maps at
        #    every time step, the cross-map mixing the depthwise skipped.
        # A full conv would need 16 input × 16 tap weights per output map;
        # the factorised pair splits pattern-finding from mixing and costs
        # ~8× fewer parameters for the same expressive family.
        # ------------------------------------------------------------------
        self.temporal_summary = nn.Sequential(
            nn.Conv2d(  # depthwise: per-map temporal kernel
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 16),
                groups=16,
                padding='same',
                bias=False
            ),
            nn.Conv2d(  # pointwise: cross-map mixing
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 1),
                bias=False
            ),

            nn.BatchNorm2d(16),

            nn.ELU(),

            # Second time-pool by 8 (192 → 24; effective rate 8 Hz). Total
            # downsampling is 32×: the classifier decides from 24 coarse
            # time steps per window rather than 768 raw samples — far fewer
            # parameters and far less to overfit.
            nn.AvgPool2d(
                kernel_size=(1, 8)
            ),

            nn.Dropout(0.5)
        )

        # ------------------------------------------------------------------
        # Size the classifier by measurement, not arithmetic: the flattened
        # feature count depends on padding modes and two floor-divisions in
        # the pools, so a closed-form formula is fragile — pushing one dummy
        # window through the three blocks measures it exactly.
        #
        # eval() + no_grad() around the dummy pass, then train() after: in
        # train mode BatchNorm would fold the all-zero dummy batch into its
        # running statistics (running_var 1.0 → 0.9) before real data ever
        # arrives. eval mode reads the stats without updating them, and
        # train() restores the freshly-constructed default so training
        # starts in the state callers expect.
        # ------------------------------------------------------------------
        self.eval()
        with torch.no_grad():
            n_times = int(segment_length * sfreq)
            dummy = torch.zeros(1, 1, n_channels, n_times)
            dummy = self.temporal_summary(
                self.spatial_filtering(
                    self.temporal_filtering(dummy)
                )
            )
        flattened_size = dummy.shape[1:].numel()  # e.g. (16, 1, 24) → 384
        self.train()

        # ------------------------------------------------------------------
        # Classifier: unroll the (16, 1, 24) feature map into a flat vector
        # (keeping the batch axis) and map linearly to class logits.
        # Deliberately no softmax — see the class docstring.
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),

            nn.Linear(
                in_features=flattened_size,
                out_features=n_classes
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of EEG windows.

        Args:
            x: ``float32`` tensor of shape
                ``(batch, 1, n_channels, n_times)`` — filtered but
                unnormalised voltage windows, as served by
                :class:`...data.RecordWindowDataset` /
                :class:`...data.ContinuousRecordDataset`.

        Returns:
            ``(batch, n_classes)`` tensor of raw logits. Apply
            ``softmax(dim=1)`` externally when probabilities are needed.
        """
        # Per-channel z-score over time: each EEG channel of
        # each window is normalised by its own mean/std computed over the
        # time axis only, so electrode-to-electrode amplitude differences
        # (impedance, location) are absorbed and the network sees morphology
        # at a uniform scale. Stateless by design - nothing carried between
        # windows - so the identical computation runs on a live wearable
        # stream. NB: nn.InstanceNorm2d(1) is NOT equivalent under this
        # input layout (see module docstring).
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)

        x = self.temporal_summary(
            self.spatial_filtering(
                self.temporal_filtering(x)
            )
        )

        out = self.classifier(x)

        return out
