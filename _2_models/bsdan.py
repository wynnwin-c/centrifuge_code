# _2_models/bsdan.py
"""
BSDAN entry preserved for compatibility.
Current implementation intentionally mirrors Partial-Domain original BSAN logic:
- 4-class output space
- source middle-sample expansion via len_share
- BSAN local adversarial loss and shared-class entropy
Use TransferNet with transfer_loss="bsdan" to get the same behavior as Partial-Domain BSAN.
"""

from .transfer_stack.transfernet import TransferNet


def build_bsdan(in_channels: int, num_classes: int, feat_dim: int = 256, batch_size: int = 32, shared_classes=None, max_iter: int = 1000):
    return TransferNet(
        num_class=num_classes,
        model_name='cnn_features',
        in_channel=in_channels,
        transfer_loss='bsdan',
        max_iter=max_iter,
        trade_off_adversarial='Step',
        lam_adversarial=1,
        batch_size=batch_size,
        shared_classes=shared_classes or [],
    )
