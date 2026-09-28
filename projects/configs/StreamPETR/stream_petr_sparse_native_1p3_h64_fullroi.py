# Diagnostic control: full six-camera backbone, ROI tokens only in cross-attention.
_base_ = ['./stream_petr_sparse_native_1p3_h64.py']
model = dict(sparse_mode='full_roi')
