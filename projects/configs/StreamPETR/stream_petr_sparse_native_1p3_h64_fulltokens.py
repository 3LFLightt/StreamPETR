# Control: same sparse metadata pipeline/detector, but full backbone + all tokens.
_base_ = ['./stream_petr_sparse_native_1p3_h64.py']
model = dict(sparse_mode='full_tokens')
