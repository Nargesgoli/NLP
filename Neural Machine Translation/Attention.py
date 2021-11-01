import random
import numpy as np
import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from trax.supervised import training
# pip list | grep trax
path=Path(r"C:\Users\nrgsg\OneDrive\Desktop\transformer\data").expanduser()
train_stream_fn = trax.data.TFDS('opus/medical',
                                 data_dir=path,
                                 keys=('en', 'de'),
                                 eval_holdout_size=0.01, # 1% for eval
                                 train=True)

# Get generator function for the eval set
eval_stream_fn = trax.data.TFDS('opus/medical',
                                data_dir=path,
                                keys=('en', 'de'),
                                eval_holdout_size=0.01, # 1% for eval
                                train=False)
train_stream = train_stream_fn()
eval_stream = eval_stream_fn()
print('train data (en, de) tuple:', next(train_stream))
print('train data (en, de) tuple:', next(eval_stream))
VOCAB_FILE = 'ende_32k.subword'
VOCAB_DIR = Path(r"C:\Users\nrgsg\OneDrive\Desktop\transformer\data").expanduser()
tokenized_train_stream = trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)(train_stream)
tokenized_eval_stream = trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR)(eval_stream)
print(next(tokenized_train_stream))
print(next(train_stream))
EOS = 1
def append_eos(stream):
    for (inputs, targets) in stream:
        inputs_with_eos = list(inputs) + [EOS]
        targets_with_eos = list(targets) + [EOS]
        yield np.array(inputs_with_eos), np.array(targets_with_eos)
tokenized_train_stream = append_eos(tokenized_train_stream)
tokenized_eval_stream = append_eos(tokenized_eval_stream)
filtered_train_stream = trax.data.FilterByLength(
    max_length=256, length_keys=[0, 1])(tokenized_train_stream)
filtered_eval_stream = trax.data.FilterByLength(
    max_length=512, length_keys=[0, 1])(tokenized_eval_stream)
train_input, train_target = next(filtered_train_stream)
boundaries =  [8,   16,  32, 64, 128, 256, 512]
batch_sizes = [256, 128, 64, 32, 16,    8,   4,  2]
train_batch_stream = trax.data.BucketByLength(boundaries, batch_sizes,length_keys=[0, 1])(filtered_train_stream)
eval_batch_stream = trax.data.BucketByLength(boundaries, batch_sizes,length_keys=[0, 1])(filtered_eval_stream)

MODE = 'train'
D_MODEL = 512
D_FF = 2048
N_LAYERS = 6
N_HEADS = 8
MAX_SEQUENCE_LENGTH = 2048
DROPOUT_RATE = .1
DROPOUT_SHARED_AXES = None
FF_ACTIVATION_TYPE = tl.Relu

def Transformer(input_vocab_size,
                output_vocab_size=None,
                d_model=D_MODEL,
                d_ff=D_FF,
                n_encoder_layers=N_LAYERS,
                n_decoder_layers=N_LAYERS,
                n_heads=N_HEADS,
                max_len=MAX_SEQUENCE_LENGTH,
                dropout=DROPOUT_RATE,
                dropout_shared_axes=DROPOUT_SHARED_AXES,
                mode=MODE,
                ff_activation=FF_ACTIVATION_TYPE):
  encoder_mode = 'eval' if mode == 'predict' else mode
  in_embedder = tl.Embedding(input_vocab_size, d_model)
  if output_vocab_size is None:
    out_embedder = in_embedder
    output_vocab_size = input_vocab_size
  else:
    out_embedder = tl.Embedding(output_vocab_size, d_model)
  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)
  def _EncBlock():
    return _EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                         mode, ff_activation)
  def _Encoder():
    encoder = tl.Serial(
        in_embedder,
        _Dropout(),
        tl.PositionalEncoding(max_len=max_len, mode=encoder_mode),
        [_EncBlock() for _ in range(n_encoder_layers)],
        tl.LayerNorm(),
    )
    return tl.Cache(encoder) if mode == 'predict' else encoder
  def _EncDecBlock():
    return _EncoderDecoderBlock(d_model, d_ff, n_heads, dropout,
                                dropout_shared_axes, mode, ff_activation)
  return tl.Serial(
      tl.Select([0, 1, 1]),  # Copies decoder tokens for use in loss.

      # Encode.
      tl.Branch([], tl.PaddingMask()),  # tok_e masks tok_d tok_d
      _Encoder(),

      # Decode.
      tl.Select([2, 1, 0]),  # Re-orders inputs: tok_d masks vec_e .....
      tl.ShiftRight(mode=mode),
      out_embedder,
      _Dropout(),
      tl.PositionalEncoding(max_len=max_len, mode=mode),
      tl.Branch([], tl.EncoderDecoderMask()),  # vec_d masks ..... .....
      [_EncDecBlock() for _ in range(n_decoder_layers)],
      tl.LayerNorm(),
      tl.Select([0], n_in=3),  # Drops masks and encoding vectors.

      # Map vectors to match output vocab size.
      tl.Dense(output_vocab_size),
  )

def _EncoderBlock(d_model,
                  d_ff,
                  n_heads,
                  dropout,
                  dropout_shared_axes,
                  mode,
                  ff_activation):
  def _Attention():
    return tl.Attention(d_model, n_heads=n_heads, dropout=dropout, mode=mode)
  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)
  def _FFBlock():
    return _FeedForwardBlock(d_model, d_ff, dropout, dropout_shared_axes, mode,
                             ff_activation)
  return [
      tl.Residual(
          tl.LayerNorm(),
          _Attention(),
          _Dropout(),
      ),
      tl.Residual(
          tl.LayerNorm(),
          _FFBlock(),
          _Dropout(),
      ),
  ]

def _EncoderDecoderBlock(d_model,
                         d_ff,
                         n_heads,
                         dropout,
                         dropout_shared_axes,
                         mode,
                         ff_activation):
  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)
  def _AttentionQKV():
    return tl.AttentionQKV(d_model, n_heads=n_heads, dropout=dropout,
                           mode=mode, cache_KV_in_predict=True)
  def _CausalAttention():
    return tl.CausalAttention(d_model, n_heads=n_heads, mode=mode)
  def _FFBlock():
    return _FeedForwardBlock(d_model, d_ff, dropout, dropout_shared_axes, mode,
                             ff_activation)
  return [
      tl.Residual(
          tl.LayerNorm(),
          _CausalAttention(),
          _Dropout(),
      ),
      tl.Residual(
          tl.LayerNorm(),
          tl.Select([0, 2, 2, 1, 2]),
          _AttentionQKV(),
          _Dropout(),
      ),
      tl.Residual(
          tl.LayerNorm(),
          _FFBlock(),
          _Dropout(),
      ),
  ]
train_task = training.TrainTask(
    labeled_data= train_batch_stream,
    loss_layer= tl.CrossEntropyLoss(),
    optimizer= trax.optimizers.Adam(0.01),
    lr_schedule= trax.lr.warmup_and_rsqrt_decay(1000,0.01),
    n_steps_per_checkpoint= 10,)
eval_task = training.EvalTask(
    labeled_data=eval_batch_stream,
    metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],)
training_loop = training.Loop(transformer(mode='train'),
                              train_task,
                              eval_tasks=[eval_task],
                              output_dir=output_dir)
training_loop.run(10)
