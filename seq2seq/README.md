# PrefixTuning code 흐름 설명

## 파일
```
.
├── gpt2                          # Code for GPT2 style autoregressive LM
│   ├── train_e2e.py              # high-level scripts to train.
│   ├── train_control.py          # code that implements prefix-tuning.
│   ├── trainer_prefix.py         # trainer code for the training loop. 
│   ├── run_language_modeling.py  # training code (contains data loading, model loading and calls trainer)
│   ├── gen.py                    # high-level scripts to decode. 
│   └── run_generation.py         # decoding code. 
│
├── seq2seq                       # Code for encoder-decoder architecture
│   ├── train_bart.py             # high-level scripts to train.
│   ├── prefixTuning.py           # code that implements prefix-tuning.
│   ├── finetune.py               # training code (contains data loading, model loading, and calls trainer)   
│   ├── lightning_base.py         # helper code
│   ├── utils.py                  # helper code
│   └── callbacks.py              # helper code
└── ...
```

### train_bart.py  
- training 관련 모든 argument 및 hyper-parameter 세팅
- pre-train, fine-tuning, prefix-tuning, ... 에 맡는 commandline 정의

### finetune.py
- 실제 학습을 주관
- 3가지 class
    - class PrefixSummarizationModule(PrefixTransformer) 
    - class SummarizationModule(BaseTransformer)
    - class TranslationModule(SummarizationModule)

### lightning_base.py
- "finetune.py"에서 사용하는 클래스들인 PrefixTransformer, BaseTransformer, ... 이 pytorch-lightning으로 구현됨
- 3가지 class
    - class OurModelCheckPoint(pl.callbacks.ModelCheckpoint)
    - class PrefixTransformer(pl.LightningModule)
        -  PrefixTuning에 맞게 transformer 모델 설정.
        -  Pre-trained 모델 불러오거나 새로 init 
        -  prefixModel을 위해 "prefixTuning.py"의 PrefixTuning 클래스 사용 

    - class BaseTransformer(pl.LightningModule)
    - class LoggingCallback(pl.Callback)

### prefixTuning.py
- prefixTransformer 클래스에서 prefixModel을 시작하기 위해 사용
- 논문에서 설명한 prefix-tuning 방법이 HuggingFace Transformer와 pytorch를 통해 구현됨

- 2가지 클래스
    - PrefixEmbTuning(GPT2PreTrainedModel)
        - 
    - PrefixTuning(PretrainedBartModel)
        - BART(seq2seq Model)
        - optim_prefix일때, deep_param 인지, low_data_init이 몇인지 등 다양한 케이스로 나뉘어 init
        - optim_prefix일때, de