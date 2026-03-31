# MLA Decode (multi_latent_attention_mix)

## Build and run

```bash
bash ./compile.sh
python ./test_mla_decode.py
```

Expected output (example):
```
multi_latent_attention_mix call finished.
output mean abs: 0.195068
output finite: True
output changed: True
mean abs diff vs ref: 0.000103
max abs diff vs ref: 0.001953
tor example: 0.041667
```