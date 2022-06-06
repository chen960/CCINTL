Run Code

消融实验（只使用CMAM）

```
CUDA_VISIBLE_DEVICES=0,1 python nometa_qhard.py -d market1501 --eps 0.5 --epochs 60  --temp0 0.05 --temp1 0.05 --temp2 1.0  --am --m 0.1 --symmetric 
CUDA_VISIBLE_DEVICES=2,3 python nometa_qhard.py -d dukemtmc --eps 0.6 --epochs 60  --temp0 0.05 --temp1 0.05 --temp2 1.0  --am --m 0.1
CUDA_VISIBLE_DEVICES=1,2 python nometa_qhard.py -d msmt17 --eps 0.6 --epochs 60  --temp0 0.05 --temp1 0.05 --temp2 1.0  --am --m 0.1

```

消融实验（使用CMAM+CMNT）

```
CUDA_VISIBLE_DEVICES=0,1 python nometa_qhard.py -d market1501 --eps 0.5 --epochs 60  --temp0 0.05 --temp1 0.05 --temp2 1.0  --am --m 0.1 --symmetric 
CUDA_VISIBLE_DEVICES=2,3 python nometa_qhard.py -d dukemtmc --eps 0.6 --epochs 60  --temp0 0.05 --temp1 0.05 --temp2 1.0  --am --m 0.1 --symmetric
CUDA_VISIBLE_DEVICES=1,2 python nometa_qhard.py -d msmt17 --eps 0.6 --epochs 60  --temp0 0.05 --temp1 0.05 --temp2 1.0  --am --m 0.1 --symmetric
```

消融实验（使用CMAM+CMML）

```
# market 
CUDA_VISIBLE_DEVICES=2,3 python meta_qhard.py -d market1501 --eps 0.5 --epochs 60  --temp0 0.05 --temp1 0.05 --temp2 1.0  --am --m 0.1 
# dukemtmc
CUDA_VISIBLE_DEVICES=0,1 python meta_qhard.py -d dukemtmc --eps 0.6 --epochs 60   --temp0 0.05 --temp1 0.05 --temp2 1.0  --am --m 0.1
# msmt17
CUDA_VISIBLE_DEVICES=0,1 python meta_qhard.py -d msmt17 --eps 0.6 --epochs 60  --temp0 0.02 --temp1 0.05 --temp2 1.0  --am --m 0.1 
```



最终实验（使用CMAM+CMML+CMNT）

```
# market 
CUDA_VISIBLE_DEVICES=2,3 python meta_qhard.py -d market1501 --eps 0.5 --epochs 60  --temp0 0.05 --temp1 0.05 --temp2 1.0  --am --m 0.1  --lamn 0.5 --symmetric 
# dukemtmc
CUDA_VISIBLE_DEVICES=0,1 python meta_qhard.py -d dukemtmc --eps 0.6 --epochs 60   --temp0 0.05 --temp1 0.05 --temp2 1.0  --am --m 0.1 --lamn 0.5 --symmetric 
# msmt17
CUDA_VISIBLE_DEVICES=0,1 python meta_qhard.py -d msmt17 --eps 0.6 --epochs 60  --temp0 0.02 --temp1 0.05 --temp2 1.0  --am --m 0.1  --lamn 0.5 --symmetric 
```

实验结果在Tabel I ，II ， Onenote中。
