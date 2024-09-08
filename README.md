# torch-cpp-playground
Repo for studying on pytorch custom kernel implementation in cpp/CUDA.

## Build & Run
### torch built from source
```bash
# pytorch/
#   $CURRENT_REPO/
#       setup.py
#       A.cpp
#       B.cu
cd pytorch/$(CURRENT_REPO)/
python setup.py install
```
### torch from pip
```bash
# venv/lib/.../torch/
#   $CURRENT_REPO/
#       setup.py
#       A.cpp
#       B.cu
cd .../torch/$(CURRENT_REPO)/
python setup.py install
```

### Run
```python
import torch
import lltm_cpp # PackageName defined in setup.py

lltm_cpp.forward
```

