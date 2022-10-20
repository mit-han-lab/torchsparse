VERSION=2.0.0

TAG=$(python -c "
import torch

tag, _ = ('torch' + torch.__version__).rsplit('.', 1)

if torch.cuda.is_available():
  tag += 'cu' + torch.version.cuda
else:
  tag += 'cpu'

print(tag.replace('.', ''))
")

pip install torchsparse==${VERSION}+${TAG} -i https://pypi.hanlab.ai/simple
