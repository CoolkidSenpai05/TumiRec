# Preparing encoder weights (manual, opt-in only)

No weights have been downloaded by this project. If you choose to obtain them,
install sentence-transformers in your environment and explicitly run:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # network download
model.save("artifacts/bge-small-en-v1.5")
```

Then pass `--model artifacts/bge-small-en-v1.5` to the S1 CLI. The production
ContentEncoder always sets local_files_only=True and never permits downloads.
For reproducible experiments pin the model revision when obtaining weights and
use an immutable local directory. The cache identity includes the supplied model
name/path, but cannot detect weights modified in place under the same name.
