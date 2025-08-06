# ✨ Oolit - Offline Chat for Python

**Oolit** is a Python-based application that provides a fully offline chat interface.

- **Chat without internet**: Interact with a local language model.
- **Easy to use**: Just import and run `oolit()`.

### Prerequisites for `oolit()`
To use the `oolit()` offline chat feature, please ensure the following:

1.  **Ollama Installation**: Ollama must be installed and running on your machine. You can download it from [ollama.com](https://ollama.com/).
2.  **TinyLlama Model**: `oolit()` utilizes a pre-trained TinyLlama model. After installing Ollama, pull the TinyLlama model by running the following command in your terminal:
    ```bash
    ollama pull tinyllama
    ```

### Example
```python
from oolit import oolit

oolit()
```
