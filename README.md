# Ghost Assimilation Relay (GAR)

[**under construction**]

> Persona-driven orchestration framework for modular LLM reasoning and adaptive style control.

---

## 🧠 Overview
**Ghost Assimilation Relay (GAR)** is a modular orchestration layer designed to coordinate
persona-based reasoning, context condensation, and style modulation across multiple LLM backends.

**Key Features**
- Persona layer for semantic and emotional modulation
- Context layer for memory condensation and relevance filtering
- Gateway relay for multi-agent orchestration


---

## 🧩 Directory Layout
```
/home/<user>
    └── 📁 gar-llm
        ├── 📄 .git
        ├── 📄 .gitignore
        ├── 📄 CITATION.cff
        ├── 📄 LICENSE
        ├── 📄 NOTICE
        ├── 📄 README.md
        ├── 📄 pyproject.toml
        └── 📁 src
            └── 📁 garllm
                ├── 📄 __init__.py
                ├── 📁 context_layer
                │   ├── 📄 cleaner.py
                │   ├── 📄 condenser.py
                │   ├── 📄 retriever.py
                │   ├── 📄 semantic_condenser.py
                │   └── 📄 thought_profiler.py
                ├── 📁 gateway
                │   └── 📄 relay_server.py
                ├── 📁 persona_layer
                │   └── 📄 persona_generator.py
                ├── 📁 style_layer
                │   ├── 📄 context_controller.py
                │   ├── 📄 response_modulator.py
                │   ├── 📄 style_modulator.py
                │   └── 📄 test_emotions.sh
                └── 📁 utils
                    ├── 📄 env_utils.py
                    ├── 📄 llm_client.py
                    ├── 📄 logger.py
                    └── 📄 vllm_client.py

```

---

## 🚀 Quick Start (Docker)
cd webui
docker compose up -d

Access at [http://localhost:3000](http://localhost:3000)

---

## ⚖️ License
- **Code:** Apache License 2.0  
- **Documents & Personas:** CC BY-NC-SA 4.0  

See [LICENSE](./LICENSE) for details.

---

## 📖 Citation
If you use GAR in academic work:

@software{Smashir_GAR_2025,
  author = {Smashir},
  title = {Ghost Assimilation Relay (GAR)},
  year = {2025},
  url = {https://github.com/Smashir/gar-llm}
}

---

## 🧩 Credits
Developed collaboratively for LLM orchestration and persona research.
