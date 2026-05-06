## 🛡️ WAF-KD Research Contribution

### 1. Contribution Type
Check the box that applies:
- [ ] 🧪 **Research Code** (Data collection, normalization, model training, etc.)
- [ ] 📝 **Documentation** (README updates, technical specifications, research logs)
- [ ] 🛠️ **Infrastructure/Tooling** (Makefile updates, CI/CD, environment fixes)
- [ ] 🐞 **Bug Fix**

---

### 2. Technical Description
**Summary of Changes:**
Provide a concise technical description of what this PR accomplishes. Avoid analogies; use correct terminology (e.g., "Modified the tokenizer to handle hex-encoded SQLi payloads" instead of "Fixed how the system reads data").

**Related Project Stage:**
(e.g., Stage 1: Data Normalization)

---

### 3. Validation Checklist
**For Research Code:**
- [ ] Executed `make lint` and resolved all PEP8/Ruff issues.
- [ ] Executed `make typecheck` and resolved all Mypy errors.
- [ ] Verified stage output (e.g., data is saved in the correct `/data/` directory).
- [ ] No data leakage or persistent static credentials identified.

**For Documentation:**
- [ ] Verified all Markdown links are functional.
- [ ] Technical terms are used accurately.
- [ ] No em-dashes used (as per project style guide).
- [ ] Spelling and grammar checked.

---

### 4. Evidence of Work
**Execution Logs / Screenshots:**
Paste the output of your terminal showing successful `make` commands or provide a brief log of your findings.
```text
[Paste terminal output here]