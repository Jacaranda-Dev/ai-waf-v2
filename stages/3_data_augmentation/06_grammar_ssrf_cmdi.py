"""Stage 2.4 — SSRF/CMDi grammar generation."""
import argparse, subprocess, sys
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__":
    args = parse_args()
    for attack in ["ssrf", "cmdi"]:
        subprocess.run([sys.executable, "stages/3_data_augmentation/03_grammar_sqli.py",
                        "--config", args.config, "--attack", attack], check=True)