
"""Stage 2.2 — XSS grammar generation. Delegates to 03_grammar_sqli.py."""
import argparse, subprocess, sys
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__":
    args = parse_args()
    subprocess.run([sys.executable, "stages/3_data_augmentation/03_grammar_sqli.py",
                    "--config", args.config, "--attack", "xss"], check=True)

