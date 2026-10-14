"""Optional command-line launcher."""
import argparse

def main():
    parser = argparse.ArgumentParser(description="AI Medical Image Analysis project")
    parser.add_argument("action", choices=["check", "train", "evaluate"])
    action = parser.parse_args().action
    if action == "check":
        from src.check_dataset import main as run
    elif action == "train":
        from src.train import main as run
    else:
        from src.evaluate import main as run
    run()

if __name__ == "__main__":
    main()
