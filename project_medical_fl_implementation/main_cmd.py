import subprocess

commands = [
    "fluke federation config/IID/LR/exp-iid-LR.yml config/IID/LR/exp-iid-LR.yml",
    "fluke federation config/IID/MLP/exp-iid-MLP.yml config/IID/MLP/exp-iid-MLP.yml",
    "fluke federation config/IID/SVM/exp-iid-SVM.yml config/IID/SVM/exp-iid-SVM.yml",

    "fluke federation config/NON-IID/0.5/LR/exp-non-iid-LR.yml config/NON-IID/0.5/LR/exp-non-iid-LR.yml",
    "fluke federation config/NON-IID/0.5/MLP/exp-non-iid-MLP.yml config/NON-IID/0.5/MLP/exp-non-iid-MLP.yml",
    "fluke federation config/NON-IID/0.5/SVM/exp-non-iid-SVM.yml config/NON-IID/0.5/SVM/exp-non-iid-SVM.yml",

    "fluke federation config/NON-IID/5/LR/exp-non-iid-LR.yml config/NON-IID/5/LR/exp-non-iid-LR.yml",
    "fluke federation config/NON-IID/5/MLP/exp-non-iid-MLP.yml config/NON-IID/5/MLP/exp-non-iid-MLP.yml",
    "fluke federation config/NON-IID/5/SVM/exp-non-iid-SVM.yml config/NON-IID/5/SVM/exp-non-iid-SVM.yml",

    "fluke federation config/NON-IID/100/LR/exp-non-iid-LR.yml config/NON-IID/100/LR/exp-non-iid-LR.yml",
    "fluke federation config/NON-IID/100/MLP/exp-non-iid-MLP.yml config/NON-IID/100/MLP/exp-non-iid-MLP.yml",
    "fluke federation config/NON-IID/100/SVM/exp-non-iid-SVM.yml config/NON-IID/100/SVM/exp-non-iid-SVM.yml",

    "fluke federation config/NOISE/0.1/LR/exp-non-iid-LR.yml config/NOISE/0.1/LR/exp-non-iid-LR.yml", 
    "fluke federation config/NOISE/0.1/MLP/exp-non-iid-MLP.yml config/NOISE/0.1/MLP/exp-non-iid-MLP.yml",
    "fluke federation config/NOISE/0.1/SVM/exp-non-iid-SVM.yml config/NOISE/0.1/SVM/exp-non-iid-SVM.yml", 

    "fluke federation config/NOISE/0.5/LR/exp-non-iid-LR.yml config/NOISE/0.5/LR/exp-non-iid-LR.yml", 
    "fluke federation config/NOISE/0.5/MLP/exp-non-iid-MLP.yml config/NOISE/0.5/MLP/exp-non-iid-MLP.yml", 
    "fluke federation config/NOISE/0.5/SVM/exp-non-iid-SVM.yml config/NOISE/0.5/SVM/exp-non-iid-SVM.yml",

    "fluke federation config/NOISE/1.2/LR/exp-non-iid-LR.yml config/NOISE/1.2/LR/exp-non-iid-LR.yml", 
    "fluke federation config/NOISE/1.2/MLP/exp-non-iid-MLP.yml config/NOISE/1.2/MLP/exp-non-iid-MLP.yml", 
    "fluke federation config/NOISE/1.2/SVM/exp-non-iid-SVM.yml config/NOISE/1.2/SVM/exp-non-iid-SVM.yml"]

def run_command(command):
    print(f"Executing : {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Er : {e}")

if __name__ == "__main__":
    for cmd in commands:
        run_command(cmd)
