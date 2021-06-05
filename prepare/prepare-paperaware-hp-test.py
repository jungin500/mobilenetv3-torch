import subprocess
import random
import os

# Paper-aware Hyperparameter test script
# 2021/06/04 LimeOrangePie.

if __name__ == '__main__':
    subprocess_env = os.environ.copy()
    subprocess_env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    epochs = 100

    for tryidx in range(epochs):
        lr_search_space = [0.00005, 0.0002]

        lr_volume = lr_search_space[1] - lr_search_space[0]
        lr_margin = lr_search_space[0]

        lr_int = random.randrange(epochs)
        lr_float = lr_margin + (lr_int / epochs * lr_volume)
        run_name = "R4-TestRun-LR%.5f-OneEpoch" % (lr_float, )

        print("Try %d of %d: learning rate %.5f, 5 epochs each" % (tryidx, epochs, lr_float))
        # EnvVar for deterministic CUDA algorithms
        result = subprocess.run(('python prepare-paperaware.py -g -b 1536 -e 5 -p 8 -l %.5f -rn %s' % (lr_float, run_name)).split(' '),
            env=subprocess_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode != 0:
            print("Error - exited abnormally")