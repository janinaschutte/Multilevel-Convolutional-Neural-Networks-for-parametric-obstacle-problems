import argparse, os, json, time
import multiprocessing as mp
import numpy as np

from .sobol import i4_sobol_generate_std_normal, i4_sobol_generate

def log(*args, **kwargs):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()), *args, **kwargs)

class Sampler(object):
    """parameter sampler for different distributions (uniform/normal) and QMC (Sobol)"""
    def __init__(self, _info):
        dimension    = _info['expansion']['size']
        distribution = _info['sampling']['distribution']
        strategy     = _info['sampling']['strategy']

        print("in sampler")
        if _info['problem']['name'] == 'obstacle' and _info['expansion']['obstacle_parameters']:
            self.add_obstacle_samples = True
            dimension -= 1
        else:
            self.add_obstacle_samples = False

        if distribution not in ['normal', 'uniform']:
            raise ValueError("distribution must be 'uniform' or 'normal'")
        if strategy not in ['random', 'sobol']:
            raise ValueError("strategy must be 'random' or 'sobol'")

        if distribution == 'normal' and strategy == 'random':
            def generator(_numSamples):
                return np.random.randn(_numSamples, dimension)
        elif distribution == 'normal' and strategy == 'sobol':
            self.offset = 1
            def generator(_numSamples):
                samples = i4_sobol_generate_std_normal(dim_num=dimension, n=_numSamples, skip=self.offset)
                self.offset += _numSamples
                return samples
        elif distribution == 'uniform' and strategy == 'random':
            def generator(_numSamples):
                return 2*np.random.rand(_numSamples, dimension)-1
        elif distribution == 'uniform-stefan':
            def generator(_numSamples):
                return (np.random.rand(_numSamples,_info["expansion"]["size"]) - 0.4)*0.035 - 1
        else:
            self.offset = 1
            def generator(_numSamples):
                samples = 2*i4_sobol_generate(dim_num=dimension, n=_numSamples, skip=self.offset)-1
                self.offset += _numSamples
                return samples
        self.generator = generator


    def __call__(self, _numSamples):
        out = self.generator(_numSamples)
        if self.add_obstacle_samples:
            out_obstacle = (2 * np.random.rand(_numSamples, 1) - 1) * 0.01 - 0.035
            out = np.concatenate((out, out_obstacle), axis=1)
        return out


def load_problem_and_sampler(info):
    # load problem
    problemName = info['problem']['name']
    log(f"Loading problem: {problemName}")
    Problem = __import__(f"code.ParamPDE.problem.{problemName}", fromlist=["Problem"]).Problem
    problem = Problem(info)

    # export meshes
    import pickle
    for l, mesh in enumerate(problem.mesh):
        pickle.dump(mesh, open(os.path.join(info['problemDir'], f'mesh-{l}.pkl'), 'wb'))

    # load sampler
    log(f"Loading sampler: {info['sampling']['strategy']}-{info['sampling']['distribution']} (dimension: {info['expansion']['size']})")
    sampler = Sampler(info)

    return problem, sampler


class NPZStorageDirectory(object):
    def __init__(self, directory):
        self.filePath = os.path.join(directory, '{fileCount}-{level}.npz')
        self.fileCount = 0

    def __lshift__(self, keysAndValues):
        for level, ul in enumerate(keysAndValues['U']):
            fileName = self.filePath.format(fileCount=self.fileCount, level=level)
            log(f"Saving samples: '{fileName}'")
            np.savez_compressed(fileName, keysAndValues['ys'], ul)
        self.fileCount += 1


def compute_batch(batchSize, storage, sampler):
    # get parameter samples
    ys = sampler(batchSize)
    if not np.all(np.isfinite(ys)):
        raise RuntimeError(f"Invalid value encountered in output of sampler: {np.count_nonzero(~np.all(np.isfinite(ys), axis=1))} samples are not finite.")

    # multiprocess PDE solution computation for parameter samples
    pool = mp.Pool()
    results = pool.map_async(problem.solution, ys).get()
    pool.close()
    pool.join()
    # collect solutions for all levels
    U = []
    for l in range(len(results[0])):
        ul = np.array([r[l] for r in results])
        U.append(ul)
    # sanity check
    # if not np.all(np.isfinite(ul)):
    #     raise RuntimeError(f"Invalid value encountered in output of problem.solution: {np.count_nonzero(~np.all(np.isfinite(us), axis=1))} samples are not finite.")

    # export parameters and solutions
    if len(ys) != len(U[0]):
        raise RuntimeError(f"Number of Samples and number of solutions differ: {len(ys)} != {len(U)}")
    storage << dict(ys=ys, U=U)

def isInt(s):
    try:
        int(s)
        return True
    except:
        return False

if __name__=='__main__':
    descr = """Sample solutions for the given problem."""
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('PROBLEM', help='path to the directory where the samples will be stored. The problem specification is assumed to lie in (PROBLEM/parameters.json)')
    parser.add_argument('SAMPLES', type=int, help='number of samples to compute')
    parser.add_argument('-b', '--batch-size', dest='BATCH_SIZE', type=int, default=100, help='size of each batch (default: 100)')
    args = parser.parse_args()

    # check problem directory and specification file
    problemDir = args.PROBLEM
    if not os.path.isdir(problemDir):
        raise IOError(f"PROBLEM '{problemDir}' is not a directory")

    problemDirContents = os.listdir(problemDir)
    if not 'parameters.json' in problemDirContents:
        raise IOError(f"'{problemDir}' does not contain a 'parameters.json'")
    for fileName in problemDirContents:
        if fileName.endswith('.npz') and isInt(fileName[:-4]):
            raise IOError(f"'{problemDir}' already contains other data ('{problemDir}/{fileName}')")

    problemFile = f"{problemDir}/parameters.json"
    try:
        with open(problemFile, 'r') as f:
            problemInfo = json.load(f)
    except FileNotFoundError:
        raise IOError(f"Cannot read file '{problemFile}'")
    except json.JSONDecodeError:
        raise IOError(f"'{problemFile}' is not a valid JSON file")

    # setup sampler
    problemInfo['problemDir'] = problemDir
    problem, sampler = load_problem_and_sampler(problemInfo)
    storage = NPZStorageDirectory(problemDir)

    number = args.SAMPLES
    batchSize = args.BATCH_SIZE
    batchNumber = 0
    while number > 0:
        log(f"Computing batch: {batchNumber}")
        compute_batch(min(batchSize, number), storage, sampler)
        number -= batchSize
        batchNumber += 1
