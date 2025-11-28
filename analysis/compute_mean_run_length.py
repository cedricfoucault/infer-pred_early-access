"""
Compute the effective mean run length of a run length distribution governed
by a given probability of a change point occurring, and a given min and max
run length. This is equal to the mean of a truncated geometric distribution
with the given probability parameter for the geomtric distribution, and
the given min and max parameters for truncation.
"""
import argparse
import scipy.stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-min", "--min_run_length", default=5)
    parser.add_argument("-max", "--max_run_length", default=40)
    parser.add_argument("-p", "--p_cp", default=1/16)
    args = parser.parse_args()
    # min_run_length_dflt = 4
    # max_run_length_dflt = 40
    # p_cp_dflt = 0.05
    # params = {
    #     "min_run_length": args.min_run_length,
    #     "max_run_length": args.max_run_length,
    #     "p_cp": args.p_cp
    # }
    n_samples = int(1e6)
    samples = scipy.stats.geom.rvs(args.p_cp, size=n_samples)
    mask = (samples >= args.min_run_length) & (samples <= args.max_run_length)
    samples_truncated = samples[mask]
    print(f"parameters: {args}")
    print(f"mean run length without truncation {samples.mean():.2f}")
    print(f"mean run length with truncation {samples_truncated.mean():.2f}")