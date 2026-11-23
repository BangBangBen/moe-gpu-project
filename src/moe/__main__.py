"""Generate a fixture or evaluate the v0 reference."""

import argparse

from .format import read_input, write_input, write_output
from .reference import reference
from .workloads import generate


def main():
    """Parse command-line options to generate inputs or run the v0 reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("output")
    for flag, default in (("tokens", 37), ("experts", 5), ("k", 13), ("n", 7)):
        gen.add_argument(f"--{flag}", type=int, default=default)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--routing", choices=["balanced", "uniform", "hotspot"], default="balanced")
    gen.add_argument("--counts", type=int, nargs="+", help="Explicit expert counts; overrides routing")
    ref = commands.add_parser("reference")
    ref.add_argument("input")
    ref.add_argument("output")
    ref.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    if args.command == "generate":
        write_input(args.output, generate(args.tokens, args.experts, args.k, args.n,
                                         routing=args.routing, seed=args.seed, counts=args.counts))
    else:
        write_output(args.output, reference(read_input(args.input), args.device))


if __name__ == "__main__":
    main()
