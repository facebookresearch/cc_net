# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import func_argparse

import cc_net.mine
import cc_net.minify


def main():
    parser = argparse.ArgumentParser(description=cc_net.mine.__doc__, add_help=True)
    subparsers = parser.add_subparsers()

    cc_net.mine.get_main_parser(subparsers)
    func_argparse.add_fn_subparser(cc_net.minify.reproduce, subparsers)

    parsed_args = vars(parser.parse_args())
    command = parsed_args.pop("__command", None)
    if not command:
        return parser.print_usage()
    command(**parsed_args)


if __name__ == "__main__":
    main()
