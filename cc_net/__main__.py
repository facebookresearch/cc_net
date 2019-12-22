# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import func_argparse

import cc_net.mine
import cc_net.minify


def main():
    parser = func_argparse.multi_argparser(
        mine=cc_net.mine.get_main_parser(),
        reproduce=func_argparse.func_argparser(cc_net.minify.reproduce),
    )
    func_argparse.parse_and_call(parser)


if __name__ == "__main__":
    main()
